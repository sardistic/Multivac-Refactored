import tempfile
import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from providers import openai_messages
from services.tool_control import ToolControlWorker
from services.tool_runtime import ToolModuleLoader, ToolRegistry, ToolRegistryError


def _spec(name: str, description: str = "test tool") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


def _module_source(version: str, value: str, name: str = "live_echo") -> str:
    return f'''TOOL_VERSION = {version!r}
TOOL_SPECS = [
    {{
        "type": "function",
        "function": {{
            "name": {name!r},
            "description": "hotloaded echo",
            "parameters": {{"type": "object", "properties": {{}}, "required": []}},
        }},
    }}
]

async def handle(args):
    return {{"ok": True, "value": {value!r}}}

TOOL_HANDLERS = {{{name!r}: handle}}
'''


class ToolRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_snapshot_keeps_matching_handler_after_reload(self):
        registry = ToolRegistry()

        async def v1(_args):
            return "v1"

        async def v2(_args):
            return "v2"

        registry.replace_source("example", [_spec("echo")], {"echo": v1}, version="1")
        original = registry.snapshot()
        registry.replace_source("example", [_spec("echo")], {"echo": v2}, version="2")

        self.assertEqual(await registry.execute("echo", {}, snapshot=original), "v1")
        self.assertEqual(await registry.execute("echo", {}), "v2")
        self.assertGreater(registry.snapshot().generation, original.generation)

    async def test_unload_and_rollback_are_atomic(self):
        registry = ToolRegistry()

        async def handler(_args):
            return {"ok": True}

        registry.replace_source("example", [_spec("echo")], {"echo": handler})
        registry.unload_source("example")
        self.assertEqual(
            await registry.execute("echo", {}),
            {"ok": False, "error": "unknown_tool: echo"},
        )

        registry.rollback_source("example")
        self.assertEqual(await registry.execute("echo", {}), {"ok": True})

    def test_specs_and_handlers_must_match(self):
        registry = ToolRegistry()
        with self.assertRaisesRegex(ToolRegistryError, "must match"):
            registry.replace_source("bad", [_spec("declared")], {"other": lambda _args: None})


class ToolModuleLoaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_load_reload_and_rollback_module(self):
        registry = ToolRegistry()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_path = root / "echo.py"
            module_path.write_text(_module_source("1", "first"), encoding="utf-8")
            loader = ToolModuleLoader(registry, root)

            loaded = loader.activate("echo.py")
            first_snapshot = registry.snapshot()
            self.assertEqual(loaded["version"], "1")
            self.assertEqual(
                await registry.execute("live_echo", {}),
                {"ok": True, "value": "first"},
            )

            module_path.write_text(_module_source("2", "second"), encoding="utf-8")
            reloaded = loader.activate("echo.py")
            self.assertEqual(reloaded["version"], "2")
            self.assertEqual(
                await registry.execute("live_echo", {}),
                {"ok": True, "value": "second"},
            )
            self.assertEqual(
                await registry.execute("live_echo", {}, snapshot=first_snapshot),
                {"ok": True, "value": "first"},
            )

            loader.rollback("echo.py")
            self.assertEqual(
                await registry.execute("live_echo", {}),
                {"ok": True, "value": "first"},
            )

    async def test_builtin_override_requires_explicit_permission(self):
        registry = ToolRegistry()

        async def builtin(_args):
            return "builtin"

        registry.replace_source("builtin", [_spec("live_echo")], {"live_echo": builtin})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "override.py").write_text(
                _module_source("1", "override"), encoding="utf-8"
            )
            loader = ToolModuleLoader(registry, root)
            with self.assertRaisesRegex(ToolRegistryError, "exactly match active conflicts"):
                loader.activate("override.py")

            (root / "override.py").write_text(
                "TOOL_OVERRIDES = ('live_echo',)\n" + _module_source("1", "override"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ToolRegistryError, "explicit activation authorization"):
                loader.activate("override.py")
            loader.activate("override.py", allow_overrides=True)
            self.assertEqual(
                await registry.execute("live_echo", {}),
                {"ok": True, "value": "override"},
            )
            loader.unload("override.py")
            self.assertEqual(await registry.execute("live_echo", {}), "builtin")

    def test_path_must_stay_beneath_trusted_root(self):
        registry = ToolRegistry()
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "trusted"
            root.mkdir()
            (parent / "outside.py").write_text(_module_source("1", "bad"), encoding="utf-8")
            loader = ToolModuleLoader(registry, root)
            with self.assertRaisesRegex(ToolRegistryError, "escapes"):
                loader.activate("../outside.py")


class ProviderToolSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_request_uses_live_registry_schema(self):
        registry = ToolRegistry()

        async def handler(_args):
            return {"ok": True}

        registry.replace_source(
            "dynamic",
            [_spec("fresh_tool", "loaded without restarting")],
            {"fresh_tool": handler},
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="ok", tool_calls=None),
                )
            ]
        )
        create = AsyncMock(return_value=response)
        with patch.object(openai_messages, "USE_RESPONSES", False), patch.object(
            openai_messages, "get_tool_snapshot", return_value=registry.snapshot()
        ), patch.object(
            openai_messages,
            "_create_chat_completion_with_token_fallback",
            create,
        ):
            result = await openai_messages.generate_openai_messages_response_with_tools(
                [{"role": "user", "content": "use it"}], model="gpt-5.6-terra"
            )

        self.assertEqual(result, "ok")
        self.assertEqual(
            create.await_args.kwargs["tools"][0]["function"]["name"],
            "fresh_tool",
        )


class ToolControlWorkerTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _write_request(control: Path, request_id: str, operations: list[dict]) -> None:
        control.mkdir(parents=True, exist_ok=True)
        (control / "request.json").write_text(
            json.dumps(
                {"version": 1, "request_id": request_id, "operations": operations}
            ),
            encoding="utf-8",
        )

    async def test_activation_persists_and_restores_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            artifacts = base / "artifacts"
            control = base / "control"
            module = artifacts / "proposal-1" / "live_tools" / "echo.py"
            module.parent.mkdir(parents=True)
            payload = _module_source("1", "persisted").encode()
            module.write_bytes(payload)
            operation = {
                "action": "activate",
                "source_id": "hotload:live_tools/echo.py",
                "relative_path": "proposal-1/live_tools/echo.py",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "proposal_id": 1,
            }
            self._write_request(control, "request-1", [operation])

            registry = ToolRegistry()
            worker = ToolControlWorker(ToolModuleLoader(registry, artifacts), control)
            result = worker.process_once()
            self.assertTrue(result["ok"])
            self.assertEqual(
                await registry.execute("live_echo", {}),
                {"ok": True, "value": "persisted"},
            )

            restarted = ToolRegistry()
            restored = ToolControlWorker(
                ToolModuleLoader(restarted, artifacts), control
            ).restore_active()
            self.assertEqual(restored["errors"], [])
            self.assertEqual(restored["restored"], ["hotload:live_tools/echo.py"])
            self.assertEqual(
                await restarted.execute("live_echo", {}),
                {"ok": True, "value": "persisted"},
            )

    async def test_failed_batch_rolls_back_prior_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            artifacts = base / "artifacts"
            control = base / "control"
            old_module = artifacts / "old" / "live_tools" / "echo.py"
            new_module = artifacts / "new" / "live_tools" / "echo.py"
            old_module.parent.mkdir(parents=True)
            new_module.parent.mkdir(parents=True)
            old_payload = _module_source("1", "old").encode()
            new_payload = _module_source("2", "new").encode()
            old_module.write_bytes(old_payload)
            new_module.write_bytes(new_payload)
            source_id = "hotload:live_tools/echo.py"

            registry = ToolRegistry()
            worker = ToolControlWorker(ToolModuleLoader(registry, artifacts), control)
            self._write_request(
                control,
                "initial",
                [{
                    "action": "activate",
                    "source_id": source_id,
                    "relative_path": "old/live_tools/echo.py",
                    "sha256": hashlib.sha256(old_payload).hexdigest(),
                }],
            )
            self.assertTrue(worker.process_once()["ok"])

            self._write_request(
                control,
                "broken-update",
                [
                    {
                        "action": "activate",
                        "source_id": source_id,
                        "relative_path": "new/live_tools/echo.py",
                        "sha256": hashlib.sha256(new_payload).hexdigest(),
                    },
                    {
                        "action": "activate",
                        "source_id": "hotload:live_tools/missing.py",
                        "relative_path": "new/live_tools/missing.py",
                        "sha256": "0" * 64,
                    },
                ],
            )
            result = worker.process_once()
            self.assertFalse(result["ok"])
            self.assertEqual(
                await registry.execute("live_echo", {}),
                {"ok": True, "value": "old"},
            )


if __name__ == "__main__":
    unittest.main()
