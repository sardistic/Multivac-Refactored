from __future__ import annotations

from typing import Any, Dict, Optional

from services.tool_handlers import TOOL_HANDLERS, list_tool_summaries


async def execute_tool(
    name: str,
    args: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    tool_specs=None,
):
    if name == "list_available_tools":
        return list_tool_summaries(tool_specs=tool_specs)

    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {"ok": False, "error": f"unknown_tool: {name}"}

    call_args = dict(args or {})
    if context:
        call_args["_context"] = context
    return await handler(call_args)
