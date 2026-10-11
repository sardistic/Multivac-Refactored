"""One cached image per Discord durable entitlement, with restart recovery."""

import asyncio
import io
import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import discord
from PIL import Image

from services.character_catalog import CharacterCatalog, character_prompt
from services.premium import PremiumError, entitlement_valid
from services import usage_costs
from services.url_utils import fetch_url_bytes_async

logger = logging.getLogger("discord_bot")
CHARACTER_SKU_ID = 1558749059609665566
CHARACTER_MODEL = "gpt-image-2.5-sunburst"


class CharacterStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS character_choices(user_id TEXT PRIMARY KEY, name TEXT);
                CREATE TABLE IF NOT EXISTS character_revocations(entitlement_id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS character_renders(
                    entitlement_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, sku_id TEXT NOT NULL,
                    requested_name TEXT, character_json TEXT, image BLOB,
                    state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                    lease TEXT, lease_until REAL NOT NULL DEFAULT 0,
                    delivery_attempted INTEGER NOT NULL DEFAULT 0
                );
            """)
        os.chmod(self.path, 0o600)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def choice(self, user_id, name):
        if name is not None and (not name.strip() or len(name) > 120):
            raise PremiumError("Use a character name of 1–120 characters, or omit it for random.")
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO character_choices VALUES(?,?)", (str(user_id), name.strip() if name else None))

    def row(self, entitlement_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM character_renders WHERE entitlement_id=?", (str(entitlement_id),)).fetchone()
            return dict(row) if row else None

    def reserve(self, entitlement):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM character_revocations WHERE entitlement_id=?", (str(entitlement.id),)).fetchone():
                raise PremiumError("This purchase was removed or refunded.")
            name = conn.execute("SELECT name FROM character_choices WHERE user_id=?", (str(entitlement.user_id),)).fetchone()
            conn.execute("INSERT OR IGNORE INTO character_renders(entitlement_id,user_id,sku_id,requested_name) VALUES(?,?,?,?)",
                         (str(entitlement.id), str(entitlement.user_id), str(entitlement.sku_id), name[0] if name else None))
            row = conn.execute("SELECT user_id,sku_id,state FROM character_renders WHERE entitlement_id=?", (str(entitlement.id),)).fetchone()
            if row[0] != str(entitlement.user_id) or row[1] != str(entitlement.sku_id) or row[2] == "revoked":
                raise PremiumError("This purchase is unavailable to this account.")

    def claim(self, entitlement_id, now):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM character_renders WHERE entitlement_id=?", (str(entitlement_id),)).fetchone()
            if row["state"] == "ready":
                return None
            if row["state"] == "revoked":
                raise PremiumError("This purchase was removed or refunded.")
            if row["lease_until"] > now:
                raise PremiumError("Your character is already being prepared. Use /character again shortly.")
            if row["attempts"] >= 3:
                raise PremiumError("Rendering could not finish after three attempts. Contact bot@sardistic.com; your purchase is still recorded.")
            lease = uuid.uuid4().hex
            conn.execute("UPDATE character_renders SET state='preparing',lease=?,lease_until=? WHERE entitlement_id=?",
                         (lease, now + 900, str(entitlement_id)))
            return lease

    def update(self, entitlement_id, lease, **fields):
        allowed = {"character_json", "image", "state", "lease_until"}
        if not fields or not fields.keys() <= allowed:
            raise ValueError("Invalid render update.")
        with self.connect() as conn:
            assignments = ",".join(key + "=?" for key in fields)
            count = conn.execute("UPDATE character_renders SET " + assignments + " WHERE entitlement_id=? AND lease=? AND state!='revoked'",
                                 (*fields.values(), str(entitlement_id), lease)).rowcount
            if not count:
                raise PremiumError("This render was removed or superseded.")

    def start_render(self, entitlement_id, lease):
        with self.connect() as conn:
            count = conn.execute("UPDATE character_renders SET state='rendering',attempts=attempts+1,lease_until=? WHERE entitlement_id=? AND lease=? AND state='preparing'",
                                 (time.time() + 900, str(entitlement_id), lease)).rowcount
            if not count:
                raise PremiumError("This render was removed or superseded.")

    def delivery_attempt(self, entitlement_id):
        with self.connect() as conn:
            return bool(conn.execute("UPDATE character_renders SET delivery_attempted=1 WHERE entitlement_id=? AND state='ready' AND delivery_attempted=0",
                                     (str(entitlement_id),)).rowcount)

    def revoke(self, entitlement_id):
        with self.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO character_revocations VALUES(?)", (str(entitlement_id),))
            conn.execute("UPDATE character_renders SET state='revoked',image=NULL WHERE entitlement_id=?", (str(entitlement_id),))


class CharacterRenderService:
    def __init__(self, store, *, sku_id=CHARACTER_SKU_ID, catalog=None):
        self.store, self.sku_id = store, sku_id
        self.catalog = catalog or CharacterCatalog()

    async def product(self, client):
        if not self.sku_id:
            raise PremiumError("Character purchases are disabled.")
        sku = next((s for s in await client.fetch_skus() if s.id == self.sku_id), None)
        if sku is None or sku.application_id != client.application_id or sku.type != discord.SKUType.durable:
            raise PremiumError("The character product must be this application's durable one-time SKU.")
        return sku

    async def purchase(self, client, user_id):
        await self.product(client)
        async for ent in client.entitlements(limit=None, user=discord.Object(int(user_id)), skus=[discord.Object(self.sku_id)], exclude_ended=True):
            if self.valid(ent, client.application_id, user_id):
                return ent
        return None

    def valid(self, entitlement, app_id, user_id):
        return (entitlement.user_id is not None and str(entitlement.user_id) == str(user_id)
                and entitlement.guild_id is None and entitlement_valid(entitlement, app_id, self.sku_id, time.time()))

    async def checked(self, client, entitlement_id, user_id):
        try:
            ent = await client.fetch_entitlement(entitlement_id)
        except discord.NotFound as exc:
            self.store.revoke(entitlement_id)
            raise PremiumError("This purchase was removed or refunded.") from exc
        if not self.valid(ent, client.application_id, user_id):
            if ent.deleted or (ent.ends_at and ent.ends_at.timestamp() <= time.time()):
                self.store.revoke(entitlement_id)
            raise PremiumError("This purchase is not available to this account.")
        with self.store.connect() as conn:
            if conn.execute("SELECT 1 FROM character_revocations WHERE entitlement_id=?", (str(entitlement_id),)).fetchone():
                raise PremiumError("This purchase was removed or refunded.")
        return ent

    async def render(self, client, entitlement_id, user_id):
        await self.product(client)
        ent = await self.checked(client, entitlement_id, user_id)
        self.store.reserve(ent)
        lease = self.store.claim(ent.id, time.time())
        if lease is None:
            return self.store.row(ent.id)
        previous_context = usage_costs.get_request_context()
        usage_costs.set_request_context(user_id=str(user_id), intent="character_render")
        try:
            row = self.store.row(ent.id)
            if row["character_json"]:
                character = json.loads(row["character_json"])
            else:
                character = await self.catalog.choose(row["requested_name"])
                self.store.update(ent.id, lease, character_json=json.dumps(character))
            references = []
            if character.get("image"):
                fetched = await fetch_url_bytes_async(character["image"], timeout=15, max_bytes=2_000_000, allowed_content_types=("image/",))
                # Keep reference input cheap and remove incidental metadata.
                with Image.open(io.BytesIO(fetched.body)) as picture:
                    picture.thumbnail((384, 384))
                    buffer = io.BytesIO()
                    picture.convert("RGB").save(buffer, format="PNG")
                    references.append(buffer.getvalue())
            await self.checked(client, ent.id, user_id)
            from providers.stability_generation import generate_gpt_image
            from services.security_limits import check_rate_limit, provider_capacity
            # This purchased product fixes the provider/model rather than accepting
            # behavior overrides. Claim a paid attempt only after capacity opens.
            async with provider_capacity(user_id):
                await self.checked(client, ent.id, user_id)
                decision = check_rate_limit("image", user_id=user_id, guild_id="DM")
                if not decision.allowed:
                    raise PremiumError(f"Image limits are busy. Retry /character in {decision.retry_after}s; your purchase is retained.")
                self.store.start_render(ent.id, lease)
                image = await asyncio.wait_for(generate_gpt_image(
                    character_prompt(character), references=references,
                    model=CHARACTER_MODEL, quality="low", size="1024x1024",
                ), timeout=600)
            if not image:
                raise PremiumError("The renderer is temporarily unavailable. Retry /character; your selection and purchase are retained.")
            raw = image.getvalue()
            if len(raw) > 8_000_000:
                raise PremiumError("The image exceeded the delivery limit. Contact bot@sardistic.com.")
            with Image.open(io.BytesIO(raw)) as picture:
                picture.verify()
            await self.checked(client, ent.id, user_id)
            self.store.update(ent.id, lease, image=raw, state="ready", lease_until=0)
            return self.store.row(ent.id)
        except BaseException:
            try:
                self.store.update(ent.id, lease, state="failed", lease_until=0)
            except PremiumError:
                pass
            raise
        finally:
            usage_costs.set_request_context(**previous_context)


_SERVICE = None


def get_character_service():
    global _SERVICE
    if _SERVICE is None:
        root = Path(os.getenv("MULTIVAC_STATE_DIR", str(Path(__file__).resolve().parent.parent)))
        sku_id = int(os.getenv("MULTIVAC_CHARACTER_SKU_ID", str(CHARACTER_SKU_ID)))
        _SERVICE = CharacterRenderService(CharacterStore(root / "character_renders.db"), sku_id=sku_id)
    return _SERVICE
