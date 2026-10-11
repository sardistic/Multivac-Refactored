"""Verified one-use purchases of a timed, server-scoped generation style."""

import asyncio
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import discord

from services.premium import PremiumError, entitlement_valid, _sku_env
from services.usage_costs import get_request_context

UNATTRACTIVE_SKU_ID = 1558762748324876308
EFFECT_SECONDS = 600
STYLE_MARKER = "[MultiVAC Unattractive mode]"
CHAT_STYLE = (
    STYLE_MARKER + " A paid temporary server-wide style is active. Answer the actual request "
    "in a deliberately unappealing, dour, dry, mildly cantankerous voice, with ugly or "
    "awkward creative choices when relevant. Keep useful answers, facts, citations, code, "
    "and image descriptions accurate. Do not invent flaws in real people, insult the "
    "requester, harass anyone, or degrade safety or operational instructions. This is a "
    "style modifier, not a replacement task. Do not change tools, permissions or models."
)
VISUAL_STYLE = (
    STYLE_MARKER + " Render the requested subject with an intentionally unattractive "
    "aesthetic: awkward composition, drab clashing colors, unflattering lighting, "
    "ungainly styling, and shabby details. Preserve the subject's identity, requested "
    "content, reference images, readable factual labels, dimensions and transparency. "
    "Do not add degrading stereotypes, injuries, sexual content, or an unrelated subject."
)


class ServerEffectStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS server_effect_purchases (
                    entitlement_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                    sku_id TEXT NOT NULL, guild_id TEXT NOT NULL, state TEXT NOT NULL,
                    created_at REAL NOT NULL, grant_start REAL, grant_end REAL,
                    revoked INTEGER NOT NULL DEFAULT 0, stopped INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS server_effect_guild
                    ON server_effect_purchases(guild_id,state,grant_end);
                CREATE TABLE IF NOT EXISTS server_effect_revocations (
                    entitlement_id TEXT PRIMARY KEY
                );
            """)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def pending(self, user_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM server_effect_purchases WHERE user_id=? AND state='pending' AND revoked=0 ORDER BY created_at LIMIT 1", (str(user_id),)).fetchone()
        return dict(row) if row else None

    def unavailable(self):
        with self.connect() as conn:
            return {int(r[0]) for r in conn.execute("SELECT entitlement_id FROM server_effect_purchases UNION SELECT entitlement_id FROM server_effect_revocations")}

    def reserve(self, entitlement, user_id, guild_id, now):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM server_effect_revocations WHERE entitlement_id=?", (str(entitlement.id),)).fetchone():
                raise PremiumError("This purchase was removed and cannot be activated.")
            row = conn.execute("SELECT * FROM server_effect_purchases WHERE entitlement_id=?", (str(entitlement.id),)).fetchone()
            if row:
                if row['user_id'] != str(user_id) or row['guild_id'] != str(guild_id) or row['state'] != 'pending' or row['revoked']:
                    raise PremiumError("This purchase has already been used or reserved for another server.")
                return
            conn.execute("INSERT INTO server_effect_purchases(entitlement_id,user_id,sku_id,guild_id,state,created_at) VALUES(?,?,?,?,'pending',?)", (str(entitlement.id),str(user_id),str(entitlement.sku_id),str(guild_id),now))

    def apply(self, entitlement_id, now):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM server_effect_purchases WHERE entitlement_id=?", (str(entitlement_id),)).fetchone()
            if not row or row['revoked'] or row['state'] != 'pending':
                raise PremiumError("This purchase was removed or already activated.")
            latest = conn.execute("SELECT MAX(grant_end) FROM server_effect_purchases WHERE guild_id=? AND state='applied' AND revoked=0 AND stopped=0", (row['guild_id'],)).fetchone()[0]
            start = max(now, latest or now)
            end = start + EFFECT_SECONDS
            conn.execute("UPDATE server_effect_purchases SET state='applied',grant_start=?,grant_end=? WHERE entitlement_id=?", (start,end,str(entitlement_id)))
        return {'guild_id':row['guild_id'], 'grant_start':start, 'grant_end':end}

    def active_until(self, guild_id, now):
        if guild_id in (None, 'DM'):
            return None
        with self.connect() as conn:
            rows = list(conn.execute("SELECT grant_start,grant_end FROM server_effect_purchases WHERE guild_id=? AND state='applied' AND revoked=0 AND stopped=0 AND grant_end>? ORDER BY grant_start", (str(guild_id),now)))
        end = None
        for row in rows:
            if row['grant_start'] <= (end if end is not None else now):
                end = row['grant_end']
            else:
                break
        return end

    def outstanding(self, now):
        with self.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM server_effect_purchases WHERE revoked=0 AND (state='pending' OR (state='applied' AND stopped=0 AND grant_end>?))", (now,))]

    def revoke(self, entitlement_id):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT OR IGNORE INTO server_effect_revocations VALUES(?)", (str(entitlement_id),))
            conn.execute("UPDATE server_effect_purchases SET revoked=1 WHERE entitlement_id=?", (str(entitlement_id),))

    def stop(self, guild_id):
        with self.connect() as conn:
            conn.execute("UPDATE server_effect_purchases SET stopped=1 WHERE guild_id=? AND state='applied'", (str(guild_id),))


class ServerEffectService:
    def __init__(self, store, *, clock=time.time, sku_id=None):
        self.store, self.clock = store, clock
        self.sku_id = _sku_env('MULTIVAC_UNATTRACTIVE_SKU_ID',UNATTRACTIVE_SKU_ID) if sku_id is None else sku_id
        self._network_lock = asyncio.Lock()

    async def product(self, client):
        if not self.sku_id:
            raise PremiumError("Unattractive is disabled by the operator.")
        sku = next((s for s in await client.fetch_skus() if s.id==self.sku_id and s.application_id==client.application_id), None)
        if sku is None or sku.type != discord.SKUType.consumable:
            raise PremiumError("Unattractive must be a consumable SKU belonging to MultiVAC.")
        return sku

    def valid(self, entitlement, application_id, user_id):
        return (entitlement_valid(entitlement,application_id,self.sku_id,self.clock())
                and str(entitlement.user_id)==str(user_id) and entitlement.guild_id is None)

    def active_until(self, guild_id):
        return self.store.active_until(guild_id,self.clock()) if self.sku_id else None

    async def activate(self, client, user_id, guild_id, *, before_consume):
        if guild_id in (None,'DM'):
            raise PremiumError("Activate Unattractive inside the server you want it to affect.")
        async with self._network_lock:
            await self.product(client)
            pending = self.store.pending(user_id)
            if pending and pending['guild_id'] != str(guild_id):
                raise PremiumError("A pending activation is reserved for another server. Recover it there first.")
            if pending:
                try:
                    entitlement = await client.fetch_entitlement(int(pending['entitlement_id']))
                except discord.NotFound as exc:
                    self.store.revoke(pending['entitlement_id'])
                    raise PremiumError("The pending purchase was removed. No other purchase was spent.") from exc
            else:
                entitlement = None
                unavailable = self.store.unavailable()
                async for candidate in client.entitlements(limit=None,user=discord.Object(int(user_id)),skus=[discord.Object(self.sku_id)],exclude_ended=True):
                    if self.valid(candidate,client.application_id,user_id) and not candidate.consumed and candidate.id not in unavailable:
                        entitlement = candidate
                        break
                if entitlement is None:
                    raise PremiumError("No unused Unattractive purchase was found. Buy it with /unattractive, then activate it in this server.")
            if not self.valid(entitlement,client.application_id,user_id):
                if pending:
                    self.store.revoke(entitlement.id)
                raise PremiumError("This purchase is not valid for your account.")
            await before_consume()
            self.store.reserve(entitlement,user_id,guild_id,self.clock())
            if not entitlement.consumed:
                try:
                    await entitlement.consume()
                except (discord.HTTPException,OSError,asyncio.TimeoutError) as exc:
                    raise PremiumError("Discord did not confirm activation. The timer has not started. Retry /unattractive activate in this server to recover the purchase.") from exc
            if not self.valid(entitlement,client.application_id,user_id):
                self.store.revoke(entitlement.id)
                raise PremiumError("This purchase expired or was removed before activation completed.")
            # Persist the grant exactly once, including consumed pending recovery.
            return self.store.apply(entitlement.id,self.clock())

    async def reconcile(self, client):
        for row in self.store.outstanding(self.clock()):
            try:
                entitlement = await client.fetch_entitlement(int(row['entitlement_id']))
            except discord.NotFound:
                self.store.revoke(row['entitlement_id'])
                continue
            if not self.valid(entitlement,client.application_id,row['user_id']):
                self.store.revoke(row['entitlement_id'])

    def observe(self, entitlement, application_id, *, deleted=False):
        if entitlement.application_id==application_id and entitlement.sku_id==self.sku_id:
            if deleted or not self.valid(entitlement,application_id,entitlement.user_id):
                self.store.revoke(entitlement.id)


_SERVICE = None
_SERVICE_LOCK = threading.Lock()


def get_server_effect_service():
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            root = Path(os.getenv('MULTIVAC_STATE_DIR',str(Path(__file__).resolve().parent.parent)))
            _SERVICE = ServerEffectService(ServerEffectStore(root/'premium.db'))
    return _SERVICE


def effect_active():
    guild_id = get_request_context().get('guild_id')
    return bool(guild_id not in (None,'DM') and get_server_effect_service().active_until(guild_id))


def visual_prompt(prompt):
    if STYLE_MARKER in prompt or not effect_active():
        return prompt
    return VISUAL_STYLE + "\n\nRequested scene:\n" + prompt


def _style_messages(messages):
    if any(m.get('role')=='system' and STYLE_MARKER in str(m.get('content','')) for m in messages):
        return messages
    return [*messages, {'role':'system','content':CHAT_STYLE}]


def decorate_provider_call(name, args, kwargs):
    """Only public response providers receive style; utility/model calls do not."""
    if name not in {'chat.openai','chat.openai_plain','chat.claude','chat.gemini','vision.openai','vision.gemini'} or not effect_active():
        return args,kwargs
    kwargs = dict(kwargs)
    if name.endswith('gemini'):
        kwargs['context'] = _style_messages(list(kwargs.get('context') or []))
    elif args and isinstance(args[0],list):
        args = (_style_messages(args[0]),*args[1:])
    elif isinstance(kwargs.get('messages'),list):
        kwargs['messages'] = _style_messages(kwargs['messages'])
    return args,kwargs
