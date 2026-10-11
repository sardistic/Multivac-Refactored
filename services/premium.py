"""Discord-verified subscriptions and durable, one-use premium purchases."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path

import discord


PASS_SECONDS = 30 * 86400
VERIFICATION_SECONDS = 300
SKU_FIELDS = ("guild", "user", "reset", "unthrottled", "donate", "donate_more")


class PremiumError(RuntimeError):
    """Safe, user-facing premium failure."""


def _positive_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _sku_env(name: str, default: int = 0) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return 0


@dataclass(frozen=True)
class SKUConfig:
    guild: int = 0
    user: int = 0
    reset: int = 0
    unthrottled: int = 0
    donate: int = 0
    donate_more: int = 0

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(getattr(self, key) for key in SKU_FIELDS if getattr(self, key))


@dataclass(frozen=True)
class PremiumBenefits:
    user_multiplier: int = 1
    guild_multiplier: int = 1
    user_pass_until: float | None = None
    guild_active: bool = False
    reset_generation: int = 0
    bypass_global_throttle: bool = False
    subscription_until: float | None = None


def entitlement_valid(entitlement, application_id: int, sku_id: int, now: float) -> bool:
    if not sku_id or entitlement.application_id != application_id or entitlement.sku_id != sku_id:
        return False
    if entitlement.deleted:
        return False
    if entitlement.starts_at is not None and entitlement.starts_at.timestamp() > now:
        return False
    return entitlement.ends_at is None or entitlement.ends_at.timestamp() > now


def validate_catalog(config: SKUConfig, catalog, application_id: int) -> dict:
    if len(config.ids) != len(set(config.ids)):
        raise PremiumError("Each premium product must have a different SKU ID.")
    known = {sku.id: sku for sku in catalog if sku.application_id == application_id}
    for kind in SKU_FIELDS:
        sku_id = getattr(config, kind)
        if not sku_id:
            continue
        sku = known.get(sku_id)
        if sku is None:
            raise PremiumError(f"The {kind} SKU does not belong to this Discord application.")
        if kind == "guild":
            if sku.type != discord.SKUType.subscription or not sku.flags.guild_subscription:
                raise PremiumError("The guild SKU must be a guild subscription, not a subscription group.")
        elif kind == "unthrottled":
            if sku.type != discord.SKUType.subscription or not sku.flags.user_subscription:
                raise PremiumError("fUnThrottled must be a user subscription, not a subscription group.")
        elif sku.type != discord.SKUType.consumable:
            raise PremiumError(f"The {kind} SKU must be a consumable one-time purchase.")
    return known


class PremiumStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS premium_settings (
                    id INTEGER PRIMARY KEY CHECK (id=1), guild_sku TEXT NOT NULL,
                    user_sku TEXT NOT NULL, reset_sku TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS premium_redemptions (
                    entitlement_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                    sku_id TEXT NOT NULL, kind TEXT NOT NULL,
                    state TEXT NOT NULL, created_at REAL NOT NULL,
                    grant_start REAL, grant_end REAL, revoked INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS premium_redemptions_user
                    ON premium_redemptions(user_id, kind, state);
                CREATE TABLE IF NOT EXISTS premium_reset_epochs (
                    user_id TEXT PRIMARY KEY, generation INTEGER NOT NULL
                );
            """)
            # Preserve configuration and grants from the earlier local schema.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(premium_settings)")}
            for key in ("unthrottled", "donate", "donate_more"):
                if f"{key}_sku" not in columns:
                    conn.execute(f"ALTER TABLE premium_settings ADD COLUMN {key}_sku TEXT NOT NULL DEFAULT '0'")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def config(self) -> SKUConfig:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM premium_settings WHERE id=1").fetchone()
        if row:
            return SKUConfig(*(int(row[f"{key}_sku"]) for key in SKU_FIELDS))
        return SKUConfig(
            _sku_env("MULTIVAC_GUILD_SUBSCRIPTION_SKU_ID"),
            _sku_env("MULTIVAC_USER_PASS_SKU_ID"),
            _sku_env("MULTIVAC_THROTTLE_RESET_SKU_ID"),
            _sku_env("MULTIVAC_UNTHROTTLED_SKU_ID", 1558736446351872112),
            _sku_env("MULTIVAC_DONATE_SKU_ID", 1558738728720007219),
            _sku_env("MULTIVAC_DONATE_MORE_SKU_ID", 1558739372851859466),
        )

    def save_config(self, config: SKUConfig) -> None:
        previous = self.config()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if config != previous and conn.execute("SELECT 1 FROM premium_redemptions LIMIT 1").fetchone():
                raise PremiumError("SKU mappings cannot be changed after consumable redemption. Contact the operator for a migration.")
            conn.execute(
                "INSERT OR REPLACE INTO premium_settings(id,guild_sku,user_sku,reset_sku,unthrottled_sku,donate_sku,donate_more_sku) VALUES(1,?,?,?,?,?,?)",
                tuple(str(getattr(config, key)) for key in SKU_FIELDS),
            )

    def pending(self, user_id: str, kind: str):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM premium_redemptions WHERE user_id=? AND kind=? AND state='pending' AND revoked=0 ORDER BY created_at LIMIT 1",
                (user_id, kind),
            ).fetchone()
        return dict(row) if row else None

    def unavailable(self, user_id: str, kind: str) -> set[int]:
        with self.connect() as conn:
            return {int(row[0]) for row in conn.execute(
                "SELECT entitlement_id FROM premium_redemptions WHERE user_id=? AND kind=? AND (state='applied' OR revoked=1)",
                (user_id, kind),
            )}

    def reserve(self, entitlement, user_id: str, kind: str, now: float) -> None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM premium_redemptions WHERE entitlement_id=?", (str(entitlement.id),)
            ).fetchone()
            if existing:
                if existing["user_id"] != user_id or existing["kind"] != kind or existing["revoked"]:
                    raise PremiumError("This purchase cannot be redeemed by this account.")
                if existing["state"] != "pending":
                    raise PremiumError("This purchase has already been redeemed.")
                return
            conn.execute(
                "INSERT INTO premium_redemptions(entitlement_id,user_id,sku_id,kind,state,created_at) VALUES(?,?,?,?,'pending',?)",
                (str(entitlement.id), user_id, str(entitlement.sku_id), kind, now),
            )

    def apply(self, entitlement_id: int, now: float) -> dict:
        """One transaction both grants the benefit and marks the purchase applied."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM premium_redemptions WHERE entitlement_id=?", (str(entitlement_id),)
            ).fetchone()
            if row is None or row["revoked"]:
                raise PremiumError("This purchase is no longer available to redeem.")
            if row["state"] == "applied":
                raise PremiumError("This purchase has already been redeemed.")
            grant_start = grant_end = None
            if row["kind"] == "user":
                latest = conn.execute(
                    "SELECT MAX(grant_end) FROM premium_redemptions WHERE user_id=? AND kind='user' AND state='applied' AND revoked=0",
                    (row["user_id"],),
                ).fetchone()[0]
                grant_start = max(now, latest or now)
                grant_end = grant_start + PASS_SECONDS
            elif row["kind"] == "reset":
                conn.execute(
                    "INSERT INTO premium_reset_epochs VALUES(?,1) ON CONFLICT(user_id) DO UPDATE SET generation=generation+1",
                    (row["user_id"],),
                )
            conn.execute(
                "UPDATE premium_redemptions SET state='applied',grant_start=?,grant_end=? WHERE entitlement_id=?",
                (grant_start, grant_end, str(entitlement_id)),
            )
        return {"kind": row["kind"], "user_id": row["user_id"],
                "grant_start": grant_start, "grant_end": grant_end}

    def grants(self, now: float) -> list[dict]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM premium_redemptions WHERE kind='user' AND state='applied' AND revoked=0 AND grant_end>?",
                (now,),
            )]

    def reset_generations(self) -> dict[str, int]:
        with self.connect() as conn:
            return {row["user_id"]: row["generation"] for row in conn.execute("SELECT * FROM premium_reset_epochs")}

    def revoke(self, entitlement_id: int) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE premium_redemptions SET revoked=1 WHERE entitlement_id=?", (str(entitlement_id),))


class PremiumService:
    def __init__(self, store: PremiumStore, *, clock=time.time):
        self.store = store
        self.clock = clock
        self.config = store.config()
        self.user_boost = min(10, _positive_env("MULTIVAC_USER_BONUS_MULTIPLIER", 2))
        self.guild_boost = min(10, _positive_env("MULTIVAC_GUILD_BONUS_MULTIPLIER", 2))
        self.catalog: dict = {}
        self._guilds: dict[int, tuple[object, float]] = {}
        self._users: dict[int, tuple[object, float]] = {}
        self._grants: dict[int, tuple[dict, float]] = {}
        self._generations = store.reset_generations()
        self._lock = threading.RLock()
        self._network_lock = asyncio.Lock()
        self._deleted: set[int] = set()
        self._revision = 0
        self._subscription_revisions: dict[int, int] = {}
        self._last_refresh = float("-inf")

    async def configure(self, client, config: SKUConfig) -> None:
        if not config.ids:
            raise PremiumError("Configure at least one product SKU ID.")
        async with self._network_lock:
            catalog = validate_catalog(config, await client.fetch_skus(), client.application_id)
            self.store.save_config(config)
            with self._lock:
                self.config = config
                self.catalog = catalog
                self._guilds.clear()
                self._users.clear()
                self._grants.clear()
                self._last_refresh = float("-inf")

    def observe(self, entitlement, application_id: int, *, deleted: bool = False) -> None:
        if entitlement.application_id != application_id:
            return
        with self._lock:
            self._revision += 1
            if deleted or entitlement.deleted:
                self._deleted.add(entitlement.id)
                self._guilds.pop(entitlement.id, None)
                self._users.pop(entitlement.id, None)
                self._grants.pop(entitlement.id, None)
                self.store.revoke(entitlement.id)
            elif (entitlement.sku_id == self.config.guild and entitlement.guild_id is not None
                  and entitlement.sku_id in self.catalog):
                if entitlement.id not in self._deleted:
                    self._guilds[entitlement.id] = (entitlement, self.clock())
                    self._subscription_revisions[entitlement.id] = self._revision
            elif (entitlement.sku_id == self.config.unthrottled and entitlement.user_id is not None
                  and entitlement.guild_id is None and entitlement.sku_id in self.catalog):
                if entitlement.id not in self._deleted:
                    self._users[entitlement.id] = (entitlement, self.clock())
                    self._subscription_revisions[entitlement.id] = self._revision

    async def refresh(self, client, *, force: bool = False) -> None:
        if not self.config.ids:
            return
        async with self._network_lock:
            if not force and self.clock() - self._last_refresh < 30:
                return
            with self._lock:
                started_revision = self._revision
            catalog = validate_catalog(self.config, await client.fetch_skus(), client.application_id)
            guilds = {}
            users = {}
            subscriptions = [value for value in (self.config.guild, self.config.unthrottled) if value]
            if subscriptions:
                async for entitlement in client.entitlements(
                    limit=None, skus=[discord.Object(value) for value in subscriptions], exclude_ended=True
                ):
                    if (entitlement.guild_id is not None
                            and entitlement_valid(entitlement, client.application_id, self.config.guild, self.clock())):
                        guilds[entitlement.id] = (entitlement, self.clock())
                    elif (entitlement.guild_id is None and entitlement.user_id is not None
                            and entitlement_valid(entitlement, client.application_id, self.config.unthrottled, self.clock())):
                        users[entitlement.id] = (entitlement, self.clock())
            grants = {}
            for grant in self.store.grants(self.clock()):
                try:
                    entitlement = await client.fetch_entitlement(int(grant["entitlement_id"]))
                except discord.NotFound:
                    self._revoke(int(grant["entitlement_id"]))
                    continue
                if (entitlement_valid(entitlement, client.application_id, self.config.user, self.clock())
                        and entitlement.user_id is not None
                        and str(entitlement.user_id) == grant["user_id"] and entitlement.consumed):
                    grants[entitlement.id] = (grant, self.clock())
                else:
                    self._revoke(entitlement.id)
            with self._lock:
                # Gateway changes delivered during REST fetches remain authoritative.
                guilds.update({key: value for key, value in self._guilds.items()
                               if self._subscription_revisions.get(key, 0) > started_revision})
                users.update({key: value for key, value in self._users.items()
                              if self._subscription_revisions.get(key, 0) > started_revision})
                self.catalog = catalog
                self._guilds = {key: value for key, value in guilds.items() if key not in self._deleted}
                self._users = {key: value for key, value in users.items() if key not in self._deleted}
                self._grants = {key: value for key, value in grants.items() if key not in self._deleted}
                self._generations = self.store.reset_generations()
                self._last_refresh = self.clock()

    def _revoke(self, entitlement_id: int) -> None:
        with self._lock:
            self.store.revoke(entitlement_id)
            self._grants.pop(entitlement_id, None)

    def benefits(self, user_id: str | int, guild_id: str | int) -> PremiumBenefits:
        now = self.clock()
        user_key, guild_key = str(user_id), str(guild_id)
        with self._lock:
            subscriptions = [ent for ent, verified in self._users.values()
                             if str(ent.user_id) == user_key and now - verified < VERIFICATION_SECONDS
                             and entitlement_valid(ent, ent.application_id, self.config.unthrottled, now)]
            subscription_end = (
                max(ent.ends_at.timestamp() for ent in subscriptions)
                if subscriptions and all(ent.ends_at for ent in subscriptions) else None
            )
            guild_active = any(
                str(ent.guild_id) == guild_key and now - verified < VERIFICATION_SECONDS
                and entitlement_valid(ent, ent.application_id, self.config.guild, now)
                for ent, verified in self._guilds.values()
            )
            active = [grant for grant, verified in self._grants.values()
                      if grant["user_id"] == user_key and grant["grant_start"] <= now < grant["grant_end"]
                      and now - verified < VERIFICATION_SECONDS]
            # Include verified contiguous future grants when displaying the pass end.
            pass_end = max((grant["grant_end"] for grant in active), default=None)
            if pass_end is not None:
                for grant, verified in sorted(self._grants.values(), key=lambda item: item[0]["grant_start"]):
                    if (grant["user_id"] == user_key and grant["grant_start"] <= pass_end
                            and now - verified < VERIFICATION_SECONDS):
                        pass_end = max(pass_end, grant["grant_end"])
            return PremiumBenefits(
                user_multiplier=(self.user_boost if active else 1) * (self.guild_boost if guild_active else 1),
                guild_multiplier=self.guild_boost if guild_active else 1,
                user_pass_until=pass_end,
                guild_active=guild_active,
                reset_generation=self._generations.get(user_key, 0),
                bypass_global_throttle=bool(subscriptions),
                subscription_until=subscription_end,
            )

    async def redeem(self, client, user_id: str | int, kind: str) -> dict:
        if kind not in {"user", "reset", "donate", "donate_more"}:
            raise PremiumError("Choose a configured consumable product. Subscriptions apply automatically.")
        user_key = str(user_id)
        async with self._network_lock:
            sku_id = getattr(self.config, kind)
            if not sku_id:
                raise PremiumError("This product has not been configured yet.")
            validate_catalog(self.config, await client.fetch_skus(), client.application_id)
            pending = self.store.pending(user_key, kind)
            if pending:
                try:
                    entitlement = await client.fetch_entitlement(int(pending["entitlement_id"]))
                except discord.NotFound as exc:
                    self._revoke(int(pending["entitlement_id"]))
                    raise PremiumError("The pending purchase was removed. No replacement was spent; try /premium_redeem again for another unused purchase.") from exc
            else:
                entitlement = None
                unavailable = self.store.unavailable(user_key, kind)
                async for candidate in client.entitlements(
                    limit=None, user=discord.Object(int(user_id)), skus=[discord.Object(sku_id)], exclude_ended=True
                ):
                    if (entitlement_valid(candidate, client.application_id, sku_id, self.clock())
                            and str(candidate.user_id) == user_key and candidate.guild_id is None
                            and not candidate.consumed and candidate.id not in unavailable):
                        entitlement = candidate
                        break
                if entitlement is None:
                    raise PremiumError("No unused purchase was found for this product. Purchase it with /premium, then redeem it here.")
            if (not entitlement_valid(entitlement, client.application_id, sku_id, self.clock())
                    or str(entitlement.user_id) != user_key or entitlement.guild_id is not None):
                if pending:
                    self._revoke(int(pending["entitlement_id"]))
                raise PremiumError("This purchase is not valid for your account.")
            if entitlement.consumed and pending is None:
                raise PremiumError("This purchase has already been consumed.")
            self.store.reserve(entitlement, user_key, kind, self.clock())
            if not entitlement.consumed:
                try:
                    await entitlement.consume()
                except (discord.HTTPException, OSError, asyncio.TimeoutError) as exc:
                    raise PremiumError("Discord did not confirm redemption. No local bonus was granted. Try /premium_redeem again to recover this purchase.") from exc
            # No await between the durable grant and publishing its local effect.
            with self._lock:
                if entitlement.id in self._deleted:
                    raise PremiumError("This purchase was removed before redemption completed.")
                result = self.store.apply(entitlement.id, self.clock())
                if kind == "user":
                    for grant in self.store.grants(self.clock()):
                        if int(grant["entitlement_id"]) == entitlement.id:
                            self._grants[entitlement.id] = (grant, self.clock())
                elif kind == "reset":
                    self._generations = self.store.reset_generations()
                    from services.security_limits import RATE_LIMITER
                    RATE_LIMITER.reset_user(user_key, generation=self._generations[user_key])
            return result


_SERVICE: PremiumService | None = None
_SERVICE_LOCK = threading.Lock()


def get_premium_service() -> PremiumService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            root = Path(os.getenv("MULTIVAC_STATE_DIR", str(Path(__file__).resolve().parent.parent)))
            _SERVICE = PremiumService(PremiumStore(root / "premium.db"))
        return _SERVICE
