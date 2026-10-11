import asyncio
import sqlite3
from contextlib import closing
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import discord

from bot.premium import PremiumCog, RedemptionView
from services import security_limits
from services.premium import (
    PASS_SECONDS, VERIFICATION_SECONDS, PremiumBenefits, PremiumError,
    PremiumService, PremiumStore, SKUConfig, validate_catalog,
)
from services.security_limits import RateLimitRule, SlidingWindowLimiter


def sku(identifier, kind, flags):
    return SimpleNamespace(id=identifier, application_id=42, type=kind,
                           flags=discord.SKUFlags._from_value(flags), name=f"Product {identifier}")


def entitlement(identifier, sku_id, user_id=7, guild_id=None, **overrides):
    value = SimpleNamespace(id=identifier, sku_id=sku_id, application_id=42,
                            user_id=user_id, guild_id=guild_id, starts_at=None,
                            ends_at=None, deleted=False, consumed=False)
    for name, item in overrides.items():
        setattr(value, name, item)

    async def consume():
        value.consumed = True

    value.consume = AsyncMock(side_effect=consume)
    return value


class FakeClient:
    application_id = 42

    def __init__(self):
        self.catalog = [sku(100, discord.SKUType.subscription, 128 | 4),
                        sku(200, discord.SKUType.consumable, 4),
                        sku(300, discord.SKUType.consumable, 4)]
        self.items = {}
        self.fetch_skus = AsyncMock(side_effect=lambda: self.catalog)
        self.fetch_entitlement = AsyncMock(side_effect=lambda identifier: self.items[identifier])
        self.queries = []

    async def entitlements(self, **kwargs):
        self.queries.append(kwargs)
        for item in sorted(list(self.items.values()), key=lambda row: row.id):
            if kwargs.get("skus") and item.sku_id not in [entry.id for entry in kwargs["skus"]]:
                continue
            if kwargs.get("user") and item.user_id != kwargs["user"].id:
                continue
            yield item


class PremiumServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = PremiumStore(Path(self.directory.name) / "premium.db")
        self.now = 2_000_000.0
        self.service = PremiumService(self.store, clock=lambda: self.now)
        limiter_patch = patch.object(security_limits, "RATE_LIMITER", SlidingWindowLimiter())
        limiter_patch.start()
        self.addCleanup(limiter_patch.stop)
        self.client = FakeClient()
        await self.service.configure(self.client, SKUConfig(100, 200, 300))

    def add(self, item):
        self.client.items[item.id] = item
        return item

    async def test_guild_subscription_is_scoped_and_stacks_with_personal_pass(self):
        self.add(entitlement(1, 100, guild_id=9))
        personal = self.add(entitlement(2, 200))
        await self.service.refresh(self.client)
        self.assertEqual(self.service.benefits(7, 9).user_multiplier, 2)
        self.assertEqual(self.service.benefits(8, 9).guild_multiplier, 2)
        self.assertEqual(self.service.benefits(7, 10).user_multiplier, 1)
        await self.service.redeem(self.client, 7, "user")
        self.assertEqual(self.service.benefits(7, 9).user_multiplier, 4)
        self.assertEqual(self.service.benefits(7, 9).guild_multiplier, 2)
        self.assertEqual(self.service.benefits(7, "DM").user_multiplier, 2)
        self.assertEqual(self.service.benefits(8, 10).user_multiplier, 1)
        personal.consume.assert_awaited_once()

    async def test_ended_future_deleted_foreign_and_unknown_entitlements_grant_nothing(self):
        timestamp = lambda seconds: datetime.fromtimestamp(seconds, timezone.utc)
        for item in (
            entitlement(1, 100, guild_id=9, ends_at=timestamp(self.now)),
            entitlement(2, 100, guild_id=9, starts_at=timestamp(self.now + 60)),
            entitlement(3, 100, guild_id=9, deleted=True),
            entitlement(4, 100, guild_id=9, application_id=99),
            entitlement(5, 999, guild_id=9),
        ):
            self.add(item)
        await self.service.refresh(self.client)
        self.assertEqual(self.service.benefits(7, 9).user_multiplier, 1)

    async def test_expiry_and_stale_verification_remove_paid_boost(self):
        item = self.add(entitlement(1, 100, guild_id=9))
        await self.service.refresh(self.client)
        self.now += VERIFICATION_SECONDS
        self.assertFalse(self.service.benefits(7, 9).guild_active)
        await self.service.refresh(self.client, force=True)
        self.assertTrue(self.service.benefits(7, 9).guild_active)
        item.ends_at = datetime.fromtimestamp(self.now + 5, timezone.utc)
        self.service.observe(item, 42)
        self.now += 5
        self.assertFalse(self.service.benefits(7, 9).guild_active)

    async def test_user_passes_extend_thirty_days_and_restore_after_restart(self):
        self.add(entitlement(1, 200))
        self.add(entitlement(2, 200))
        first = await self.service.redeem(self.client, 7, "user")
        second = await self.service.redeem(self.client, 7, "user")
        self.assertEqual(first["grant_end"], self.now + PASS_SECONDS)
        self.assertEqual(second["grant_end"], self.now + 2 * PASS_SECONDS)
        restarted = PremiumService(self.store, clock=lambda: self.now)
        self.assertEqual(restarted.benefits(7, 9).user_multiplier, 1)
        await restarted.refresh(self.client)
        self.assertEqual(restarted.benefits(7, 9).user_pass_until, self.now + 2 * PASS_SECONDS)
        self.now += 2 * PASS_SECONDS
        self.assertEqual(restarted.benefits(7, 9).user_multiplier, 1)

    async def test_owned_purchase_cannot_be_redeemed_by_another_user(self):
        item = self.add(entitlement(1, 200, user_id=8))
        with self.assertRaises(PremiumError):
            await self.service.redeem(self.client, 7, "user")
        item.consume.assert_not_awaited()
        self.assertEqual(self.store.grants(self.now), [])

    async def test_guild_owned_or_foreign_consumable_is_rejected(self):
        self.add(entitlement(1, 200, guild_id=9))
        self.add(entitlement(2, 200, application_id=99))
        with self.assertRaises(PremiumError):
            await self.service.redeem(self.client, 7, "user")

    async def test_consumption_failure_does_not_grant_and_can_be_retried(self):
        item = self.add(entitlement(1, 200))
        item.consume.side_effect = asyncio.TimeoutError()
        with self.assertRaises(PremiumError):
            await self.service.redeem(self.client, 7, "user")
        self.assertEqual(self.store.grants(self.now), [])
        self.assertIsNotNone(self.store.pending("7", "user"))
        item.consume.side_effect = None
        await self.service.redeem(self.client, 7, "user")
        self.assertEqual(len(self.store.grants(self.now)), 1)

    async def test_invalid_pending_purchase_does_not_block_later_unused_purchases(self):
        old = self.add(entitlement(1, 200))
        self.store.reserve(old, "7", "user", self.now)
        old.deleted = True
        replacement = self.add(entitlement(2, 200))
        with self.assertRaises(PremiumError):
            await self.service.redeem(self.client, 7, "user")
        replacement.consume.assert_not_awaited()
        await self.service.redeem(self.client, 7, "user")
        replacement.consume.assert_awaited_once()

    async def test_missing_pending_purchase_can_be_cleared_without_spending_replacement(self):
        old = self.add(entitlement(1, 200))
        self.store.reserve(old, "7", "user", self.now)
        replacement = self.add(entitlement(2, 200))
        old.deleted = True

        async def fetch(identifier):
            if identifier == 1:
                raise discord.NotFound(SimpleNamespace(status=404, reason="Not Found"), "not found")
            return self.client.items[identifier]

        self.client.fetch_entitlement.side_effect = fetch
        with self.assertRaises(PremiumError):
            await self.service.redeem(self.client, 7, "user")
        replacement.consume.assert_not_awaited()
        self.assertIsNone(self.store.pending("7", "user"))
        await self.service.redeem(self.client, 7, "user")
        replacement.consume.assert_awaited_once()

    async def test_remote_consumption_with_lost_response_is_recovered_without_second_consume(self):
        item = self.add(entitlement(1, 200))

        async def lost_response():
            item.consumed = True
            raise asyncio.TimeoutError()

        item.consume.side_effect = lost_response
        with self.assertRaises(PremiumError):
            await self.service.redeem(self.client, 7, "user")
        restarted = PremiumService(self.store, clock=lambda: self.now)
        await restarted.redeem(self.client, 7, "user")
        item.consume.assert_awaited_once()
        self.assertEqual(len(self.store.grants(self.now)), 1)

    async def test_concurrent_duplicate_redemptions_grant_once(self):
        item = self.add(entitlement(1, 200))
        results = await asyncio.gather(
            self.service.redeem(self.client, 7, "user"),
            self.service.redeem(self.client, 7, "user"), return_exceptions=True,
        )
        self.assertEqual(sum(isinstance(result, dict) for result in results), 1)
        item.consume.assert_awaited_once()
        self.assertEqual(len(self.store.grants(self.now)), 1)

    async def test_applied_purchase_is_skipped_if_discord_listing_is_stale(self):
        first = self.add(entitlement(1, 200))
        await self.service.redeem(self.client, 7, "user")
        first.consumed = False
        second = self.add(entitlement(2, 200))
        result = await self.service.redeem(self.client, 7, "user")
        first.consume.assert_awaited_once()
        second.consume.assert_awaited_once()
        self.assertEqual(result["grant_end"], self.now + 2 * PASS_SECONDS)

    async def test_local_commit_failure_after_remote_consumption_is_recoverable(self):
        item = self.add(entitlement(1, 200))
        with patch.object(self.store, "apply", side_effect=sqlite3.OperationalError("temporary failure")):
            with self.assertRaises(sqlite3.OperationalError):
                await self.service.redeem(self.client, 7, "user")
        self.assertTrue(item.consumed)
        self.assertEqual(self.store.grants(self.now), [])
        restarted = PremiumService(self.store, clock=lambda: self.now)
        await restarted.redeem(self.client, 7, "user")
        item.consume.assert_awaited_once()
        self.assertEqual(len(self.store.grants(self.now)), 1)

    async def test_partial_refresh_failure_does_not_restore_a_revoked_pass(self):
        self.add(entitlement(1, 200))
        await self.service.redeem(self.client, 7, "user")
        self.add(entitlement(2, 200, user_id=8))
        await self.service.redeem(self.client, 8, "user")
        self.client.items[1].deleted = True

        async def fetch(identifier):
            if identifier == 2:
                raise asyncio.TimeoutError()
            return self.client.items[identifier]

        self.client.fetch_entitlement.side_effect = fetch
        with self.assertRaises(asyncio.TimeoutError):
            await self.service.refresh(self.client, force=True)
        self.assertEqual(self.service.benefits(7, 9).user_multiplier, 1)

    async def test_gateway_event_cannot_bypass_invalid_catalog_configuration(self):
        self.service.catalog.clear()
        self.service.observe(entitlement(1, 100, guild_id=9), 42)
        self.assertFalse(self.service.benefits(7, 9).guild_active)

    async def test_refunded_pass_is_revoked_and_cannot_be_replayed(self):
        item = self.add(entitlement(1, 200))
        await self.service.redeem(self.client, 7, "user")
        self.service.observe(item, 42, deleted=True)
        self.assertEqual(self.service.benefits(7, 9).user_multiplier, 1)
        await self.service.refresh(self.client, force=True)
        self.assertEqual(self.service.benefits(7, 9).user_multiplier, 1)

    async def test_refund_detected_via_rest_after_missed_event(self):
        item = self.add(entitlement(1, 200))
        await self.service.redeem(self.client, 7, "user")
        item.deleted = True
        await self.service.refresh(self.client, force=True)
        self.assertEqual(self.service.benefits(7, 9).user_multiplier, 1)
        self.assertEqual(self.store.grants(self.now), [])

    async def test_gateway_delete_during_refresh_cannot_be_undone_by_snapshot(self):
        item = self.add(entitlement(1, 100, guild_id=9))

        async def racing_entitlements(**kwargs):
            yield item
            self.service.observe(item, 42, deleted=True)

        self.client.entitlements = racing_entitlements
        await self.service.refresh(self.client)
        self.assertFalse(self.service.benefits(7, 9).guild_active)

    async def test_gateway_update_during_refresh_wins_over_stale_snapshot(self):
        stale = self.add(entitlement(1, 100, guild_id=9))
        updated = entitlement(1, 100, guild_id=9, ends_at=datetime.fromtimestamp(self.now, timezone.utc))

        async def racing_entitlements(**kwargs):
            yield stale
            self.service.observe(updated, 42)

        self.client.entitlements = racing_entitlements
        await self.service.refresh(self.client)
        self.assertFalse(self.service.benefits(7, 9).guild_active)

    async def test_reset_is_durable_and_does_not_reset_again_after_restart(self):
        item = self.add(entitlement(1, 300))
        await self.service.redeem(self.client, 7, "reset")
        self.assertEqual(self.service.benefits(7, 9).reset_generation, 1)
        restarted = PremiumService(self.store, clock=lambda: self.now)
        self.assertEqual(restarted.benefits(7, 9).reset_generation, 1)
        with self.assertRaises(PremiumError):
            await restarted.redeem(self.client, 7, "reset")
        item.consume.assert_awaited_once()
        self.assertEqual(restarted.benefits(7, 9).reset_generation, 1)

    async def test_applied_purchase_cannot_be_reapplied_at_store_boundary(self):
        item = self.add(entitlement(1, 300))
        await self.service.redeem(self.client, 7, "reset")
        with self.assertRaises(PremiumError):
            self.store.apply(item.id, self.now)
        self.assertEqual(self.store.reset_generations()["7"], 1)

    async def test_configuration_is_saved_and_requires_correct_sku_types(self):
        self.assertEqual(PremiumStore(self.store.path).config(), SKUConfig(100, 200, 300))
        bad = [sku(100, discord.SKUType.subscription_group, 128), *self.client.catalog[1:]]
        with self.assertRaises(PremiumError):
            validate_catalog(self.service.config, bad, 42)
        with self.assertRaises(PremiumError):
            validate_catalog(SKUConfig(100, 200, 200), self.client.catalog, 42)
        bad = [sku(100, discord.SKUType.subscription, 256), *self.client.catalog[1:]]
        with self.assertRaises(PremiumError):
            validate_catalog(self.service.config, bad, 42)

    async def test_owner_cannot_reassign_skus_after_consumable_redemption(self):
        self.add(entitlement(1, 200))
        await self.service.redeem(self.client, 7, "user")
        with self.assertRaises(PremiumError):
            self.store.save_config(SKUConfig(100, 300, 200))
        self.assertEqual(self.store.config(), SKUConfig(100, 200, 300))


class PremiumLimiterTests(unittest.TestCase):
    def test_reset_preserves_other_users_guild_and_global_events(self):
        limiter = SlidingWindowLimiter(clock=lambda: 100)
        rule = RateLimitRule(user=1, guild=3, global_=4, window_seconds=60)
        self.assertTrue(limiter.check("request", user_id=7, guild_id=9, rule=rule).allowed)
        self.assertFalse(limiter.check("request", user_id=7, guild_id=9, rule=rule).allowed)
        self.assertTrue(limiter.check("request", user_id=8, guild_id=9, rule=rule).allowed)
        limiter.reset_user(7, generation=1)
        self.assertFalse(limiter.check("request", user_id=8, guild_id=9, rule=rule).allowed)
        self.assertTrue(limiter.check("request", user_id=7, guild_id=9, rule=rule, user_generation=1).allowed)
        limiter.reset_user(7, generation=1)
        self.assertFalse(limiter.check("request", user_id=7, guild_id=9, rule=rule, user_generation=1).allowed)
        self.assertEqual(limiter.check("request", user_id=10, guild_id=9, rule=rule).scope, "guild")
        self.assertTrue(limiter.check("request", user_id=10, guild_id=11, rule=rule).allowed)
        self.assertEqual(limiter.check("request", user_id=12, guild_id=13, rule=rule).scope, "global")

    def test_new_generation_resets_all_personal_actions_once(self):
        limiter = SlidingWindowLimiter(clock=lambda: 100)
        rule = RateLimitRule(user=1, guild=10, global_=10, window_seconds=60)
        for action in security_limits.DEFAULT_RULES:
            limiter.check(action, user_id=7, guild_id=9, rule=rule)
        limiter.reset_user(7, generation=2)
        for action in security_limits.DEFAULT_RULES:
            self.assertTrue(limiter.check(action, user_id=7, guild_id=9, rule=rule, user_generation=2).allowed)
        limiter.reset_user(7, generation=1)
        self.assertFalse(limiter.check("request", user_id=7, guild_id=9, rule=rule).allowed)

    def test_central_gate_applies_bonuses_without_raising_global_limit(self):
        service = SimpleNamespace(benefits=lambda *args: PremiumBenefits(user_multiplier=4, guild_multiplier=2))
        limiter = SlidingWindowLimiter(clock=lambda: 100)
        with patch("services.premium.get_premium_service", return_value=service), patch.object(
            security_limits, "RATE_LIMITER", limiter
        ), patch.dict(security_limits.DEFAULT_RULES, {"request": RateLimitRule(1, 2, 3, 60)}):
            for _ in range(3):
                self.assertTrue(security_limits.check_rate_limit("request", user_id=7, guild_id=9).allowed)
            decision = security_limits.check_rate_limit("request", user_id=7, guild_id=9)
        self.assertEqual(decision.scope, "global")


class CurrentProductTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = PremiumStore(Path(directory.name) / "premium.db")
        self.now = 2_000_000.0
        self.service = PremiumService(self.store, clock=lambda: self.now)
        self.client = FakeClient()
        self.client.catalog = [sku(400, discord.SKUType.subscription, 256 | 4),
                               sku(500, discord.SKUType.consumable, 4),
                               sku(600, discord.SKUType.consumable, 4)]
        self.config = SKUConfig(unthrottled=400, donate=500, donate_more=600)
        await self.service.configure(self.client, self.config)

    def add(self, identifier=1, user_id=7, **kwargs):
        item = entitlement(identifier, 400, user_id=user_id, **kwargs)
        self.client.items[item.id] = item
        return item

    async def test_user_subscription_automatically_follows_only_buyer(self):
        item = self.add(ends_at=datetime.fromtimestamp(self.now + 1000, timezone.utc))
        await self.service.refresh(self.client)
        for guild in (9, 10, "DM"):
            benefits = self.service.benefits(7, guild)
            self.assertTrue(benefits.bypass_global_throttle)
            self.assertEqual(benefits.subscription_until, self.now + 1000)
            self.assertEqual(benefits.user_multiplier, 1)
            self.assertEqual(benefits.guild_multiplier, 1)
            self.assertEqual(benefits.reset_generation, 0)
            self.assertFalse(self.service.benefits(8, guild).bypass_global_throttle)
        item.consume.assert_not_awaited()
        with self.assertRaises(PremiumError):
            await self.service.redeem(self.client, 7, "unthrottled")

    async def test_subscription_must_be_current_owned_and_user_scoped(self):
        timestamp = lambda value: datetime.fromtimestamp(value, timezone.utc)
        for index, overrides in enumerate((
            {"ends_at": timestamp(self.now)}, {"starts_at": timestamp(self.now + 1)},
            {"deleted": True}, {"application_id": 99}, {"guild_id": 9},
        ), start=1):
            self.add(index, **overrides)
        await self.service.refresh(self.client)
        self.assertFalse(self.service.benefits(7, 9).bypass_global_throttle)

    async def test_expiry_outage_restart_and_refund_remove_subscription_access(self):
        item = self.add()
        await self.service.refresh(self.client)
        restarted = PremiumService(self.store, clock=lambda: self.now)
        self.assertFalse(restarted.benefits(7, 9).bypass_global_throttle)
        await restarted.refresh(self.client)
        self.assertTrue(restarted.benefits(7, 9).bypass_global_throttle)
        self.client.fetch_skus.side_effect = asyncio.TimeoutError()
        with self.assertRaises(asyncio.TimeoutError):
            await restarted.refresh(self.client, force=True)
        self.now += VERIFICATION_SECONDS
        self.assertFalse(restarted.benefits(7, 9).bypass_global_throttle)
        self.client.fetch_skus.side_effect = lambda: self.client.catalog
        await restarted.refresh(self.client, force=True)
        restarted.observe(item, 42, deleted=True)
        self.assertFalse(restarted.benefits(7, 9).bypass_global_throttle)
        await restarted.refresh(self.client, force=True)
        self.assertFalse(restarted.benefits(7, 9).bypass_global_throttle)

    async def test_gateway_update_during_rest_snapshot_wins_for_subscription(self):
        stale = self.add()
        updated = entitlement(1, 400, ends_at=datetime.fromtimestamp(self.now, timezone.utc))
        await self.service.refresh(self.client)

        async def racing(**kwargs):
            yield stale
            self.service.observe(updated, 42)

        self.client.entitlements = racing
        await self.service.refresh(self.client, force=True)
        self.assertFalse(self.service.benefits(7, 9).bypass_global_throttle)

    async def test_deleted_subscription_during_rest_is_not_restored(self):
        item = self.add()

        async def racing(**kwargs):
            yield item
            self.service.observe(item, 42, deleted=True)

        self.client.entitlements = racing
        await self.service.refresh(self.client)
        self.assertFalse(self.service.benefits(7, 9).bypass_global_throttle)

    async def test_unverified_gateway_catalog_cannot_grant_subscription(self):
        self.service.catalog.clear()
        self.service.observe(self.add(), 42)
        self.assertFalse(self.service.benefits(7, 9).bypass_global_throttle)

    async def test_donations_are_recorded_once_without_any_throttle_benefit(self):
        for identifier, sku_id, kind in ((1, 500, "donate"), (2, 600, "donate_more")):
            item = entitlement(identifier, sku_id)
            self.client.items[item.id] = item
            result = await self.service.redeem(self.client, 7, kind)
            self.assertEqual(result["kind"], kind)
            self.assertIsNone(result["grant_start"])
            self.assertIsNone(result["grant_end"])
            item.consume.assert_awaited_once()
            with self.assertRaises(PremiumError):
                await self.service.redeem(self.client, 7, kind)
            self.assertEqual(self.service.benefits(7, 9), PremiumBenefits())
        self.assertEqual(self.store.reset_generations(), {})
        self.assertEqual(self.store.grants(self.now), [])

    async def test_donation_with_lost_consume_response_recovers_without_second_charge(self):
        item = entitlement(1, 500)
        self.client.items[item.id] = item

        async def lost_response():
            item.consumed = True
            raise asyncio.TimeoutError()

        item.consume.side_effect = lost_response
        with self.assertRaises(PremiumError):
            await self.service.redeem(self.client, 7, "donate")
        restarted = PremiumService(self.store, clock=lambda: self.now)
        await restarted.redeem(self.client, 7, "donate")
        item.consume.assert_awaited_once()
        self.assertEqual(restarted.benefits(7, 9), PremiumBenefits())

    async def test_donation_belongs_to_purchasing_user(self):
        item = entitlement(1, 500, user_id=8)
        self.client.items[item.id] = item
        with self.assertRaises(PremiumError):
            await self.service.redeem(self.client, 7, "donate")
        item.consume.assert_not_awaited()

    async def test_subscription_rejects_guild_subscription_group_and_consumable(self):
        for wrong_type, flags in ((discord.SKUType.subscription, 128),
                                  (discord.SKUType.subscription_group, 256),
                                  (discord.SKUType.consumable, 4)):
            wrong = [sku(400, wrong_type, flags), *self.client.catalog[1:]]
            with self.assertRaises(PremiumError):
                validate_catalog(self.config, wrong, 42)

    async def test_new_configuration_survives_restart(self):
        self.assertEqual(PremiumStore(self.store.path).config(), self.config)

    async def test_gateway_create_activates_subscription_and_end_time_expires_it(self):
        item = self.add(ends_at=datetime.fromtimestamp(self.now + 5, timezone.utc))
        self.service.observe(item, 42)
        self.assertTrue(self.service.benefits(7, 9).bypass_global_throttle)
        self.now += 5
        self.assertFalse(self.service.benefits(7, 9).bypass_global_throttle)

    async def test_rest_removes_subscription_after_missed_gateway_delete(self):
        item = self.add()
        await self.service.refresh(self.client)
        self.client.items.pop(item.id)
        await self.service.refresh(self.client, force=True)
        self.assertFalse(self.service.benefits(7, 9).bypass_global_throttle)

    async def test_old_settings_schema_migrates_without_changing_saved_products(self):
        path = self.store.path.parent / "old-premium.db"
        with closing(sqlite3.connect(path)) as conn:
            with conn:
                conn.execute("CREATE TABLE premium_settings (id INTEGER PRIMARY KEY, guild_sku TEXT, user_sku TEXT, reset_sku TEXT)")
                conn.execute("INSERT INTO premium_settings VALUES(1,'100','200','300')")
        migrated = PremiumStore(path)
        self.assertEqual(migrated.config(), SKUConfig(100, 200, 300))
        migrated.save_config(self.config)
        self.assertEqual(PremiumStore(path).config(), self.config)

    async def test_subscriber_bypasses_only_shared_bot_counters_for_every_action(self):
        self.add()
        await self.service.refresh(self.client)
        limiter = SlidingWindowLimiter(clock=lambda: 100)
        rules = {action: RateLimitRule(2, 3, 1, 60) for action in security_limits.DEFAULT_RULES}
        with patch("services.premium.get_premium_service", return_value=self.service), patch.object(
            security_limits, "RATE_LIMITER", limiter
        ), patch.dict(security_limits.DEFAULT_RULES, rules):
            for action in rules:
                # An unrelated free user fills the bot-wide bucket.
                self.assertTrue(security_limits.check_rate_limit(action, user_id=8, guild_id=10).allowed)
                self.assertTrue(security_limits.check_rate_limit(action, user_id=7, guild_id=9).allowed)
                self.assertTrue(security_limits.check_rate_limit(action, user_id=7, guild_id=9).allowed)
                self.assertEqual(security_limits.check_rate_limit(action, user_id=7, guild_id=9).scope, "user")
                self.assertEqual(security_limits.check_rate_limit(action, user_id=9, guild_id=11).scope, "global")

    async def test_paid_requests_count_toward_global_and_guild_buckets(self):
        self.add()
        await self.service.refresh(self.client)
        limiter = SlidingWindowLimiter(clock=lambda: 100)
        with patch("services.premium.get_premium_service", return_value=self.service), patch.object(
            security_limits, "RATE_LIMITER", limiter
        ), patch.dict(security_limits.DEFAULT_RULES, {"request": RateLimitRule(10, 1, 1, 60)}):
            self.assertTrue(security_limits.check_rate_limit("request", user_id=7, guild_id=9).allowed)
            self.assertEqual(security_limits.check_rate_limit("request", user_id=7, guild_id=9).scope, "guild")
            self.assertEqual(security_limits.check_rate_limit("request", user_id=8, guild_id=10).scope, "global")
            self.assertTrue(security_limits.check_rate_limit("request", user_id=7, guild_id=10).allowed)


class PremiumCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_donation_confirmation_records_support_without_claiming_reset(self):
        service = SimpleNamespace(redeem=AsyncMock(return_value={"kind": "donate"}))
        view = RedemptionView(None, service, 7, "donate")
        interaction = SimpleNamespace(
            user=SimpleNamespace(id=7), response=SimpleNamespace(defer=AsyncMock()),
            edit_original_response=AsyncMock(),
        )
        await view.confirm.callback(interaction)
        text = interaction.edit_original_response.call_args.kwargs["content"]
        self.assertIn("donation was recorded", text)
        self.assertNotIn("were reset", text)
        service.redeem.assert_awaited_once_with(None, 7, "donate")

    async def test_current_shop_lists_correct_products_prices_and_purchase_ids(self):
        catalog = [sku(400, discord.SKUType.subscription, 256 | 4),
                   sku(500, discord.SKUType.consumable, 4),
                   sku(600, discord.SKUType.consumable, 4)]
        service = SimpleNamespace(
            refresh=AsyncMock(), config=SKUConfig(unthrottled=400, donate=500, donate_more=600),
            catalog={item.id: item for item in catalog}, user_boost=2, guild_boost=2,
        )
        cog = PremiumCog(SimpleNamespace(), service)
        interaction = SimpleNamespace(
            response=SimpleNamespace(defer=AsyncMock()), followup=SimpleNamespace(send=AsyncMock()),
        )
        await PremiumCog.shop.callback(cog, interaction)
        args = interaction.followup.send.call_args.kwargs
        self.assertEqual([field.name for field in args["embed"].fields],
                         ["fUnThrottled — $4.99/month", "Donate — $2.99", "Donate More — $9.99"])
        self.assertEqual([item.sku_id for item in args["view"].children], [400, 500, 600])
        self.assertIn("no throttle benefit", args["embed"].fields[1].value)

    async def test_confirmed_reset_is_delivered_once_and_view_is_removed(self):
        service = SimpleNamespace(redeem=AsyncMock(return_value={"kind": "reset"}))
        view = RedemptionView(None, service, 7, "reset")
        interaction = SimpleNamespace(
            user=SimpleNamespace(id=7), response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
            edit_original_response=AsyncMock(),
        )
        await view.confirm.callback(interaction)
        await view.confirm.callback(interaction)
        service.redeem.assert_awaited_once_with(None, 7, "reset")
        self.assertIsNone(interaction.edit_original_response.call_args.kwargs["view"])
        self.assertTrue(view.is_finished())

    async def test_shop_uses_native_sku_buttons_for_available_products(self):
        service = SimpleNamespace(
            refresh=AsyncMock(), config=SKUConfig(100, 200, 300), user_boost=2, guild_boost=2,
            catalog={item.id: item for item in FakeClient().catalog},
        )
        cog = PremiumCog(SimpleNamespace(), service)
        interaction = SimpleNamespace(
            response=SimpleNamespace(defer=AsyncMock()), followup=SimpleNamespace(send=AsyncMock()),
        )
        await PremiumCog.shop.callback(cog, interaction)
        view = interaction.followup.send.call_args.kwargs["view"]
        self.assertEqual([button.sku_id for button in view.children], [100, 200, 300])
        self.assertTrue(all(button.style == discord.ButtonStyle.premium for button in view.children))
        view.stop()

    async def test_non_owner_cannot_list_or_configure_skus(self):
        bot = SimpleNamespace(is_owner=AsyncMock(return_value=False), fetch_skus=AsyncMock())
        service = SimpleNamespace(configure=AsyncMock())
        cog = PremiumCog(bot, service)
        interaction = SimpleNamespace(user=SimpleNamespace(id=7), response=SimpleNamespace(send_message=AsyncMock()))
        await PremiumCog.skus.callback(cog, interaction)
        await PremiumCog.configure.callback(cog, interaction, "100", "200", "300")
        bot.fetch_skus.assert_not_awaited()
        service.configure.assert_not_awaited()

    async def test_confirmation_cannot_be_used_by_another_user(self):
        service = SimpleNamespace(redeem=AsyncMock())
        view = RedemptionView(None, service, 7, "reset")
        interaction = SimpleNamespace(user=SimpleNamespace(id=8), response=SimpleNamespace(send_message=AsyncMock()))
        self.assertFalse(await view.interaction_check(interaction))
        service.redeem.assert_not_awaited()
        view.stop()


if __name__ == "__main__":
    unittest.main()
