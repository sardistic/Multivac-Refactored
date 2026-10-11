import asyncio
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import discord
from PIL import Image

from bot.characters import CharacterCog
from providers import stability_generation
from services import character_catalog, character_renders, security_limits, usage_costs
from services.character_catalog import CharacterCatalog
from services.character_renders import CharacterRenderService, CharacterStore, CHARACTER_SKU_ID
from services.premium import PremiumError


def png():
    output = io.BytesIO()
    Image.new('RGB', (8, 8), 'blue').save(output, format='PNG')
    return output.getvalue()


def purchase(identifier=10, **fields):
    values = dict(id=identifier, user_id=7, guild_id=None, application_id=42,
                  sku_id=CHARACTER_SKU_ID, deleted=False, starts_at=None, ends_at=None, consume=AsyncMock())
    values.update(fields)
    return SimpleNamespace(**values)


class Client:
    application_id = 42
    def __init__(self):
        self.item = purchase()
        self.product = SimpleNamespace(id=CHARACTER_SKU_ID, application_id=42,
                                       type=discord.SKUType.durable, flags=discord.SKUFlags._from_value(4))
        self.fetch_skus = AsyncMock(side_effect=lambda: [self.product])
        self.fetch_entitlement = AsyncMock(side_effect=lambda identifier: self.item)
    async def entitlements(self, **kwargs):
        yield self.item


class CharacterRenderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)/'characters.db'
        self.store = CharacterStore(self.path)
        self.catalog = SimpleNamespace(choose=AsyncMock(return_value={
            'name': 'Naruto', 'description': 'Orange clothes', 'source': 'Kitsu', 'url': 'https://kitsu.io/characters/1', 'image': '',
        }))
        self.service = CharacterRenderService(self.store, catalog=self.catalog)
        self.client = Client()
        self.render = self.enterContext(patch.object(stability_generation, 'generate_gpt_image', new=AsyncMock(return_value=io.BytesIO(png()))))
        self.limit = self.enterContext(patch.object(security_limits, 'check_rate_limit', return_value=security_limits.RateLimitDecision(True)))
        self.old_context = usage_costs.get_request_context()
        self.addCleanup(lambda: usage_costs.set_request_context(**self.old_context))

    async def test_one_render_survives_restart_and_repeated_downloads(self):
        result = await self.service.render(self.client, 10, 7)
        restarted = CharacterRenderService(CharacterStore(self.path), catalog=self.catalog)
        again = await restarted.render(self.client, 10, 7)
        self.assertEqual(result['image'], again['image'])
        self.assertEqual(again['state'], 'ready')
        self.render.assert_awaited_once()
        self.client.item.consume.assert_not_awaited()
        self.assertEqual(self.render.await_args.kwargs['model'], 'gpt-image-2.5-sunburst')
        self.assertEqual(self.render.await_args.kwargs['quality'], 'low')
        self.assertEqual(self.render.await_args.kwargs['size'], '1024x1024')

    async def test_concurrent_purchase_events_cannot_render_twice(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def wait(*args, **kwargs):
            entered.set(); await release.wait(); return io.BytesIO(png())
        self.render.side_effect = wait
        first = asyncio.create_task(self.service.render(self.client, 10, 7))
        await entered.wait()
        with self.assertRaises(PremiumError):
            await self.service.render(self.client, 10, 7)
        release.set(); await first
        self.render.assert_awaited_once()

    async def test_foreign_expired_deleted_future_or_guild_purchase_never_renders(self):
        past = datetime(2000,1,1,tzinfo=timezone.utc)
        future = datetime(2100,1,1,tzinfo=timezone.utc)
        for changed in [dict(user_id=8),dict(application_id=99),dict(sku_id=1),dict(deleted=True),
                        dict(ends_at=past),dict(starts_at=future),dict(guild_id=55)]:
            with self.subTest(changed=changed):
                self.client.item = purchase(**changed)
                with self.assertRaises(PremiumError):
                    await self.service.render(self.client, 10, 7)
        self.render.assert_not_awaited()

    async def test_missing_entitlement_never_returns_saved_image(self):
        await self.service.render(self.client, 10, 7)
        self.client.item.deleted = True
        with self.assertRaises(PremiumError): await self.service.render(self.client, 10, 7)
        self.assertIsNone(self.store.row(10)['image'])

    async def test_refund_before_reservation_is_a_durable_tombstone(self):
        self.store.revoke(10)
        with self.assertRaises(PremiumError): await self.service.render(self.client, 10, 7)
        self.render.assert_not_awaited()

    async def test_refund_during_render_never_persists_or_delivers_image(self):
        async def refunded(*args, **kwargs):
            self.store.revoke(10); return io.BytesIO(png())
        self.render.side_effect = refunded
        with self.assertRaises(PremiumError): await self.service.render(self.client, 10, 7)
        self.assertEqual(self.store.row(10)['state'], 'revoked')
        self.assertIsNone(self.store.row(10)['image'])

    async def test_provider_failure_retries_same_character_with_bounded_attempts(self):
        self.render.return_value = None
        for _ in range(4):
            with self.assertRaises(PremiumError): await self.service.render(self.client, 10, 7)
        self.assertEqual(self.render.await_count, 3)
        self.catalog.choose.assert_awaited_once()
        self.assertEqual(self.store.row(10)['attempts'], 3)

    async def test_catalog_failure_does_not_spend_render_attempt(self):
        self.catalog.choose.side_effect = OSError()
        with self.assertRaises(OSError): await self.service.render(self.client, 10, 7)
        self.assertEqual(self.store.row(10)['attempts'], 0)
        self.render.assert_not_awaited()

    async def test_personal_guild_global_limits_preserve_purchase(self):
        self.limit.return_value = security_limits.RateLimitDecision(False, retry_after=30)
        with self.assertRaises(PremiumError): await self.service.render(self.client, 10, 7)
        self.assertEqual(self.store.row(10)['attempts'], 0)
        self.render.assert_not_awaited()

    async def test_full_provider_queue_does_not_spend_render_attempt(self):
        @asynccontextmanager
        async def full(user_id):
            raise security_limits.ProviderCapacityError('full')
            yield
        with patch.object(security_limits, 'provider_capacity', full):
            with self.assertRaises(security_limits.ProviderCapacityError):
                await self.service.render(self.client, 10, 7)
        self.assertEqual(self.store.row(10)['attempts'], 0)
        self.render.assert_not_awaited()
        self.limit.assert_not_called()

    async def test_failed_purchase_events_do_not_automatically_retry_provider(self):
        self.render.return_value = None
        with self.assertRaises(PremiumError):
            await self.service.render(self.client, 10, 7)
        cog = CharacterCog(self.client, self.service)
        await cog.on_entitlement_create(self.client.item)
        await asyncio.sleep(0)
        self.assertFalse(cog._tasks)
        self.assertEqual(self.render.await_count, 1)

    async def test_named_choice_is_frozen_at_reservation(self):
        self.store.choice(7, 'Sonic')
        self.store.reserve(self.client.item)
        self.store.choice(7, 'Mario')
        await self.service.render(self.client, 10, 7)
        self.catalog.choose.assert_awaited_once_with('Sonic')

    async def test_reference_download_failure_does_not_silently_drop_identity(self):
        self.catalog.choose.return_value['image']='https://example.org/photo.png'
        with patch.object(character_renders,'fetch_url_bytes_async',new=AsyncMock(side_effect=OSError())):
            with self.assertRaises(OSError): await self.service.render(self.client, 10, 7)
        self.render.assert_not_awaited()
        self.assertEqual(self.store.row(10)['attempts'],0)

    async def test_references_are_bounded_and_metadata_removed(self):
        self.catalog.choose.return_value['image']='https://example.org/photo.png'
        output=io.BytesIO(); Image.new('RGB',(1000,1000)).save(output,format='PNG')
        with patch.object(character_renders,'fetch_url_bytes_async',new=AsyncMock(return_value=SimpleNamespace(body=output.getvalue()))):
            await self.service.render(self.client,10,7)
        with Image.open(io.BytesIO(self.render.await_args.kwargs['references'][0])) as ref:
            self.assertEqual(ref.size,(384,384))

    async def test_wrong_sku_type_never_renders(self):
        self.client.product.type=discord.SKUType.consumable
        with self.assertRaises(PremiumError): await self.service.render(self.client,10,7)
        self.render.assert_not_awaited()

    async def test_dm_failure_leaves_cached_download_available(self):
        user=SimpleNamespace(send=AsyncMock(side_effect=OSError()))
        self.client.get_user=lambda identifier:user
        cog=CharacterCog(self.client,self.service)
        await cog._deliver(self.client.item)
        result=await self.service.render(self.client,10,7)
        self.assertTrue(result['image'])
        self.render.assert_awaited_once()
        self.assertTrue(result['delivery_attempted'])

    async def test_user_command_returns_owned_cached_image(self):
        await self.service.render(self.client,10,7)
        interaction=SimpleNamespace(user=SimpleNamespace(id=7),response=SimpleNamespace(defer=AsyncMock()),
                                    followup=SimpleNamespace(send=AsyncMock()))
        await CharacterCog.character.callback(CharacterCog(self.client,self.service),interaction)
        self.assertTrue(interaction.followup.send.await_args.kwargs['ephemeral'])
        self.assertEqual(interaction.followup.send.await_args.kwargs['file'].fp.read(),png())
        self.render.assert_awaited_once()

    async def test_purchase_command_can_save_a_name_before_checkout(self):
        self.client.item=purchase(user_id=8)
        interaction=SimpleNamespace(user=SimpleNamespace(id=7),response=SimpleNamespace(defer=AsyncMock()),
                                    followup=SimpleNamespace(send=AsyncMock()))
        await CharacterCog.character.callback(CharacterCog(self.client,self.service),interaction,'Sonic')
        self.client.item=purchase()
        await self.service.render(self.client,10,7)
        self.catalog.choose.assert_awaited_once_with('Sonic')
        self.assertIn('$0.99',interaction.followup.send.await_args.args[0])

    async def test_cancelled_render_releases_lease_and_restores_usage_context(self):
        usage_costs.set_request_context(user_id='previous')
        self.render.side_effect=asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError): await self.service.render(self.client,10,7)
        self.assertEqual(self.store.row(10)['state'],'failed')
        self.assertEqual(usage_costs.get_request_context()['user_id'],'previous')


class CatalogTests(unittest.IsolatedAsyncioTestCase):
    async def test_named_choice_works_without_external_catalog(self):
        catalog=CharacterCatalog()
        with patch.object(catalog,'_get',new_callable=AsyncMock) as fetch:
            record=await catalog.choose('Literally Anyone')
        self.assertEqual(record['name'],'Literally Anyone')
        fetch.assert_not_awaited()

    async def test_random_selection_uses_entire_catalog_and_caches_count(self):
        catalog=CharacterCatalog()
        data={'data':[{'id':'42','type':'characters','attributes':{'canonicalName':'Naruto','description':'hero','image':{'original':'https://example.org/a.png'}}}]}
        with patch.object(catalog,'_get',new=AsyncMock(side_effect=[{'meta':{'count':105906}},data,data])) as fetch, patch.object(
            character_catalog.secrets,'randbelow',return_value=105905
        ) as random:
            first=await catalog.choose(); await catalog.choose()
        self.assertEqual(first['name'],'Naruto')
        self.assertEqual(fetch.await_args_list[1].args,(105905,))
        self.assertEqual(fetch.await_count,3)
        random.assert_called_with(105906)

    async def test_invalid_count_or_missing_character_fails_without_inventing_identity(self):
        catalog=CharacterCatalog()
        with patch.object(catalog,'_get',new=AsyncMock(return_value={'meta':{'count':0}})):
            with self.assertRaises(ValueError): await catalog.choose()


if __name__=='__main__': unittest.main()
