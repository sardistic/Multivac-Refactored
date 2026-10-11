import asyncio
import base64
import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import discord

from bot.unattractive import EffectActivationView, UnattractiveCog
from services import server_effects, usage_costs, behavior_registry
from services.premium import PremiumError
from services.server_effects import EFFECT_SECONDS, STYLE_MARKER, ServerEffectStore, ServerEffectService
from test_premium import FakeClient, entitlement, sku


class ServerEffectTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)/'premium.db'
        self.now = 1000.0
        self.store = ServerEffectStore(self.path)
        self.service = ServerEffectService(self.store,clock=lambda:self.now,sku_id=400)
        self.client = FakeClient()
        self.client.catalog = [sku(400,discord.SKUType.consumable,4)]
        self.client.items = {10:entitlement(10,400)}
        self.guard = AsyncMock()
        old_context = usage_costs.get_request_context()
        self.addCleanup(lambda:usage_costs.set_request_context(**old_context))
        usage_costs.set_request_context(user_id='7',guild_id='40',intent='chat')
        service_patch = patch.object(server_effects,'get_server_effect_service',return_value=self.service)
        service_patch.start()
        self.addCleanup(service_patch.stop)

    async def activate(self,user_id=7,guild_id=40):
        return await self.service.activate(self.client,user_id,guild_id,before_consume=self.guard)

    async def test_exact_ten_minute_window_and_restart_persistence(self):
        result = await self.activate()
        self.assertEqual(result,{'guild_id':'40','grant_start':1000,'grant_end':1600})
        self.client.items[10].consume.assert_awaited_once()
        restarted = ServerEffectService(ServerEffectStore(self.path),clock=lambda:self.now,sku_id=400)
        self.now = 1599.999
        self.assertEqual(restarted.active_until(40),1600)
        self.now = 1600
        self.assertIsNone(restarted.active_until(40))

    async def test_effect_is_server_wide_and_other_servers_and_dms_are_unchanged(self):
        await self.activate()
        usage_costs.set_request_context(user_id='99',guild_id='40')
        self.assertIn(STYLE_MARKER,server_effects.visual_prompt('a cat'))
        for guild_id in ('41','DM',None):
            usage_costs.set_request_context(user_id='7',guild_id=guild_id)
            self.assertEqual(server_effects.visual_prompt('a cat'),'a cat')

    async def test_two_purchases_extend_window_by_exactly_ten_minutes_each(self):
        self.client.items[11] = entitlement(11,400,user_id=8)
        first, second = await asyncio.gather(self.activate(),self.activate(8))
        self.assertEqual(first['grant_end'],second['grant_start'])
        self.assertEqual(second['grant_end']-second['grant_start'],EFFECT_SECONDS)
        self.assertEqual(self.service.active_until(40),2200)
        self.now = 1600
        self.assertEqual(self.service.active_until(40),2200)

    async def test_no_duplicate_activation_after_consumption_or_expiry(self):
        await self.activate()
        self.now = 1800
        with self.assertRaises(PremiumError):
            await self.activate()
        self.client.items[10].consume.assert_awaited_once()
        self.assertIsNone(self.service.active_until(40))

    async def test_ambiguous_consume_recovery_is_bound_to_original_server(self):
        item = self.client.items[10]

        async def uncertain_consume():
            item.consumed = True
            raise OSError('response lost')

        item.consume = AsyncMock(side_effect=uncertain_consume)
        with self.assertRaises(PremiumError):
            await self.activate()
        self.assertIsNone(self.service.active_until(40))
        restarted = ServerEffectService(ServerEffectStore(self.path),clock=lambda:self.now,sku_id=400)
        with self.assertRaises(PremiumError):
            await restarted.activate(self.client,7,41,before_consume=self.guard)
        result = await restarted.activate(self.client,7,40,before_consume=self.guard)
        self.assertEqual(result['grant_end'],1600)
        item.consume.assert_awaited_once()

    async def test_failed_consume_does_not_start_clock_and_can_retry(self):
        item = self.client.items[10]
        item.consume.side_effect = OSError('not confirmed')
        with self.assertRaises(PremiumError):
            await self.activate()
        self.now += 100
        item.consume.side_effect = lambda:setattr(item,'consumed',True)
        result = await self.activate()
        self.assertEqual(result['grant_start'],1100)
        self.assertEqual(result['grant_end'],1700)

    async def test_foreign_invalid_expired_consumed_and_guild_scoped_entitlements_are_not_spent(self):
        cases = ({'user_id':9},{'application_id':99},{'sku_id':999},{'guild_id':40},
                 {'deleted':True},{'consumed':True},
                 {'starts_at':datetime.fromtimestamp(1100,timezone.utc)},
                 {'ends_at':datetime.fromtimestamp(900,timezone.utc)})
        for changes in cases:
            with self.subTest(changes=changes):
                self.client.items = {10:entitlement(10,changes.get('sku_id',400),**{k:v for k,v in changes.items() if k!='sku_id'})}
                with self.assertRaises(PremiumError):
                    await self.activate()
                self.client.items[10].consume.assert_not_awaited()

    async def test_unavailable_checkout_does_not_block_owned_purchase_redemption(self):
        self.client.catalog[0].flags = discord.SKUFlags._from_value(0)
        await self.activate()
        self.assertEqual(self.service.active_until(40),1600)

    async def test_wrong_product_type_and_disabled_product_do_not_spend_purchase(self):
        self.client.catalog = [sku(400,discord.SKUType.durable,4)]
        with self.assertRaises(PremiumError):
            await self.activate()
        self.client.items[10].consume.assert_not_awaited()
        self.service.sku_id = 0
        with self.assertRaises(PremiumError):
            await self.activate()

    async def test_channel_guard_failure_does_not_consume_or_create_grant(self):
        self.guard.side_effect = PremiumError('channel disabled')
        with self.assertRaises(PremiumError):
            await self.activate()
        self.client.items[10].consume.assert_not_awaited()
        self.assertIsNone(self.store.pending(7))

    async def test_refund_during_consume_prevents_effect(self):
        item = self.client.items[10]

        async def consume_and_refund():
            item.consumed = True
            self.service.observe(item,42,deleted=True)

        item.consume.side_effect = consume_and_refund
        with self.assertRaises(PremiumError):
            await self.activate()
        self.assertIsNone(self.service.active_until(40))

    async def test_refund_tombstone_prevents_stale_events_after_restart(self):
        self.store.revoke(10)
        restarted = ServerEffectService(ServerEffectStore(self.path),clock=lambda:self.now,sku_id=400)
        with self.assertRaises(PremiumError):
            await restarted.activate(self.client,7,40,before_consume=self.guard)
        self.client.items[10].consume.assert_not_awaited()

    async def test_refund_of_active_purchase_stops_style_and_preserves_future_grant_window(self):
        await self.activate()
        self.client.items[11] = entitlement(11,400)
        await self.activate()
        self.client.items[10].deleted = True
        await self.service.reconcile(self.client)
        self.assertIsNone(self.service.active_until(40))
        self.now = 1600
        self.assertEqual(self.service.active_until(40),2200)

    async def test_manager_stop_cancels_active_and_queued_time_without_allowing_reuse(self):
        await self.activate()
        self.client.items[11] = entitlement(11,400)
        await self.activate()
        self.store.stop(40)
        self.assertIsNone(self.service.active_until(40))
        self.now = 1700
        self.assertIsNone(self.service.active_until(40))
        with self.assertRaises(PremiumError):
            await self.activate()
        self.client.items[12] = entitlement(12,400)
        result = await self.activate()
        self.assertEqual(result['grant_start'],1700)

    async def test_style_is_added_once_without_mutating_multimodal_messages_or_model(self):
        await self.activate()
        messages = [{'role':'system','content':'Keep facts accurate'}, {'role':'user','content':[
            {'type':'text','text':'describe'}, {'type':'image_url','image_url':{'url':'https://example.com/a.png'}}]}]
        original = copy.deepcopy(messages)
        kwargs = {'model':'gpt-6-luna','max_tokens':400}
        args, result = server_effects.decorate_provider_call('chat.openai',(messages,),kwargs)
        self.assertEqual(messages,original)
        self.assertEqual(result,kwargs)
        self.assertEqual(args[0][0:2],messages)
        self.assertIn(STYLE_MARKER,args[0][-1]['content'])
        again, _ = server_effects.decorate_provider_call('chat.openai',args,result)
        self.assertEqual(len(again[0]),3)
        self.assertEqual(server_effects.visual_prompt(server_effects.visual_prompt('a cat')),server_effects.visual_prompt('a cat'))

    async def test_claude_and_gemini_context_receive_style_but_utility_calls_do_not(self):
        await self.activate()
        for name in ('chat.claude','chat.openai_plain','vision.openai'):
            args,_ = server_effects.decorate_provider_call(name,([],),{})
            self.assertIn(STYLE_MARKER,args[0][-1]['content'])
        _,kw = server_effects.decorate_provider_call('chat.gemini',('a question',),{'context':[{'role':'user','content':'prior'}]})
        self.assertEqual(len(kw['context']),2)
        for name in ('classifier','code.reflect','video.sora.status','video.sora.download'):
            self.assertEqual(server_effects.decorate_provider_call(name,([],),{}),(([],),{}))

    async def test_real_provider_dispatch_receives_style_only_while_active(self):
        await self.activate()
        fallback = AsyncMock(return_value='answer')
        await behavior_registry.invoke_provider('chat.openai_plain',fallback,[],model='gpt-6-luna')
        self.assertIn(STYLE_MARKER,fallback.await_args.args[0][-1]['content'])
        self.now = 1600
        fallback.reset_mock()
        await behavior_registry.invoke_provider('chat.openai_plain',fallback,[],model='gpt-6-luna')
        self.assertEqual(fallback.await_args.args,([],))

    async def test_final_reviewer_knows_style_is_intentional_without_losing_quality_checks(self):
        from bot import draft_verifier
        await self.activate()
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(
            {'action':'accept','reason':'accurate','research_query':'','revised_answer':''})))],usage=None)
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(draft_verifier,'get_openai_client',return_value=client),patch.object(draft_verifier,'_load_review_context',return_value=('','',[])):
            await draft_verifier.verify_chat_draft(user_id=7,display_name='test',prompt='What is two plus two?',draft='Four. What a dreary sum.',research_used=False)
        messages = create.await_args.kwargs['messages']
        self.assertIn('accuracy, useful answers and safety',messages[0]['content'])
        self.assertIn('"temporary_server_style": "unattractive"',messages[1]['content'])

    async def test_image_sdk_gets_visual_style_with_original_model_quality_and_references(self):
        from providers import stability_generation
        await self.activate()
        response = SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(b'image').decode())],usage=None)
        client = SimpleNamespace(images=SimpleNamespace(generate=AsyncMock(return_value=response),edit=AsyncMock(return_value=response)))
        with patch.object(stability_generation,'get_openai_image_client',return_value=client):
            await stability_generation.generate_gpt_image('a cat with transparent background',model='gpt-image-1-mini',quality='low')
            call = client.images.generate.await_args.kwargs
            self.assertIn(STYLE_MARKER,call['prompt'])
            self.assertIn('a cat',call['prompt'])
            self.assertEqual((call['model'],call['quality'],call['background']),('gpt-image-1-mini','low','transparent'))
            await stability_generation.generate_gpt_image('a cat',references=[b'reference'],model='gpt-image-2.5-sunburst',quality='low',size='1024x1024')
            call = client.images.edit.await_args.kwargs
            self.assertIn(STYLE_MARKER,call['prompt'])
            self.assertEqual(call['image'][0].getvalue(),b'reference')
            self.assertEqual((call['model'],call['quality'],call['size']),('gpt-image-2.5-sunburst','low','1024x1024'))

    async def test_async_contexts_do_not_leak_server_effect(self):
        await self.activate()

        async def check(guild_id):
            usage_costs.set_request_context(guild_id=guild_id)
            await asyncio.sleep(0)
            return server_effects.visual_prompt('cat')

        effected,normal = await asyncio.gather(check('40'),check('41'))
        self.assertIn(STYLE_MARKER,effected)
        self.assertEqual(normal,'cat')

    async def test_confirmation_is_locked_to_buyer_and_server(self):
        view = EffectActivationView(SimpleNamespace(),7,40)
        for user_id,guild_id in ((8,40),(7,41)):
            interaction = SimpleNamespace(user=SimpleNamespace(id=user_id),guild_id=guild_id,response=SimpleNamespace(send_message=AsyncMock()))
            self.assertFalse(await view.interaction_check(interaction))
        interaction.user.id,interaction.guild_id = 7,40
        self.assertTrue(await view.interaction_check(interaction))

    async def test_ordinary_member_cannot_stop_server_effect(self):
        await self.activate()
        interaction = SimpleNamespace(guild=SimpleNamespace(id=40),guild_id=40,
            channel=SimpleNamespace(),
            user=SimpleNamespace(id=8,guild_permissions=SimpleNamespace(manage_guild=False,administrator=False)),
            response=SimpleNamespace(defer=AsyncMock()),followup=SimpleNamespace(send=AsyncMock()))
        cog = UnattractiveCog(self.client,self.service)
        await UnattractiveCog.unattractive.callback(cog,interaction,'stop')
        self.assertIn('Manage Server',interaction.followup.send.await_args.args[0])
        self.assertEqual(self.service.active_until(40),1600)
        interaction.user.guild_permissions.manage_guild = True
        await UnattractiveCog.unattractive.callback(cog,interaction,'stop')
        self.assertIsNone(self.service.active_until(40))

    async def test_cog_rechecks_channel_and_quota_before_consumption(self):
        cog = UnattractiveCog(self.client,self.service)
        interaction = SimpleNamespace(guild=SimpleNamespace(id=40),guild_id=40,user=SimpleNamespace(id=7),channel=SimpleNamespace())
        with patch('bot.unattractive.server_policy.require_active_channel',new=AsyncMock(side_effect=PremiumError('disabled'))):
            with self.assertRaises(PremiumError):
                await cog.activate(interaction)
        self.client.items[10].consume.assert_not_awaited()
        with patch('bot.unattractive.server_policy.require_active_channel',new=AsyncMock()),patch('bot.unattractive.check_rate_limit',return_value=SimpleNamespace(allowed=False,retry_after=5)):
            with self.assertRaises(PremiumError):
                await cog.activate(interaction)
        self.client.items[10].consume.assert_not_awaited()

    async def test_text_shortcut_routes_to_effect_cog_without_classification_or_generation(self):
        from bot.app_actions import handle_app_text
        respond = AsyncMock()
        bot = SimpleNamespace(get_cog=lambda name:SimpleNamespace(respond=respond) if name=='Unattractive' else None)
        message = SimpleNamespace(author=SimpleNamespace(id=7),guild=SimpleNamespace(id=40),channel=SimpleNamespace(),reply=AsyncMock())
        with patch('bot.app_actions.check_rate_limit',return_value=SimpleNamespace(allowed=True)):
            self.assertTrue(await handle_app_text(bot,message,'unattractive activate'))
        self.assertEqual(respond.await_args.args[-1],'activate')
        self.assertFalse(await handle_app_text(bot,message,'describe an unattractive painting'))
