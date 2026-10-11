import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import discord
from PIL import Image

from bot import app_actions, intent_dispatcher
from bot.characters import CharacterCog
from bot.fallback_consent import confirm_provider_fallback
from bot.model_defaults import ModelDefaultsView
from bot.server_settings import ServerSettingsCog
from providers import stability_generation
from services import model_preferences, premium, security_limits, server_policy, usage_costs
from services.character_renders import CharacterRenderService, CharacterStore, CHARACTER_SKU_ID
from services.premium import PremiumBenefits, PremiumError
from services.sqlite_store import SQLiteStore


def request(user_id=7, *, manager=False, guild_id=40):
    guild = SimpleNamespace(id=guild_id, me=SimpleNamespace(id=99)) if guild_id else None
    permissions = discord.Permissions(view_channel=True, send_messages=True, send_messages_in_threads=True)
    channel = SimpleNamespace(id=5, guild=guild, permissions_for=lambda member: permissions, history=history)
    user = SimpleNamespace(id=user_id, bot=False, guild_permissions=discord.Permissions(manage_guild=manager))
    return SimpleNamespace(user=user, author=user, guild=guild, channel=channel,
                           response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock(), edit_message=AsyncMock(), send_modal=AsyncMock()),
                           followup=SimpleNamespace(send=AsyncMock()), reply=AsyncMock(), content='')


async def history(**kwargs):
    for row in []:
        yield row


def png():
    result = io.BytesIO()
    Image.new('RGB', (8, 8)).save(result, format='PNG')
    result.seek(0)
    return result


class GuildPolicyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.store = SQLiteStore(self.path)
        self.enterContext(patch.object(server_policy, 'store', self.store))
        self.enterContext(patch.object(model_preferences, 'store', self.store))
        self.limiter = security_limits.SlidingWindowLimiter()
        self.enterContext(patch.object(security_limits, 'RATE_LIMITER', self.limiter))
        self.enterContext(patch.object(premium, 'get_premium_service', return_value=SimpleNamespace(benefits=lambda *args: PremiumBenefits())))
        self.old_context = usage_costs.get_request_context()
        self.addCleanup(lambda: usage_costs.set_request_context(**self.old_context))
        self.bot = SimpleNamespace(user=SimpleNamespace(id=99))

    async def test_policy_persists_and_caps_personal_choices_only_in_its_guild(self):
        model_preferences.set_preferences(7, 'image', 'quality')
        server_policy.update_policy(40, workload='image', tier='economy', max_tier='balanced', throttle_percent=50)
        with server_policy.server_scope(40):
            self.assertEqual(model_preferences.image_choice(7).quality, 'medium')
            self.assertEqual(model_preferences.image_choice(8).model, 'gpt-image-1-mini')
        with server_policy.server_scope(41):
            self.assertEqual(model_preferences.image_choice(7).quality, 'high')
        self.assertEqual(model_preferences.get_preferences(7), {'image':'quality'})
        with patch.object(server_policy, 'store', SQLiteStore(self.path)):
            self.assertEqual(server_policy.get_policy(40)['throttle_percent'], 50)

    async def test_auto_clears_defaults_and_caps_without_touching_other_workloads(self):
        server_policy.update_policy(40, tier='economy', max_tier='balanced')
        server_policy.update_policy(40, workload='image', tier='auto', max_tier='auto')
        settings = server_policy.get_policy(40)
        self.assertNotIn('image', settings['ceilings'])
        self.assertNotIn('image', settings['defaults'])
        self.assertEqual(settings['ceilings']['chat'], 'balanced')

    async def test_ceiling_alone_cannot_upgrade_existing_cheaper_text_routes(self):
        server_policy.update_policy(40,workload='chat',max_tier='balanced')
        with server_policy.server_scope(40):
            tier=model_preferences.preference_tier(7,'chat')
            self.assertEqual(model_preferences.text_model_for_tier(tier,'gpt-6-luna',user_id=7),'gpt-6-luna')
            self.assertEqual(model_preferences.text_model_for_tier(tier,'gpt-6-astra',user_id=7),'gpt-6.1-sol')
            model_preferences.set_preferences(7,'chat','quality')
            self.assertEqual(model_preferences.text_model_for_tier(tier,'gpt-6-luna',user_id=7),'gpt-6.1-sol')

    async def test_invalid_server_changes_cannot_raise_limits_or_partially_apply(self):
        for value in (0,101,-1,1.5,True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                server_policy.update_policy(40, tier='quality', throttle_percent=value)
        self.assertEqual(server_policy.get_policy(40)['defaults'], {})

    async def test_only_manage_server_or_admin_can_update(self):
        outsider = request()
        await ServerSettingsCog.settings.callback(ServerSettingsCog(), outsider, tier='economy')
        self.assertEqual(server_policy.get_policy(40)['defaults'], {})
        manager = request(manager=True)
        await ServerSettingsCog.settings.callback(ServerSettingsCog(), manager, workload='chat', tier='economy', max_tier='balanced', throttle_percent=25)
        self.assertEqual(server_policy.get_policy(40)['defaults']['chat'], 'economy')
        manager.user.guild_permissions = discord.Permissions(administrator=True)
        await ServerSettingsCog.settings.callback(ServerSettingsCog(), manager, throttle_percent=100)
        self.assertEqual(server_policy.get_policy(40)['throttle_percent'],100)

    async def test_cross_server_channel_cannot_be_changed(self):
        manager = request(manager=True)
        await ServerSettingsCog.settings.callback(ServerSettingsCog(), manager, tier='economy', channel_enabled=False, channel=request(guild_id=41).channel)
        self.assertEqual(server_policy.get_policy(40)['defaults'], {})
        self.assertEqual(server_policy.channel_state(41,5),0)

    async def test_stricter_server_member_limit_does_not_reduce_other_guild_allowance(self):
        server_policy.update_policy(40, throttle_percent=50)
        for _ in range(2):
            self.assertTrue(security_limits.check_rate_limit('image',user_id=7,guild_id=40).allowed)
        self.assertEqual(security_limits.check_rate_limit('image',user_id=7,guild_id=40).scope,'guild_user')
        self.assertTrue(security_limits.check_rate_limit('image',user_id=7,guild_id=41).allowed)
        self.assertTrue(security_limits.check_rate_limit('image',user_id=7,guild_id=41).allowed)
        self.assertEqual(security_limits.check_rate_limit('image',user_id=7,guild_id=41).scope,'user')

    async def test_global_bypass_does_not_bypass_server_caps(self):
        server_policy.update_policy(40, throttle_percent=50)
        with patch.object(premium,'get_premium_service',return_value=SimpleNamespace(benefits=lambda *args: PremiumBenefits(bypass_global_throttle=True))):
            for _ in range(2): security_limits.check_rate_limit('image',user_id=7,guild_id=40)
            self.assertEqual(security_limits.check_rate_limit('image',user_id=7,guild_id=40).scope,'guild_user')

    async def test_activity_survives_restart_and_disabled_channel_cannot_reactivate(self):
        item = request()
        with self.assertRaises(PremiumError): await server_policy.require_active_channel(self.bot,item.guild,item.channel)
        server_policy.note_activity(40,5)
        with patch.object(server_policy,'store',SQLiteStore(self.path)):
            await server_policy.require_active_channel(self.bot,item.guild,item.channel)
        server_policy.set_channel_enabled(40,5,False)
        server_policy.note_activity(40,5)
        with self.assertRaises(PremiumError): await server_policy.require_active_channel(self.bot,item.guild,item.channel)
        server_policy.set_channel_enabled(40,5,True)
        await server_policy.require_active_channel(self.bot,item.guild,item.channel)

    async def test_channel_enable_does_not_invent_previous_activity(self):
        server_policy.set_channel_enabled(40,5,True)
        item=request()
        with self.assertRaises(PremiumError): await server_policy.require_active_channel(self.bot,item.guild,item.channel)

    async def test_history_import_accepts_bot_reply_to_normal_request_only(self):
        item=request()
        original=SimpleNamespace(author=item.user,content='<@99> hello')
        previous=SimpleNamespace(author=SimpleNamespace(id=99),reference=SimpleNamespace(resolved=original),content='Hello!',webhook_id=None)
        async def past(**kwargs): yield previous
        item.channel.history=past
        await server_policy.require_active_channel(self.bot,item.guild,item.channel)
        self.assertEqual(server_policy.channel_state(40,5),1)
        server_policy.set_channel_enabled(40,6,True)
        item.channel.id=6;original.content='<@99> $1 random plz'
        with self.assertRaises(PremiumError): await server_policy.require_active_channel(self.bot,item.guild,item.channel)

    async def test_bot_must_still_have_channel_permissions(self):
        item=request();server_policy.note_activity(40,5)
        item.channel.permissions_for=lambda member: discord.Permissions.none()
        with self.assertRaises(PremiumError): await server_policy.require_active_channel(self.bot,item.guild,item.channel)

    async def test_dm_recovery_does_not_require_active_guild(self):
        await server_policy.require_active_channel(self.bot,None,None)

    async def test_uncached_server_interaction_is_not_mistaken_for_a_dm(self):
        item=request(guild_id=None);item.guild_id=40
        service=SimpleNamespace(purchase=AsyncMock())
        await CharacterCog(self.bot,service).handle_interaction(item)
        service.purchase.assert_not_awaited()
        with patch.object(stability_generation,'generate_gpt_image',new_callable=AsyncMock) as generate:
            await app_actions.AppActionsCog(self.bot).generate(item,'a cat')
            generate.assert_not_awaited()

    async def test_real_mention_ingress_routes_paid_shortcut_before_intent_classifier(self):
        import discord_bot
        item=request(guild_id=None)
        item.id=1001;item.content='<@99> $1 random plz';item.reference=None;item.mention_everyone=False
        paid=SimpleNamespace(respond=AsyncMock())
        reflection=SimpleNamespace(observe_message=AsyncMock(return_value=set()))
        with patch.object(discord_bot.bot._connection,'user',self.bot.user), patch.object(discord_bot.bot,'get_cog',return_value=paid), \
             patch.object(discord_bot,'_already_processed',return_value=False), patch.object(discord_bot,'_get_reflection_worker',return_value=reflection), \
             patch.object(discord_bot,'should_suppress_response',new=AsyncMock(return_value=False)), \
             patch.object(discord_bot,'classify_intent',new_callable=AsyncMock) as classify:
            await discord_bot._builtin_on_message(item)
        paid.respond.assert_awaited_once_with(7,None,item.channel,item.reply,'random')
        classify.assert_not_awaited()

    async def test_literal_unattractive_prefix_uses_app_flow_without_legacy_command_or_classifier(self):
        import discord_bot
        for content, action in (('/unattractive','shop'),('/unattractive activate','activate'),('/unattractive action:status','status')):
            with self.subTest(content=content):
                item = request()
                item.id=1002; item.content=content
                effect = SimpleNamespace(respond=AsyncMock())
                reflection = SimpleNamespace(observe_message=AsyncMock(return_value=set()))
                with patch.object(discord_bot.bot,'get_cog',return_value=effect), \
                     patch.object(discord_bot,'_already_processed',return_value=False), \
                     patch.object(discord_bot,'_get_reflection_worker',return_value=reflection), \
                     patch.object(discord_bot.bot,'process_commands',new_callable=AsyncMock) as legacy, \
                     patch.object(discord_bot,'classify_intent',new_callable=AsyncMock) as classify:
                    await discord_bot._builtin_on_message(item)
                effect.respond.assert_awaited_once_with(item.author,item.guild,item.channel,item.reply,action)
                legacy.assert_not_awaited()
                classify.assert_not_awaited()

    async def test_other_prefix_commands_still_use_legacy_handler(self):
        import discord_bot
        item = request(); item.id=1003; item.content='/help'
        reflection = SimpleNamespace(observe_message=AsyncMock(return_value=set()))
        with patch.object(discord_bot,'_already_processed',return_value=False), \
             patch.object(discord_bot,'_get_reflection_worker',return_value=reflection), \
             patch.object(discord_bot.bot,'process_commands',new_callable=AsyncMock) as legacy:
            await discord_bot._builtin_on_message(item)
        legacy.assert_awaited_once_with(item)

    async def test_unknown_text_command_returns_guidance_without_internal_error(self):
        import discord_bot
        from discord.ext import commands
        context = SimpleNamespace(command=None,reply=AsyncMock())
        with patch.object(discord_bot.logger,'error') as log_error:
            await discord_bot._builtin_on_command_error(context,commands.CommandNotFound('missing'))
        text = context.reply.await_args.args[0]
        self.assertIn("Discord's / menu",text)
        self.assertNotIn('CommandNotFound',text)
        log_error.assert_not_called()

    async def test_history_does_not_count_unattractive_checkout_as_normal_activity(self):
        item = request()
        original = SimpleNamespace(author=item.user,content='/unattractive')
        previous = SimpleNamespace(author=SimpleNamespace(id=99),reference=SimpleNamespace(resolved=original),content='Buy Unattractive here',webhook_id=None)
        async def past(**kwargs): yield previous
        item.channel.history = past
        with self.assertRaises(PremiumError):
            await server_policy.require_active_channel(self.bot,item.guild,item.channel)
        self.assertEqual(server_policy.channel_state(40,5),0)

    async def test_explicit_external_provider_and_fallback_cannot_bypass_ceiling(self):
        server_policy.update_policy(40,workload='chat',max_tier='economy')
        item=request();item.content='claude hello'
        context=intent_dispatcher.DispatchContext(intent='claude_chat',message=item,prompt=item.content,raw_prompt=item.content,user_id=7,bot_user=self.bot.user,response_gate_checked=True)
        with patch.object(intent_dispatcher,'handle_claude_chat_intent',new_callable=AsyncMock) as handle:
            self.assertFalse(await intent_dispatcher.dispatch_intent(context))
            handle.assert_not_awaited()
        usage_costs.set_request_context(guild_id='40',intent='chat')
        self.assertFalse(await confirm_provider_fallback(item,model='Gemini',reason='failed'))

    async def test_capped_video_config_rejects_pro_veo_and_longer_durations(self):
        server_policy.update_policy(40,workload='video',max_tier='economy')
        self.assertTrue(server_policy.video_config_allowed(dict(provider='sora',model='sora-2',seconds=4),40))
        for provider,model,seconds in [('sora','sora-2',8),('sora','sora-2-pro',4),('veo','veo',4)]:
            self.assertFalse(server_policy.video_config_allowed(dict(provider=provider,model=model,seconds=seconds),40))

    async def test_default_dropdown_cannot_edit_another_account(self):
        view=ModelDefaultsView(7,40)
        outsider=request(8)
        await view.tier_select.callback(outsider)
        self.assertEqual(model_preferences.get_preferences(7),{})
        view.workload='image';view.tier_select._values=['economy']
        await view.tier_select.callback(request())
        self.assertEqual(model_preferences.get_preferences(7),{'image':'economy'})
        view.stop()

    async def test_text_purchase_alias_uses_paid_flow_and_never_bills_itself(self):
        paid=SimpleNamespace(respond=AsyncMock())
        self.bot.get_cog=lambda name:paid
        item=request()
        self.assertTrue(await app_actions.handle_app_text(self.bot,item,'$1 random plz'))
        paid.respond.assert_awaited_once_with(7,item.guild,item.channel,item.reply,'random')
        self.assertFalse(await app_actions.handle_app_text(self.bot,item,'what does $1 mean?'))

    async def test_text_settings_enforce_manager_permission_and_account_scope(self):
        await app_actions.handle_app_text(self.bot,request(),'server throttle 25')
        self.assertEqual(server_policy.get_policy(40)['throttle_percent'],100)
        await app_actions.handle_app_text(self.bot,request(manager=True),'server throttle 25')
        self.assertEqual(server_policy.get_policy(40)['throttle_percent'],25)
        await app_actions.handle_app_text(self.bot,request(),'model defaults image economy')
        self.assertEqual(model_preferences.get_preferences(7),{'image':'economy'})
        self.assertEqual(model_preferences.get_preferences(8),{})

    async def test_image_app_gate_blocks_inactive_channel_before_provider(self):
        cog=app_actions.AppActionsCog(self.bot)
        with patch.object(stability_generation,'generate_gpt_image',new_callable=AsyncMock) as generate:
            await cog.generate(request(),'a cat')
            generate.assert_not_awaited()

    async def test_image_app_applies_current_server_maximum_and_personal_default(self):
        model_preferences.set_preferences(7,'image','quality')
        server_policy.update_policy(40,workload='image',max_tier='balanced')
        server_policy.note_activity(40,5)
        with patch.object(stability_generation,'generate_gpt_image',new=AsyncMock(return_value=png())) as generate:
            item=request();await app_actions.AppActionsCog(self.bot).generate(item,'a cat')
            self.assertEqual(generate.await_args.kwargs['model'],'gpt-image-2.5-sunburst')
            self.assertEqual(generate.await_args.kwargs['quality'],'medium')
            self.assertTrue(item.followup.send.await_args.kwargs['ephemeral'])

    async def test_paid_offer_is_unavailable_in_inactive_channel_without_entitlement_queries(self):
        service=SimpleNamespace(purchase=AsyncMock())
        item=request()
        await CharacterCog(self.bot,service).respond(7,item.guild,item.channel,item.reply,'random')
        service.purchase.assert_not_awaited()

    async def test_apps_menus_register_serialize_and_remove_for_both_surfaces(self):
        self.bot.tree=app_commands_tree=discord.app_commands.CommandTree(discord.Client(intents=discord.Intents.none()))
        cog=app_actions.AppActionsCog(self.bot)
        await cog.cog_load()
        labels={'Character portrait $0.99','Generate image','Model defaults'}
        for command_type in (discord.AppCommandType.user, discord.AppCommandType.message):
            menus=app_commands_tree.get_commands(type=command_type)
            self.assertEqual({c.name for c in menus}, labels)
            serialized=[c.to_dict(app_commands_tree) for c in menus]
            self.assertEqual({(c['name'],c['type']) for c in serialized}, {(name,command_type.value) for name in labels})
        await cog.cog_unload()
        self.assertFalse(app_commands_tree.get_commands(type=discord.AppCommandType.user))
        self.assertFalse(app_commands_tree.get_commands(type=discord.AppCommandType.message))

    async def test_profile_image_opens_blank_requester_form_without_rendering(self):
        server_policy.note_activity(40,5)
        cog=app_actions.AppActionsCog(self.bot)
        cog.generate=AsyncMock()
        item=request()
        await cog.image_profile_menu(item,SimpleNamespace(id=8))
        modal=item.response.send_modal.await_args.args[0]
        self.assertEqual(modal.prompt.default,'')
        self.assertEqual(modal.user_id,7)
        cog.generate.assert_not_awaited()
        outsider=request(8)
        await modal.on_submit(outsider)
        cog.generate.assert_not_awaited()
        outsider.response.send_message.assert_awaited_once()
        modal.stop()

    async def test_profile_image_denies_inactive_and_disabled_channels(self):
        cog=app_actions.AppActionsCog(self.bot)
        for enabled in (False,True):
            if enabled:
                server_policy.note_activity(40,5)
                server_policy.set_channel_enabled(40,5,False)
            item=request()
            await cog.image_profile_menu(item,SimpleNamespace(id=8))
            item.response.send_modal.assert_not_awaited()
            self.assertTrue(item.response.send_message.await_args.kwargs['ephemeral'])

    async def test_message_image_still_prefills_description_without_mentions(self):
        server_policy.note_activity(40,5)
        item=request()
        await app_actions.AppActionsCog(self.bot).image_menu(item,SimpleNamespace(content='<@99> a cat'))
        modal=item.response.send_modal.await_args.args[0]
        self.assertEqual(modal.prompt.default,'a cat')
        self.assertEqual(modal.user_id,7)
        modal.stop()

    async def test_profile_paid_action_uses_requester_existing_purchase_flow(self):
        paid=CharacterCog(self.bot,SimpleNamespace())
        paid.respond=AsyncMock()
        self.bot.get_cog=lambda name:paid
        item=request()
        await app_actions.AppActionsCog(self.bot).character_profile_menu(item,SimpleNamespace(id=8))
        paid.respond.assert_awaited_once_with(7,item.guild,item.channel,item.followup.send,None,private=True)
        self.assertTrue(item.response.defer.await_args.kwargs['ephemeral'])

    async def test_profile_defaults_open_requesters_own_controls(self):
        self.bot.get_cog=lambda name:app_actions.ModelDefaultsCog(self.bot)
        model_preferences.set_preferences(8,'image','quality')
        item=request()
        await app_actions.AppActionsCog(self.bot).defaults_profile_menu(item,SimpleNamespace(id=8))
        view=item.followup.send.await_args.kwargs['view']
        self.assertEqual(view.user_id,7)
        self.assertTrue(item.followup.send.await_args.kwargs['ephemeral'])
        self.assertEqual(model_preferences.get_preferences(7),{})
        self.assertEqual(model_preferences.get_preferences(8),{'image':'quality'})
        view.stop()

    async def test_paid_sunburst_maximum_blocks_new_render_but_dm_recovery_works(self):
        service = CharacterRenderService(CharacterStore(self.path/'characters.db'), catalog=SimpleNamespace(choose=AsyncMock(return_value={
            'name':'Sonic','description':'','source':'Named selection','url':'','image':''})))
        product = SimpleNamespace(id=CHARACTER_SKU_ID, application_id=42, type=discord.SKUType.consumable)
        entitlement = SimpleNamespace(id=10,user_id=7,sku_id=CHARACTER_SKU_ID,application_id=42,guild_id=None,deleted=False,starts_at=None,ends_at=None,consumed=False,consume=AsyncMock())
        client = SimpleNamespace(application_id=42,fetch_skus=AsyncMock(return_value=[product]),fetch_entitlement=AsyncMock(return_value=entitlement))
        server_policy.update_policy(40,workload='image',max_tier='economy')
        with patch.object(stability_generation,'generate_gpt_image',new=AsyncMock(return_value=png())) as generate:
            with self.assertRaises(PremiumError): await service.render(client,10,7,guild_id=40)
            generate.assert_not_awaited()
            self.assertEqual(service.store.row(10)['attempts'],0)
            row = await service.render(client,10,7)
            self.assertEqual(row['state'],'ready')
            self.assertEqual(generate.await_args.kwargs['model'],'gpt-image-2.5-sunburst')
            # Cached downloads need no model permission or new inference.
            await service.render(client,10,7,guild_id=40)
            generate.assert_awaited_once()

    async def test_async_guild_contexts_remain_isolated(self):
        server_policy.update_policy(40,workload='chat',max_tier='economy')
        server_policy.update_policy(41,workload='chat',max_tier='balanced')
        async def read(guild_id):
            with server_policy.server_scope(guild_id):
                before=model_preferences.preference_tier(7,'chat')
                await asyncio.sleep(0)
                return before,model_preferences.preference_tier(7,'chat')
        self.assertEqual(await asyncio.gather(read(40),read(41)),[('economy','economy'),('balanced','balanced')])
        self.assertIsNone(server_policy.current_guild())


if __name__=='__main__': unittest.main()
