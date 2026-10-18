import asyncio
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

from aiohttp.test_utils import TestClient, TestServer

from services.portal import COOKIE, STATE_COOKIE, Portal, digest, manageable
from services.portal_store import PortalStore, admission_allowed
from services.portal_usage import overview, publish_reports, read_report


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / 'portal.db')
        self.store = PortalStore(self.path)
        self.guild = {'id': '123', 'name': 'A server'}
        self.user = {'id': '456', 'username': 'manager'}

    def tearDown(self):
        self.tmp.cleanup()

    def test_request_approve_revoke_and_audit_persist(self):
        self.store.request(self.guild, self.user, 'Use the bot to help our community.')
        self.assertFalse(self.store.allowed('123'))
        self.store.review('123', 'owner', 'approved', 'Welcome')
        self.assertTrue(PortalStore(self.path).allowed('123'))
        self.store.review('123', 'owner', 'revoked', 'Stopped access')
        self.assertFalse(self.store.allowed('123'))
        self.assertEqual([r['action'] for r in self.store.audit()], ['revoked','approved','requested'])
        with self.assertRaises(ValueError):
            self.store.request(self.guild, self.user, 'Trying to bypass the revoked decision')

    def test_duplicate_and_rejected_cooldown(self):
        self.store.request(self.guild, self.user, 'Community research bot')
        with self.assertRaises(ValueError):
            self.store.request(self.guild, self.user, 'Duplicate pending request')
        self.store.review('123', 'owner', 'rejected', '')
        with self.assertRaises(ValueError):
            self.store.request(self.guild, self.user, 'Retry immediately after rejection')
        with self.store.connection() as conn:
            conn.execute('UPDATE requests SET updated_at=0')
        self.store.request(self.guild, self.user, 'Revised community request')
        self.assertEqual(self.store.requests()[0]['status'], 'pending')

    def test_seed_is_explicit_one_time_and_preserves_approved(self):
        self.assertFalse(self.store.allowed('123'))
        self.store.seed_existing([self.guild], 'owner')
        self.assertTrue(self.store.allowed('123'))
        with self.assertRaises(ValueError):
            self.store.seed_existing([{'id':'999','name':'Unexpected join'}], 'owner')
        self.assertFalse(self.store.allowed('999'))

    def test_admission_missing_database_fails_closed_without_creation(self):
        missing = Path(self.tmp.name) / 'missing.db'
        self.assertFalse(admission_allowed(missing, '123'))
        self.assertFalse(missing.exists())
        self.assertTrue(admission_allowed(missing, 'DM'))
        self.assertTrue(admission_allowed('', '123'))
        self.store.seed_existing([self.guild], 'owner')
        self.assertTrue(admission_allowed(self.path, '123'))

    def test_review_validation_and_idempotence(self):
        with self.assertRaises(ValueError):
            self.store.review('missing', 'owner', 'approved', '')
        self.store.seed_existing([self.guild], 'owner')
        with self.assertRaises(ValueError):
            self.store.review('123','owner','rejected','')
        self.store.review('123','owner','approved','')
        self.assertEqual(len(self.store.audit()), 1)

    def test_manager_permissions(self):
        self.assertTrue(manageable({'owner':True}))
        self.assertTrue(manageable({'permissions':'32'}))
        self.assertTrue(manageable({'permissions':'8'}))
        self.assertFalse(manageable({'permissions':'1024'}))


class UsageTests(unittest.TestCase):
    def test_export_reads_uncheckpointed_wal_and_excludes_raw_metadata(self):
        from services import usage_costs
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / 'wal.db')
            with patch.object(usage_costs, 'DB_PATH', path):
                with usage_costs._conn_rw():
                    pass
            writer = sqlite3.connect(path)
            try:
                writer.execute('PRAGMA journal_mode=WAL')
                writer.execute("INSERT INTO usage_logs(ts_utc,model,prompt_tokens,completion_tokens,total_tokens,cost_usd,meta_json) VALUES(datetime('now'),'wal-model',2,3,5,.01,?)", (json.dumps({'guild_id':'123','private_field':'never-export'}),))
                writer.commit()
                published = str(Path(temp) / 'reports' / 'usage.json')
                publish_reports(path, published)
                self.assertEqual(read_report(published,30)['total']['tokens'],5)
                self.assertNotIn('never-export',Path(published).read_text())
            finally:
                writer.close()

    def test_aggregates_attribution_no_private_fields_and_readonly(self):
        from services import usage_costs
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / 'usage.db')
            with patch.object(usage_costs, 'DB_PATH', path), patch.object(usage_costs, 'METRICS_PATH', ''):
                usage_costs.set_request_context()
                usage_costs.record('model-a', {'prompt_tokens':7,'completion_tokens':3}, .01,
                                   meta={'guild_id':'123','user_id':'private-user','secret':'must-not-appear'})
                usage_costs.record('model-b', {'input_tokens':2,'output_tokens':3}, .02,
                                   meta={'guild_id':'DM'})
                usage_costs.record('model-a', {}, .005)
                with usage_costs._conn_rw() as conn:
                    conn.execute("UPDATE usage_logs SET meta_json='broken' WHERE id=3")
            before = Path(path).read_bytes()
            report = overview(path)
            self.assertEqual(report['total']['tokens'], 15)
            self.assertEqual(report['total']['calls'], 3)
            self.assertAlmostEqual(report['total']['estimated_usd'], .035)
            self.assertEqual({r['key'] for r in report['servers']}, {'123','DM','unattributed'})
            self.assertNotIn('private-user', json.dumps(report))
            self.assertNotIn('must-not-appear', json.dumps(report))
            self.assertEqual(Path(path).read_bytes(), before)
            published = str(Path(temp) / 'reports' / 'usage.json')
            publish_reports(path, published)
            self.assertEqual(read_report(published, 30)['total']['tokens'], 15)
            self.assertNotIn('private-user', Path(published).read_text())
            self.assertFalse(list(Path(published).parent.glob('tmp*')))
            with self.assertRaises(ValueError):
                overview(path, -1)


class PortalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.portal = Portal({'PORTAL_DB_PATH': str(Path(self.tmp.name)/'portal.db'),
                              'PORTAL_ORIGIN':'https://insufficient.dev','DISCORD_CLIENT_SECRET':'test-only-secret'})
        self.portal.client_id = '111'
        self.portal.owner_id = 'owner'
        self.client = TestClient(TestServer(self.portal.app()))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp.cleanup()

    def session_headers(self, user='owner', origin='https://insufficient.dev', csrf='csrf'):
        sid = 'session-' + user
        self.portal.sessions[digest(sid)] = dict(user={'id':user,'username':user}, token='private-test-token', csrf='csrf', expires=time.time()+3600)
        return {'Cookie':COOKIE+'='+sid, 'Origin':origin,'X-CSRF-Token':csrf}

    async def test_anonymous_and_non_owner_denied_without_querying_discord(self):
        with patch.object(self.portal,'guilds',new_callable=AsyncMock) as guilds:
            self.assertEqual((await self.client.get('/api/admin')).status, 401)
            self.assertEqual((await self.client.get('/api/admin', headers=self.session_headers('visitor'))).status,403)
            guilds.assert_not_awaited()

    async def test_owner_admin_loads_only_private_aggregate_snapshot(self):
        snapshot={'total':{'calls':3,'tokens':12},'generated_at':'2026-10-17T12:00:00+00:00','servers':[],'models':[],'daily':[]}
        self.portal.report_path=str(Path(self.tmp.name)/'usage.json')
        Path(self.portal.report_path).write_text(json.dumps({'30':snapshot}))
        with patch.object(self.portal,'guilds',new=AsyncMock(return_value=[{'id':'123','name':'Current','approximate_member_count':12,'private_field':'not-exposed'}])):
            response=await self.client.get('/api/admin',headers=self.session_headers())
            self.assertEqual(response.status,200)
            body=await response.json()
            self.assertEqual(body['usage']['total']['tokens'],12)
            self.assertEqual(body['guilds'],[{'id':'123','name':'Current','members':12}])
            self.assertNotIn('not-exposed',json.dumps(body))
        response=await self.client.get('/api/admin?days=1000',headers=self.session_headers())
        self.assertEqual(response.status,400)

    async def kofi_post(self, **fields):
        data = {'verification_token':'kofi-test-token','message_id':'m1','timestamp':'2026-10-05T15:00:00Z',
                'type':'Donation','amount':'3.50','currency':'USD','from_name':'Private Name','message':'private note',**fields}
        return await self.client.post('/api/kofi', data={'data': json.dumps(data)})

    async def test_kofi_webhook_requires_configuration_and_token(self):
        self.assertEqual((await self.kofi_post()).status, 503)
        self.portal.kofi_token = 'kofi-test-token'
        self.assertEqual((await self.kofi_post(verification_token='wrong')).status, 403)
        self.assertEqual((await self.kofi_post(amount='-1')).status, 400)
        self.assertEqual(self.portal.db.kofi_payments(), [])

    async def test_kofi_payments_feed_public_funding_without_identities(self):
        self.portal.kofi_token = 'kofi-test-token'
        self.portal.funding_path = str(Path(self.tmp.name)/'funding.json')
        Path(self.portal.funding_path).write_text(json.dumps({
            'fixed':[{'name':'Domain','usdPerMonth':1.25,'from':'2026-10','note':'insufficient.dev'}],
            'priorDonations':[{'month':'2026-09','usd':10,'count':2}],
            'paid':[{'month':'2026-09','provider':'openai','usd':30}],'undated':[{'name':'Subscriptions','usd':80}]}))
        self.assertEqual((await self.kofi_post()).status, 200)
        self.assertEqual((await self.kofi_post()).status, 200)                       # Ko-fi retries are idempotent
        self.assertEqual((await self.kofi_post(message_id='m2', amount='5', currency='EUR')).status, 200)
        response = await self.client.get('/api/public/funding')
        body = await response.json()
        self.assertEqual(body['donations']['months'], [{'month':'2026-09','usd':10.0,'count':2},{'month':'2026-10','usd':3.5,'count':1}])
        self.assertEqual(body['donations']['allTimeUsd'], 13.5)
        self.assertTrue(body['donations']['live'])
        self.assertEqual(body['fixed'][0]['usdPerMonth'], 1.25)
        self.assertEqual(body['paid'], [{'month':'2026-09','provider':'openai','usd':30.0}])
        self.assertEqual(body['undated'], [{'name':'Subscriptions','usd':80.0,'note':''}])
        self.assertNotIn('Private Name', json.dumps(body)); self.assertNotIn('private note', json.dumps(body))

    async def test_csrf_and_origin_required_for_reviews(self):
        self.portal.db.seed_existing([{'id':'123','name':'Existing'}],'owner')
        for origin,csrf in [('https://evil.example','csrf'),('https://insufficient.dev','wrong')]:
            response=await self.client.post('/api/admin/requests/123',headers=self.session_headers(origin=origin,csrf=csrf),json={'status':'revoked'})
            self.assertEqual(response.status,403)
            self.assertTrue(self.portal.db.allowed('123'))
        response=await self.client.post('/api/admin/requests/123',headers=self.session_headers(),json={'status':'revoked','note':'Review'})
        self.assertEqual(response.status,200)
        self.assertFalse(self.portal.db.allowed('123'))

    async def test_submission_uses_fresh_discord_permission_and_name(self):
        with patch.object(self.portal,'user_guilds',new=AsyncMock(return_value=[{'id':'123','name':'Real Discord name','permissions':'32'}])) as guilds:
            response=await self.client.post('/api/requests',headers=self.session_headers('manager'),json={'guild_id':'123','guild_name':'Forged','reason':'Help our community with research'})
            self.assertEqual(response.status,201)
            guilds.assert_awaited_once_with('private-test-token')
            self.assertEqual(self.portal.db.requests()[0]['guild_name'],'Real Discord name')
        with patch.object(self.portal,'user_guilds',new=AsyncMock(return_value=[{'id':'999','name':'No permission','permissions':'1024'}])):
            response=await self.client.post('/api/requests',headers=self.session_headers('manager'),json={'guild_id':'999','reason':'This user is no longer a manager'})
            self.assertEqual(response.status,403)

    async def test_pending_has_no_invite_and_approved_manager_can_get_link(self):
        guild={'id':'123','name':'Test','permissions':'32'}
        self.portal.db.request(guild,{'id':'manager','username':'Manager'},'Our server wants the bot')
        with patch.object(self.portal,'user_guilds',new=AsyncMock(return_value=[guild])):
            response=await self.client.get('/api/servers',headers=self.session_headers('manager'))
            self.assertIsNone((await response.json())['guilds'][0]['invite_url'])
            self.portal.db.review('123','owner','approved','Welcome')
            response=await self.client.get('/api/servers',headers=self.session_headers('manager'))
            url=(await response.json())['guilds'][0]['invite_url']
            query=parse_qs(urlparse(url).query)
            self.assertEqual(query['guild_id'],['123'])
            self.assertEqual(query['disable_guild_select'],['true'])
            self.assertFalse(int(query['permissions'][0]) & 8)
            self.assertEqual(query['scope'],['bot applications.commands'])

    async def test_state_cookie_binding_replay_and_secure_login(self):
        response=await self.client.get('/auth/discord',allow_redirects=False)
        self.assertEqual(response.status,302)
        query=parse_qs(urlparse(response.headers['Location']).query)
        self.assertEqual(query['scope'],['identify guilds'])
        self.assertEqual(query['redirect_uri'],['https://insufficient.dev/auth/discord/callback'])
        state=query['state'][0]
        cookie=response.cookies[STATE_COOKIE]
        self.assertTrue(cookie['secure'])
        self.assertTrue(cookie['httponly'])
        self.assertEqual(cookie['samesite'],'Lax')
        # Correct state without its browser cookie is rejected before exchange.
        response=await self.client.get('/auth/discord/callback?code=code&state='+state,allow_redirects=False)
        self.assertEqual(response.status,400)
        response=await self.client.get('/auth/discord/callback?code=code&state='+state,headers={'Cookie':STATE_COOKIE+'='+state},allow_redirects=False)
        self.assertEqual(response.status,400)

    async def test_successful_callback_rotates_session_and_does_not_expose_tokens(self):
        state='unique-state'
        self.portal.states[digest(state)]=time.time()+600
        response_mock=MagicMock()
        response_mock.status=200
        response_mock.json=AsyncMock(return_value={'access_token':'secret-access-token','expires_in':3600})
        context=MagicMock()
        context.__aenter__=AsyncMock(return_value=response_mock)
        context.__aexit__=AsyncMock(return_value=False)
        old=digest('old-session')
        self.portal.sessions[old]=dict(user={'id':'old'},expires=time.time()+600)
        with patch.object(self.portal.http,'post',return_value=context), patch.object(self.portal,'discord',new=AsyncMock(return_value={'id':'owner','username':'Owner'})):
            response=await self.client.get('/auth/discord/callback?state='+state+'&code=test-code',headers={'Cookie':STATE_COOKIE+'='+state+'; '+COOKIE+'=old-session'},allow_redirects=False)
        self.assertEqual(response.status,302)
        self.assertEqual(response.headers['Location'],'/admin')
        self.assertTrue(response.cookies[COOKIE]['secure'])
        self.assertNotIn(old,self.portal.sessions)
        sid=response.cookies[COOKIE].value
        me=await self.client.get('/api/me',headers={'Cookie':COOKIE+'='+sid})
        body=await me.json()
        self.assertTrue(body['admin'])
        self.assertNotIn('secret-access-token',json.dumps(body))
        self.assertNotIn(digest(state),self.portal.states)

    async def test_team_owner_uses_real_user_not_synthetic_application_owner(self):
        self.portal.owner_id=None
        with patch.object(self.portal,'discord',new=AsyncMock(return_value={'id':'111','owner':{'id':'synthetic'},'team':{'owner_user_id':'real-owner'}})):
            await self.portal.identity()
        self.assertEqual(self.portal.owner_id,'real-owner')

    async def test_oauth_configuration_mismatch_and_missing_secret_fail_closed(self):
        self.portal.owner_id=None
        with patch.object(self.portal,'discord',new=AsyncMock(return_value={'id':'different','owner':{'id':'owner'}})):
            response=await self.client.get('/auth/discord',allow_redirects=False)
            self.assertEqual(response.status,503)
        self.portal.secret=''
        self.assertEqual((await self.client.get('/auth/discord',allow_redirects=False)).status,503)

    async def test_security_headers_and_static_pages(self):
        for path in ('/','/admin','/request','/privacy','/terms','/policies','/static/portal.js'):
            response=await self.client.get(path)
            self.assertEqual(response.status,200,path)
            self.assertEqual(response.headers['Cache-Control'],'no-store')
            self.assertEqual(response.headers['X-Frame-Options'],'DENY')
            self.assertIn("script-src 'self'",response.headers['Content-Security-Policy'])
        self.assertEqual((await self.client.get('/.portal.env')).status,404)

    async def test_expired_session_logout_and_rate_limit(self):
        headers=self.session_headers()
        self.portal.sessions[digest('session-owner')]['expires']=0
        self.assertEqual((await self.client.get('/api/me',headers=headers)).status,401)
        headers=self.session_headers()
        self.assertEqual((await self.client.post('/api/logout',headers=headers,json={})).status,200)
        self.assertEqual((await self.client.get('/api/me',headers=headers)).status,401)
        self.portal.limits['127.0.0.1'].extend([time.time()]*60)
        self.assertEqual((await self.client.get('/api/me')).status,429)


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_approved_preserved_revoked_left_unknown_left_db_error_retries(self):
        from bot.invite_approval import InviteApprovalCog
        with tempfile.TemporaryDirectory() as temp:
            cog=InviteApprovalCog(SimpleNamespace(),str(Path(temp)/'portal.db'))
            guild=SimpleNamespace(id=123,leave=AsyncMock())
            cog.store.seed_existing([{'id':'123','name':'Existing'}],'owner')
            await cog.enforce(guild)
            guild.leave.assert_not_awaited()
            cog.store.review('123','owner','revoked','')
            await cog.enforce(guild)
            guild.leave.assert_awaited_once()
            unknown=SimpleNamespace(id=999,leave=AsyncMock())
            await cog.on_guild_join(unknown)
            unknown.leave.assert_awaited_once()
            guild.leave.reset_mock()
            with patch.object(cog.store,'allowed',side_effect=sqlite3.OperationalError('unavailable')):
                with self.assertLogs('bot.invite_approval',level='ERROR'):
                    await cog.enforce(guild)
            guild.leave.assert_not_awaited()

    async def test_interactions_block_unapproved_and_keep_previous_checks(self):
        from bot.invite_approval import InviteApprovalCog
        with tempfile.TemporaryDirectory() as temp:
            cog=InviteApprovalCog(SimpleNamespace(),str(Path(temp)/'portal.db'))
            cog.previous_interaction_check=AsyncMock(return_value=True)
            interaction=SimpleNamespace(guild_id=123,response=SimpleNamespace(send_message=AsyncMock()))
            self.assertFalse(await cog.interaction_check(interaction))
            cog.previous_interaction_check.assert_not_awaited()
            cog.store.seed_existing([{'id':'123','name':'Existing'}],'owner')
            self.assertTrue(await cog.interaction_check(interaction))
            cog.previous_interaction_check.assert_awaited_once_with(interaction)


if __name__ == '__main__':
    unittest.main()
