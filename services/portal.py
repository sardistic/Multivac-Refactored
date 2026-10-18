"""Self-hosted MultiVAC website and Discord owner portal (aiohttp)."""

import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlencode, urlparse
from zoneinfo import ZoneInfo

from aiohttp import ClientSession, ClientTimeout, web

from services.portal_store import PortalStore
from services.portal_usage import read_report

ROOT = Path(__file__).resolve().parents[1] / 'website'
API = 'https://discord.com/api/v10'
COOKIE = '__Host-multivac'
STATE_COOKIE = '__Host-multivac-oauth'
log = logging.getLogger(__name__)


def manageable(guild):
    return bool(guild.get('owner') or int(guild.get('permissions', 0)) & (8 | 32))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class Portal:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.origin = env.get('PORTAL_ORIGIN', 'https://insufficient.dev').rstrip('/')
        self.secret = env.get('DISCORD_CLIENT_SECRET', '')
        self.bot_token = env.get('DISCORD_TOKEN', '')
        self.client_id = env.get('DISCORD_CLIENT_ID', '')
        self.owner_id = None  # Always resolved from the actual bot application.
        self.db = PortalStore(env.get('PORTAL_DB_PATH', './state/portal/portal.db'))
        self.report_path = env.get('PORTAL_REPORT_PATH', './state/portal-metrics/usage.json')
        self.metrics_path = env.get('USAGE_METRICS_PATH', '')
        self.tz = env.get('USAGE_TZ', 'America/New_York')
        self.kofi_token = env.get('KOFI_VERIFICATION_TOKEN', '')
        self.funding_path = env.get('FUNDING_PATH', str(ROOT / 'funding.json'))
        self.sessions = {}
        self.states = {}
        self.limits = defaultdict(deque)
        self.inventory = (0, [])
        self.inventory_lock = asyncio.Lock()
        self.identity_lock = asyncio.Lock()
        self.http = None

    async def discord(self, method, path, *, token=None, data=None):
        auth = 'Bot ' + self.bot_token if token is None else 'Bearer ' + token
        async with self.http.request(method, API + path, headers={
            'Authorization': auth, 'User-Agent': 'DiscordBot (https://insufficient.dev, 1.0)',
        }, data=data) as response:
            if response.status >= 400:
                # Provider bodies can contain credentials and are never logged.
                raise web.HTTPBadGateway(text='Discord is unavailable. Please try again shortly.')
            return await response.json()

    async def identity(self):
        async with self.identity_lock:
            if not self.owner_id:
                app = await self.discord('GET', '/oauth2/applications/@me')
                owner = (app.get('team') or {}).get('owner_user_id') or app.get('owner', {}).get('id')
                if not owner:
                    raise web.HTTPServiceUnavailable(text='Application owner could not be verified.')
                if self.client_id and self.client_id != app['id']:
                    raise web.HTTPServiceUnavailable(text='OAuth application does not match the bot.')
                self.owner_id = owner
                self.client_id = app['id']

    async def guilds(self):
        async with self.inventory_lock:
            if time.time() - self.inventory[0] < 60:
                return self.inventory[1]
            result = await self.user_guilds(None, counts=True)
            self.inventory = (time.time(), result)
            return result

    async def user_guilds(self, token, counts=False):
        result, after = [], '0'
        for _ in range(10):
            page = await self.discord('GET', f'/users/@me/guilds?limit=200&after={after}&with_counts={str(counts).lower()}', token=token)
            result.extend(page)
            if len(page) < 200:
                return result
            after = max(page, key=lambda guild: int(guild['id']))['id']
        raise web.HTTPBadGateway(text='Too many Discord servers to load safely.')

    def session(self, request, admin=False):
        session = self.sessions.get(digest(request.cookies.get(COOKIE, '')))
        if not session or session['expires'] <= time.time():
            raise web.HTTPUnauthorized(text='Sign in with Discord to continue.')
        if admin and (not self.owner_id or session['user']['id'] != self.owner_id):
            raise web.HTTPForbidden(text='Only the Discord application owner can access this view.')
        if request.method == 'POST':
            if request.headers.get('Origin') != self.origin or not secrets.compare_digest(
                request.headers.get('X-CSRF-Token', ''), session['csrf'],
            ):
                raise web.HTTPForbidden(text='Session check failed. Reload the page and try again.')
        return session

    async def login(self, request):
        if not self.secret:
            raise web.HTTPServiceUnavailable(text='Discord sign-in is awaiting operator configuration. Please check back soon.')
        await self.identity()
        state = secrets.token_urlsafe(32)
        self.states[digest(state)] = time.time() + 600
        url = 'https://discord.com/oauth2/authorize?' + urlencode({
            'client_id': self.client_id, 'response_type': 'code',
            'redirect_uri': self.origin + '/auth/discord/callback',
            'scope': 'identify guilds', 'state': state,
        })
        response = web.HTTPFound(url)
        response.set_cookie(STATE_COOKIE, state, secure=True, httponly=True, samesite='Lax', max_age=600, path='/')
        return response

    async def callback(self, request):
        state = request.query.get('state', '')
        expiry = self.states.pop(digest(state), 0)
        if not state or expiry <= time.time() or not secrets.compare_digest(state, request.cookies.get(STATE_COOKIE, '')):
            raise web.HTTPBadRequest(text='Discord sign-in expired. Start again from the website.')
        if request.query.get('error'):
            raise web.HTTPBadRequest(text='Discord sign-in was cancelled. You can try again from the website.')
        if request.query.get('iss', 'https://discord.com') != 'https://discord.com':
            raise web.HTTPBadRequest(text='Unexpected sign-in provider.')
        code = request.query.get('code', '')
        if not code or not self.secret:
            raise web.HTTPBadRequest(text='Missing sign-in configuration or authorization code.')
        await self.identity()
        async with self.http.post(API + '/oauth2/token', data={
            'client_id': self.client_id, 'client_secret': self.secret,
            'grant_type': 'authorization_code', 'code': code,
            'redirect_uri': self.origin + '/auth/discord/callback',
        }) as response:
            if response.status != 200:
                raise web.HTTPBadGateway(text='Discord sign-in failed. Please start again.')
            tokens = await response.json()
        token = tokens['access_token']
        user = await self.discord('GET', '/users/@me', token=token)
        session_id = secrets.token_urlsafe(32)
        self.sessions[digest(session_id)] = {
            'user': {'id': user['id'], 'username': user['username']}, 'token': token,
            'csrf': secrets.token_urlsafe(32),
            'expires': time.time() + min(3600, int(tokens.get('expires_in', 3600))),
        }
        self.sessions.pop(digest(request.cookies.get(COOKIE, '')), None)
        response = web.HTTPFound('/admin' if user['id'] == self.owner_id else '/request')
        response.set_cookie(COOKIE, session_id, secure=True, httponly=True, samesite='Lax', max_age=3600, path='/')
        response.del_cookie(STATE_COOKIE, path='/', secure=True, httponly=True, samesite='Lax')
        return response

    async def me(self, request):
        session = self.session(request)
        await self.identity()
        return web.json_response(dict(user=session['user'], csrf=session['csrf'], admin=session['user']['id'] == self.owner_id))

    async def logout(self, request):
        self.session(request)
        self.sessions.pop(digest(request.cookies.get(COOKIE, '')), None)
        response = web.json_response({'ok': True})
        response.del_cookie(COOKIE, path='/', secure=True, httponly=True, samesite='Lax')
        return response

    async def request_list(self, request):
        session = self.session(request)
        guilds = [g for g in await self.user_guilds(session['token']) if manageable(g)]
        requests = {row['guild_id']: row for row in await asyncio.to_thread(self.db.requests)}
        result = []
        await self.identity()
        for guild in guilds:
            row = requests.get(guild['id'])
            status = row['status'] if row else 'not_requested'
            result.append(dict(id=guild['id'], name=guild['name'], status=status,
                               review_note=row['review_note'] if row else '',
                               invite_url=self.invite(guild['id']) if status == 'approved' else None))
        return web.json_response({'guilds': result})

    def invite(self, guild_id):
        return 'https://discord.com/oauth2/authorize?' + urlencode({
            'client_id': self.client_id, 'scope': 'bot applications.commands',
            'permissions': str(sum(1 << bit for bit in (6, 10, 11, 14, 15, 16, 18, 38))), 'guild_id': guild_id,
            'disable_guild_select': 'true', 'integration_type': '0',
        })

    async def submit(self, request):
        session = self.session(request)
        body = await request.json()
        guild_id = str(body.get('guild_id', ''))
        # Refresh authority from Discord on every submission; never trust a
        # submitted server name, cached permissions, or client-side filtering.
        guild = next((g for g in await self.user_guilds(session['token']) if g['id'] == guild_id and manageable(g)), None)
        if not guild:
            raise web.HTTPForbidden(text='You must currently have Manage Server or Administrator permission in that server.')
        await asyncio.to_thread(self.db.request, guild, session['user'], body.get('reason', ''))
        return web.json_response({'ok': True}, status=201)

    async def admin(self, request):
        self.session(request, admin=True)
        days = int(request.query.get('days', '30'))
        usage, guilds, requests, audit = await asyncio.gather(
            asyncio.to_thread(read_report, self.report_path, days), self.guilds(),
            asyncio.to_thread(self.db.requests), asyncio.to_thread(self.db.audit),
        )
        safe_guilds = [dict(id=g['id'], name=g['name'], members=g.get('approximate_member_count')) for g in guilds]
        return web.json_response(dict(usage=usage, guilds=safe_guilds, requests=requests, audit=audit,
                                      inventory_updated_at=self.inventory[0]))

    async def review(self, request):
        session = self.session(request, admin=True)
        body = await request.json()
        await asyncio.to_thread(self.db.review, request.match_info['guild_id'], session['user']['id'], body.get('status'), body.get('note', ''))
        return web.json_response({'ok': True})

    async def public_metrics(self, request):
        if not self.metrics_path:
            raise web.HTTPServiceUnavailable(text='Metrics are not configured.')
        try:
            data = json.loads(await asyncio.to_thread(Path(self.metrics_path).read_text))
        except (OSError, ValueError):
            raise web.HTTPServiceUnavailable(text='Metrics are temporarily unavailable.')
        try:
            data['guilds'] = len(await self.guilds())
        except web.HTTPException:
            if self.inventory[0]:
                data['guilds'] = len(self.inventory[1])
        return web.json_response(data)

    async def kofi(self, request):
        """Ko-fi's webhook: a form post whose `data` field is JSON carrying our verification token."""
        if not self.kofi_token:
            raise web.HTTPServiceUnavailable(text='Ko-fi is not configured.')
        data = json.loads((await request.post())['data'])
        if not secrets.compare_digest(str(data.get('verification_token', '')), self.kofi_token):
            raise web.HTTPForbidden(text='Verification failed.')
        try:
            cents = int((Decimal(str(data['amount'])) * 100).to_integral_value())
        except (InvalidOperation, KeyError):
            raise web.HTTPBadRequest(text='Invalid amount.')
        if not 0 < cents < 10_000_000:
            raise web.HTTPBadRequest(text='Invalid amount.')
        at = datetime.fromisoformat(str(data['timestamp']).replace('Z', '+00:00'))
        await asyncio.to_thread(self.db.record_kofi, data['message_id'], at.timestamp(),
                                data.get('type') or 'Donation', cents, data.get('currency') or 'USD')
        return web.json_response({'ok': True})

    async def public_funding(self, request):
        """What the bot costs to run and what supporters have given, month by month."""
        try:
            config = json.loads(await asyncio.to_thread(Path(self.funding_path).read_text, encoding='utf8'))
        except (OSError, ValueError):
            config = {}
        currency = str(config.get('currency') or 'USD').upper()
        tz, months = ZoneInfo(self.tz), {}
        for row in config.get('priorDonations') or []:
            entry = months.setdefault(str(row['month']), {'month': str(row['month']), 'usd': 0.0, 'count': 0})
            entry['usd'] += float(row.get('usd') or 0)
            entry['count'] += int(row.get('count') or 0)
        for row in await asyncio.to_thread(self.db.kofi_payments):
            if row['currency'] != currency:
                continue
            month = datetime.fromtimestamp(row['at'], timezone.utc).astimezone(tz).strftime('%Y-%m')
            entry = months.setdefault(month, {'month': month, 'usd': 0.0, 'count': 0})
            entry['usd'] += row['amount_cents'] / 100
            entry['count'] += 1
        donations = sorted(({**e, 'usd': round(e['usd'], 2)} for e in months.values()), key=lambda e: e['month'])
        return web.json_response({
            'currency': currency,
            'kofi': config.get('kofi') or 'https://ko-fi.com/sardistic/goal?g=32',
            'fixed': [{'name': str(f['name']), 'usdPerMonth': float(f['usdPerMonth']), 'from': f.get('from'),
                       'until': f.get('until'), 'note': f.get('note', '')} for f in config.get('fixed') or []],
            'donations': {'months': donations, 'allTimeUsd': round(sum(e['usd'] for e in donations), 2),
                          'live': bool(self.kofi_token)},
            # what providers actually billed, by month; it stands in for the bot's estimate where present
            'paid': [{'month': str(p['month']), 'provider': str(p['provider']), 'usd': round(float(p['usd']), 2)}
                     for p in config.get('paid') or []],
            'undated': [{'name': str(u['name']), 'usd': round(float(u['usd']), 2), 'note': str(u.get('note', ''))}
                        for u in config.get('undated') or []],
            'notes': {str(k): str(v) for k, v in (config.get('notes') or {}).items()},
        })

    async def page(self, request):
        name = request.match_info.get('page', 'index')
        return web.FileResponse(ROOT / (name + '.html'))

    async def health(self, request):
        return web.json_response({'ok': True})

    async def startup(self, app):
        self.http = ClientSession(timeout=ClientTimeout(total=15))
        try:
            await self.identity()
        except Exception:
            log.warning('Discord application verification deferred until next sign-in')

    async def cleanup(self, app):
        await self.http.close()

    def app(self):
        @web.middleware
        async def boundary(request, handler):
            now = time.time()
            for collection in (self.states, self.sessions):
                for key, val in list(collection.items()):
                    if (val['expires'] if isinstance(val, dict) else val) <= now:
                        collection.pop(key, None)
            try:
                if request.host.split(':')[0] == 'www.' + urlparse(self.origin).hostname:
                    raise web.HTTPPermanentRedirect(self.origin + str(request.rel_url))
                if request.path.startswith(('/api/', '/auth/')):
                    # Only the Cloudflare tunnel reaches this service; no host ports.
                    ip = request.headers.get('CF-Connecting-IP', request.remote or '')
                    for key in list(self.limits):
                        if not self.limits[key] or self.limits[key][-1] < now - 60:
                            self.limits.pop(key, None)
                    queue = self.limits[ip]
                    while queue and queue[0] < now - 60:
                        queue.popleft()
                    if len(queue) >= 60 or len(self.limits) > 10000 or len(self.sessions) > 2000 or len(self.states) > 2000:
                        raise web.HTTPTooManyRequests(text='Please wait a minute and try again.')
                    queue.append(now)
                response = await handler(request)
            except web.HTTPException as exc:
                response = exc
            except (ValueError, TypeError, KeyError):
                response = web.HTTPBadRequest(text='Invalid request. Check your entries and try again.')
            except Exception:
                # Do not log request bodies, tokens, query strings or raw errors.
                log.warning('Portal operation failed (%s)', request.path.split('/')[1])
                response = web.HTTPServiceUnavailable(text='This service is temporarily unavailable. Please try again shortly.')
            response.headers.update({
                'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                'Referrer-Policy': 'no-referrer', 'X-Frame-Options': 'DENY',
                'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            })
            if isinstance(response, web.HTTPException):
                raise response
            return response

        app = web.Application(middlewares=[boundary, web.normalize_path_middleware(append_slash=False, remove_slash=True)], client_max_size=8192)
        app.on_startup.append(self.startup)
        app.on_cleanup.append(self.cleanup)
        app.router.add_get('/health', self.health)
        app.router.add_get('/auth/discord', self.login)
        app.router.add_get('/auth/discord/callback', self.callback)
        app.router.add_get('/api/me', self.me)
        app.router.add_post('/api/logout', self.logout)
        app.router.add_get('/api/servers', self.request_list)
        app.router.add_post('/api/requests', self.submit)
        app.router.add_get('/api/admin', self.admin)
        app.router.add_post('/api/admin/requests/{guild_id}', self.review)
        app.router.add_get('/api/public/metrics', self.public_metrics)
        app.router.add_get('/api/public/funding', self.public_funding)
        app.router.add_post('/api/kofi', self.kofi)
        app.router.add_get('/', self.page)
        app.router.add_get('/{page:admin|request|terms|privacy|policies}', self.page)
        app.router.add_static('/static/', ROOT / 'static')
        return app


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # OAuth callback query strings carry authorization codes; disable access logs.
    web.run_app(Portal().app(), host='0.0.0.0', port=8080, access_log=None)
