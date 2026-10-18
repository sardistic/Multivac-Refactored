'use strict';
const $ = id => document.getElementById(id);
const number = value => Number(value || 0).toLocaleString('en-US');
const money = value => new Intl.NumberFormat('en-US', {style:'currency', currency:'USD', maximumFractionDigits:4}).format(value || 0);
const date = value => new Date(value * 1000).toLocaleString();
let account;
function el(tag, value, className) {
  const node = document.createElement(tag);
  if (value !== undefined) node.textContent = value;
  if (className) node.className = className;
  return node;
}
function notice(message) { $('notice').textContent = message; $('notice').hidden = !message; }
async function api(path, body) {
  const response = await fetch(path, {method: body ? 'POST' : 'GET', cache:'no-store', headers: body ? {'Content-Type':'application/json', 'X-CSRF-Token':account.csrf} : {}, body:body ? JSON.stringify(body) : undefined});
  if (!response.ok) { const error = new Error(await response.text()); error.status = response.status; throw error; }
  return response.json();
}
function table(target, headers, rows) {
  const result = el('table'), head = el('thead'), tr = el('tr'), body = el('tbody');
  headers.forEach(text => tr.append(el('th', text)));
  head.append(tr); result.append(head);
  rows.forEach(row => { const line = el('tr'); row.forEach(text => line.append(el('td', text))); body.append(line); });
  if (!rows.length) { const line = el('tr'), cell = el('td', 'No activity in this period.'); cell.colSpan = headers.length; line.append(cell); body.append(line); }
  result.append(body); $(target).replaceChildren(result);
}
function badge(status) { return el('span', status.replaceAll('_',' '), 'badge ' + status); }
async function busy(button, action) {
  button.disabled = true; notice('');
  try { await action(); } catch (error) { notice(error.message); } finally { button.disabled = false; }
}
async function loadAdmin() {
  const data = await api('/api/admin?days=' + $('days').value), usage = data.usage;
  $('stats').replaceChildren();
  for (const [name, value] of [['Installed servers',number(data.guilds.length)],['Provider calls',number(usage.total.calls)],['Total tokens',number(usage.total.tokens)],['Estimated cost',money(usage.total.estimated_usd)]]) {
    const card = el('div', undefined, 'stat'); card.append(el('strong', value),el('span', name)); $('stats').append(card);
  }
  const age = Date.now() - Date.parse(usage.generated_at);
  $('updated').textContent = (age > 180000 ? 'STALE — ' : '') + 'Usage updated ' + new Date(usage.generated_at).toLocaleString();
  $('requests').replaceChildren();
  const pendingFirst = [...data.requests].sort((a,b) => (a.status !== 'pending') - (b.status !== 'pending'));
  if (!pendingFirst.length) $('requests').append(el('p','No server requests yet.'));
  for (const row of pendingFirst) {
    const card = el('article',undefined,'request-card'); card.append(el('h3',row.guild_name),badge(row.status),el('p',row.reason),el('p',`Requested by ${row.user_name} · ${date(row.created_at)} · Server ${row.guild_id}`,'meta'));
    if (row.review_note) card.append(el('p','Review note: ' + row.review_note));
    const label = el('label','Review note (visible to this server’s managers)'), note = el('textarea'); note.maxLength = 1000; note.rows = 2; label.append(note); card.append(label);
    const actions = el('div',undefined,'toolbar');
    for (const [text,status] of [['Approve','approved'],['Reject','rejected'],['Revoke access','revoked']]) {
      if (status === row.status || (status === 'rejected' && row.status !== 'pending') || (status === 'revoked' && row.status !== 'approved')) continue;
      const button = el('button',text);
      button.addEventListener('click', () => busy(button, async () => {
        if (status === 'revoked' && !confirm(`Revoke ${row.guild_name}? MultiVAC will leave this server within a minute.`)) return;
        await api('/api/admin/requests/' + encodeURIComponent(row.guild_id), {status, note:note.value}); await loadAdmin();
      })); actions.append(button);
    }
    card.append(actions); $('requests').append(card);
  }
  const current = new Map(data.guilds.map(g => [g.id,g])), serverUsage = new Map(usage.servers.map(g => [g.key,g]));
  const serverIds = new Set([...current.keys(),...serverUsage.keys()]);
  const zero = {calls:0,input_tokens:0,output_tokens:0,cached_tokens:0,tokens:0,estimated_usd:0};
  table('servers',['Server','Installation','Members','Calls','Tokens','Estimated cost'],[...serverIds].map(id => {
    const g = current.get(id), u = serverUsage.get(id) || zero;
    const special = id === 'DM' || id === 'unattributed';
    return [g ? `${g.name} (${id})` : id === 'DM' ? 'Direct messages' : id === 'unattributed' ? 'Unattributed / background' : 'Historical server ' + id, g ? 'Installed' : special ? '—' : 'No longer installed',g && g.members != null ? number(g.members) : '—',number(u.calls),number(u.tokens),money(u.estimated_usd)];
  }));
  table('models',['Model','Calls','Input','Cached input','Output','Tokens','Estimated cost'],usage.models.map(row => [row.key,number(row.calls),number(row.input_tokens),number(row.cached_tokens),number(row.output_tokens),number(row.tokens),money(row.estimated_usd)]));
  $('daily').replaceChildren();
  const daily = new Map(usage.daily.map(row => [row.key,row.tokens]));
  const start = new Date(usage.since), end = new Date(usage.generated_at), dates = [];
  start.setUTCHours(0,0,0,0);
  while (start <= end) { const key = start.toISOString().slice(0,10); dates.push([key,daily.get(key) || 0]); start.setUTCDate(start.getUTCDate()+1); }
  const max = Math.max(1,...dates.map(row => row[1]));
  for (const [day,tokens] of dates) { const bar = el('div'); bar.style.height = Math.max(tokens ? 2 : 0,tokens/max*100) + '%'; bar.title = `${day}: ${number(tokens)} tokens`; $('daily').append(bar); }
  $('daily-caption').textContent = `${dates[0]?.[0] || ''} — ${dates.at(-1)?.[0] || ''} · daily totals in UTC`;
  table('audit',['When','Server','Decision','Actor','Note'],data.audit.map(row => [date(row.at),row.guild_id,row.action,row.actor_id,row.note]));
}
async function loadRequests() {
  const data = await api('/api/servers');
  $('your-servers').replaceChildren(); $('guild').replaceChildren();
  for (const guild of data.guilds) {
    const card = el('article',undefined,'request-card'); card.append(el('h3',guild.name),badge(guild.status));
    if (guild.review_note) card.append(el('p',guild.review_note));
    if (guild.invite_url) { const link = el('a','Add MultiVAC to this server →','btn'); link.href = guild.invite_url; card.append(el('p','Your server is approved.'),link); }
    if (['not_requested','rejected'].includes(guild.status)) { const option = el('option',guild.name); option.value = guild.id; $('guild').append(option); }
    $('your-servers').append(card);
  }
  if (!data.guilds.length) $('your-servers').append(el('p','No eligible servers found. You need Manage Server or Administrator permission to request an invite.'));
  $('request-section').hidden = !$('guild').options.length;
}
async function init() {
  try { account = await api('/api/me'); }
  catch(error) { if (error.status !== 401) notice(error.message); $('signin').hidden = false; return; }
  $('account').textContent = account.user.username; $('logout').hidden = false;
  $('logout').addEventListener('click', () => busy($('logout'),async () => { await api('/api/logout',{}); location.assign('/'); }));
  const admin = document.body.dataset.view === 'admin';
  if (admin && !account.admin) { notice('This overview is available to the MultiVAC owner. You can request access for a server you manage.'); return; }
  $('content').hidden = false;
  if ($('admin-link')) $('admin-link').hidden = !account.admin;
  const load = admin ? loadAdmin : loadRequests;
  $('refresh').addEventListener('click', () => busy($('refresh'),load));
  if (admin) $('days').addEventListener('change', () => busy($('refresh'),load));
  else $('request-form').addEventListener('submit',event => { event.preventDefault(); busy(event.submitter,async () => { await api('/api/requests',{guild_id:$('guild').value,reason:$('reason').value}); $('reason').value = ''; await loadRequests(); notice('Your request has been submitted. Check back here for the owner’s decision.'); }); });
  await busy($('refresh'),load);
}
init().catch(error => notice(error.message));
