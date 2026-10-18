
  // live figures from sardistic.com's own feeds: whether the bot is up, its commits, and the tokens it has used
  document.getElementById("yr").textContent = new Date().getFullYear();
  (async () => {
    const get = async u => { try { const r = await fetch(u, { cache: "no-store" }); return r.ok ? r.json() : null; } catch (e) { return null; } };
    const U = await get('/api/public/metrics'), W = U && U.windows;
    const status = document.getElementById('lv-status');
    if (U && U.generatedAt) {
      const fresh = Date.now() - Date.parse(U.generatedAt) < 15 * 60 * 1000;
      status.textContent = fresh ? 'Reporting' : 'Stale';
      if (fresh) status.classList.add('ok');
    } else status.textContent = 'Unavailable';
    if (W && W.allTime) document.getElementById('lv-tokens').textContent = W.allTime.totalTokens.toLocaleString('en-US');
    if (U && Number.isInteger(U.guilds)) { document.getElementById("lv-servers").textContent = U.guilds.toLocaleString("en-US"); document.getElementById("lv-servers-l").textContent = U.guilds === 1 ? "Server" : "Servers"; }
    if (W && W.allTime) {
      const $ = id => document.getElementById(id);
      const short = n => n >= 1e9 ? (n / 1e9).toFixed(2) + "B" : n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e4 ? Math.round(n / 1e3) + "K" : n.toLocaleString("en-US");
      const calls = n => n.toLocaleString("en-US") + (n === 1 ? " call" : " calls");
      for (const [w, id] of [["today", "u-today"], ["thisWeek", "u-week"], ["monthToDate", "u-month"], ["allTime", "u-all"]]) {
        const x = W[w]; if (!x) continue; $(id).textContent = short(x.totalTokens); $(id + "-c").textContent = calls(x.calls) + " · " + short(x.completionTokens) + " written";
      }
      const a = W.allTime, fresh = Math.max(0, a.promptTokens - a.cachedPromptTokens), parts = [
        ["Prompt (fresh)", fresh, "#1f4fa0"], ["Prompt (cached)", a.cachedPromptTokens, "#7fa0d8"], ["Cache writes", a.cacheWriteTokens, "#e2a21c"], ["Completion", a.completionTokens, "#d6302a"]];
      const sum = parts.reduce((t, p) => t + p[1], 0) || 1;
      $("u-split").innerHTML = parts.map(([, n, c]) => `<div style="width:${(n / sum * 100).toFixed(2)}%;background:${c}"></div>`).join("");
      $("u-legend").innerHTML = parts.map(([l, n, c]) => `<span><i style="background:${c}"></i>${l} ${short(n)} · ${(n / sum * 100).toFixed(0)}%</span>`).join("");
      const d = (U.daily && U.daily.days) || [];
      if (d.length) {
        const mx = Math.max(...d, 1), bw = 700 / d.length;
        $("u-spark").innerHTML = d.map((v, i) => { const h = Math.max(v ? 2 : 0, v / mx * 112); return `<rect x="${(i * bw + 3).toFixed(1)}" y="${(118 - h).toFixed(1)}" width="${(bw - 6).toFixed(1)}" height="${h.toFixed(1)}" fill="${i === d.length - 1 ? "#d6302a" : "currentColor"}" opacity="${i === d.length - 1 ? 1 : .78}"><title>${short(v)} tokens</title></rect>`; }).join("");
        $("u-from").textContent = U.daily.from || ""; $("u-to").textContent = U.daily.to || "";
      }
      const ago = iso => { const s = (Date.now() - Date.parse(iso)) / 1000; return !isFinite(s) ? "" : s < 3600 ? Math.max(1, Math.round(s / 60)) + " min ago" : s < 86400 ? Math.round(s / 3600) + " h ago" : Math.round(s / 86400) + " days ago"; };
      $("u-asof").textContent = [U.lastActivityAt ? "Last used " + ago(U.lastActivityAt) : "", U.generatedAt ? "figures updated " + ago(U.generatedAt) : ""].filter(Boolean).join(" · ");
      $("usage").hidden = false;
    }
  })();

  // what it costs and what people have given: OpenAI as billed, the rest as the bot's own estimate, fixed costs, and Ko-fi
  (async () => {
    const get = async u => { try { const r = await fetch(u, { cache: "no-store" }); return r.ok ? r.json() : null; } catch (e) { return null; } };
    const [U, F] = await Promise.all([get("/api/public/metrics"), get("/api/public/funding")]);
    if (!F) return;
    const C = (U && U.costs) || { months: [] };
    const $ = id => document.getElementById(id), usd = n => "$" + (Math.round(n * 100) / 100).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const signed = n => (n < 0 ? "−" : "+") + usd(Math.abs(n));
    const name = m => new Date(m + "-15T12:00:00Z").toLocaleString("en-US", { month: "short", year: "numeric", timeZone: "UTC" });
    const now = new Date().toLocaleString("sv-SE", { timeZone: (U && U.timezone) || "America/New_York" }).slice(0, 7);
    const est = Object.fromEntries((C.months || []).map(m => [m.month, m.providers || {}]));
    const given = Object.fromEntries((F.donations.months || []).map(m => [m.month, m]));
    const paid = {}; for (const p of F.paid || []) (paid[p.month] = paid[p.month] || {})[p.provider] = (paid[p.month][p.provider] || 0) + p.usd;
    const fixedIn = m => (F.fixed || []).filter(f => (!f.from || f.from <= m) && (!f.until || m <= f.until)).reduce((t, f) => t + f.usdPerMonth, 0);
    const months = [...new Set([...Object.keys(est), ...Object.keys(given), ...Object.keys(paid)])].filter(m => m <= now).sort().reverse();
    if (!months.length) return;
    // a provider's month: what it billed if we have the bill, otherwise the bot's estimate (marked);
    // before the bot kept a ledger, a provider with no bills on file is unknown rather than zero
    const ledgerFrom = Object.keys(est).sort()[0] || now, billed = p => (F.paid || []).some(x => x.provider === p);
    const cell = (m, p) => paid[m] && paid[m][p] != null ? { v: paid[m][p], est: false } : est[m] && est[m][p] ? { v: est[m][p], est: true }
      : m < ledgerFrom && !billed(p) ? { v: 0, unknown: true } : { v: 0, est: false };
    const COLS = [["openai", "OpenAI"], ["google", "Google"], ["anthropic", "Anthropic"], ["other", "Other"]].filter(([p]) => months.some(m => cell(m, p).v));
    const tot = { spent: 0, given: 0, fixed: 0 }; COLS.forEach(([p]) => tot[p] = 0);
    const rows = months.map((m, i) => {
      const cs = COLS.map(([p]) => cell(m, p)), fx = fixedIn(m), g = given[m] ? given[m].usd : 0, spent = cs.reduce((t, c) => t + c.v, 0) + fx, net = g - spent;
      COLS.forEach(([p], k) => tot[p] += cs[k].v); tot.fixed += fx; tot.spent += spent; tot.given += g;
      return `<tr${i >= 12 ? ' class="older" hidden' : ""}><td>${name(m)}${m === now ? "<small>SO FAR</small>" : ""}</td>${cs.map(c => c.unknown ? `<td class="dim" title="No bills on file for this month">—</td>` : `<td class="${c.v ? (c.est ? "est" : "") : "dim"}">${usd(c.v)}${c.v && c.est ? "<small>EST</small>" : ""}</td>`).join("")}<td class="${fx ? "" : "dim"}">${usd(fx)}</td><td class="sum">${usd(spent)}</td><td class="${g ? "pos" : "dim"}">${usd(g)}</td><td class="${net >= 0 ? "pos" : "neg"}">${signed(net)}</td></tr>`;
    });
    const und = (F.undated || []).reduce((t, u) => t + u.usd, 0); tot.spent += und;
    const tnet = tot.given - tot.spent, older = months.length > 12;
    $("f-table").innerHTML = `<thead><tr><th>Month</th>${COLS.map(c => `<th>${c[1]}</th>`).join("")}<th>Hosting &amp; fixed</th><th>Spent</th><th>Given</th><th>Difference</th></tr></thead><tbody>${rows.join("")}`
      + (older ? `<tr class="more"><td colspan="${COLS.length + 5}"><button type="button" id="f-more">Show all ${months.length} months, back to ${name(months[months.length - 1])}</button></td></tr>` : "")
      + (und ? (F.undated || []).map(u => `<tr class="und"><td>Undated</td><td colspan="${COLS.length + 1}">${u.name}${u.note ? " · " + u.note : ""}</td><td class="sum">${usd(u.usd)}</td><td></td><td class="neg">${signed(-u.usd)}</td></tr>`).join("") : "")
      + `<tr class="total"><td>All time</td>${COLS.map(([p]) => `<td>${usd(tot[p])}</td>`).join("")}<td>${usd(tot.fixed)}</td><td>${usd(tot.spent)}</td><td>${usd(tot.given)}</td><td class="${tnet >= 0 ? "pos" : "neg"}">${signed(tnet)}</td></tr></tbody>`;
    if (older) $("f-more").addEventListener("click", e => { document.querySelectorAll("#f-table tr.older").forEach(r => r.hidden = false); e.target.closest("tr").remove(); });
    const curCells = COLS.map(([p]) => cell(now, p).v), curSpent = curCells.reduce((t, v) => t + v, 0) + fixedIn(now), curGiven = given[now] ? given[now].usd : 0;
    const tips = (F.donations.months || []).reduce((t, m) => t + (m.count || 0), 0), first = months[months.length - 1];
    $("f-given").textContent = usd(tot.given); $("f-given-c").textContent = (tips ? tips + (tips === 1 ? " tip" : " tips") + " on Ko-fi" : "on Ko-fi") + " since " + name(first);
    $("f-spent").textContent = usd(tot.spent); $("f-spent-c").textContent = usd(tot.openai) + " OpenAI · " + usd(tot.spent - tot.openai) + " everything else";
    $("f-month-l").textContent = name(now) + " so far"; $("f-month").textContent = usd(curSpent); $("f-month-c").textContent = usd(curGiven) + " given";
    $("f-net").textContent = signed(tnet); $("f-net-c").textContent = tnet < 0 ? "paid out of pocket" : "ahead, thanks to you";
    $("f-net").className = tnet < 0 ? "neg" : "pos";
    if (F.kofi) $("f-kofi").href = F.kofi;
    const fixedNotes = (F.fixed || []).map(f => f.name + " " + usd(f.usdPerMonth) + "/mo" + (f.note ? " (" + f.note + ")" : "")).join(", ");
    $("f-note").textContent = [(F.notes && F.notes.paid) || "", "Amounts marked EST are the bot's own estimate from token counts and published prices.", fixedNotes ? "Fixed: " + fixedNotes + "." : "", F.donations.live ? "New Ko-fi tips show up here as they arrive." : ""].filter(Boolean).join(" ");
    $("funding").hidden = false;
  })();
