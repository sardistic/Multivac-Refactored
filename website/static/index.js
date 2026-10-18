
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

  // what it costs and what people have given: the bot's ledger (estimated API spend by month), fixed costs, and Ko-fi
  (async () => {
    const get = async u => { try { const r = await fetch(u, { cache: "no-store" }); return r.ok ? r.json() : null; } catch (e) { return null; } };
    const [U, F] = await Promise.all([get("/api/public/metrics"), get("/api/public/funding")]);
    const C = U && U.costs;
    if (!C || !F) return;
    const $ = id => document.getElementById(id), usd = n => "$" + (Math.round(n * 100) / 100).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const name = m => new Date(m + "-15T12:00:00Z").toLocaleString("en-US", { month: "short", year: "numeric", timeZone: "UTC" });
    const now = new Date().toLocaleString("sv-SE", { timeZone: U.timezone || "America/New_York" }).slice(0, 7);
    const api = Object.fromEntries((C.months || []).map(m => [m.month, m])), given = Object.fromEntries((F.donations.months || []).map(m => [m.month, m]));
    const fixedIn = m => (F.fixed || []).filter(f => (!f.from || f.from <= m) && (!f.until || m <= f.until)).reduce((t, f) => t + f.usdPerMonth, 0);
    const months = [...new Set([...Object.keys(api), ...Object.keys(given)])].filter(m => m <= now).sort().reverse();
    if (!months.length) return;
    const kinds = ["chat", "images", "video", "tools"], label = { chat: "Chat", images: "Images", video: "Video", tools: "Tools" };
    const tot = { spent: 0, given: 0, fixed: 0 }; kinds.forEach(k => tot[k] = 0);
    const rows = months.map(m => {
      const a = api[m] || { usd: 0, kinds: {} }, fx = fixedIn(m), g = given[m] ? given[m].usd : 0, spent = a.usd + fx, net = g - spent;
      kinds.forEach(k => tot[k] += a.kinds[k] || 0); tot.fixed += fx; tot.spent += spent; tot.given += g;
      return `<tr><td>${name(m)}${m === now ? "<small>SO FAR</small>" : ""}</td>${kinds.map(k => `<td class="${a.kinds[k] ? "" : "dim"}">${usd(a.kinds[k] || 0)}</td>`).join("")}<td class="${fx ? "" : "dim"}">${usd(fx)}</td><td class="sum">${usd(spent)}</td><td class="${g ? "pos" : "dim"}">${usd(g)}</td><td class="${net >= 0 ? "pos" : "neg"}">${net < 0 ? "−" : "+"}${usd(Math.abs(net))}</td></tr>`;
    });
    const tnet = tot.given - tot.spent;
    $("f-table").innerHTML = `<thead><tr><th>Month</th>${kinds.map(k => `<th>${label[k]}</th>`).join("")}<th>Hosting &amp; fixed</th><th>Spent</th><th>Given</th><th>Difference</th></tr></thead><tbody>${rows.join("")}`
      + `<tr class="total"><td>All months</td>${kinds.map(k => `<td>${usd(tot[k])}</td>`).join("")}<td>${usd(tot.fixed)}</td><td>${usd(tot.spent)}</td><td>${usd(tot.given)}</td><td class="${tnet >= 0 ? "pos" : "neg"}">${tnet < 0 ? "−" : "+"}${usd(Math.abs(tnet))}</td></tr></tbody>`;
    const cur = api[now] || { usd: 0 }, curSpent = cur.usd + fixedIn(now), curGiven = given[now] ? given[now].usd : 0, donors = (F.donations.months || []).reduce((t, m) => t + (m.count || 0), 0);
    $("f-given").textContent = usd(tot.given); $("f-given-c").textContent = donors ? donors + (donors === 1 ? " payment" : " payments") + " on Ko-fi" : "on Ko-fi";
    $("f-spent").textContent = usd(tot.spent); $("f-spent-c").textContent = usd(tot.spent - tot.fixed) + " API · " + usd(tot.fixed) + " fixed";
    $("f-month-l").textContent = name(now) + " so far"; $("f-month").textContent = usd(curSpent); $("f-month-c").textContent = usd(curGiven) + " given";
    $("f-net").textContent = (tnet < 0 ? "−" : "+") + usd(Math.abs(tnet)); $("f-net-c").textContent = tnet < 0 ? "paid out of pocket" : "ahead";
    $("f-net").className = tnet < 0 ? "neg" : "pos";
    if (F.kofi) $("f-kofi").href = F.kofi;
    const fixedNotes = (F.fixed || []).map(f => f.name + " " + usd(f.usdPerMonth) + "/mo" + (f.note ? " (" + f.note + ")" : "")).join(", ");
    $("f-note").textContent = "API spend is the bot's own estimate from token counts and each provider's published prices. " + (fixedNotes ? "Fixed: " + fixedNotes + ". " : "") + (F.donations.live ? "Ko-fi payments show up here as they arrive." : "");
    $("funding").hidden = false;
  })();
