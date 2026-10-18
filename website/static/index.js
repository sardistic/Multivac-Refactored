
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
