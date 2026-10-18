// The tab, as on sardistic.com: the eye as a live icon (it looks toward the cursor, darts about when left alone, blinks),
// and the title "insufficient.dev" worn away by a wave of ASCII foam and built back again.
// Crawlers and reduced motion keep the plain title and the still icon.
(() => {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches || /bot|crawl|spider|slurp|headless|lighthouse/i.test(navigator.userAgent) || navigator.webdriver) return;
  const NAME = "insufficient.dev", FOAM = "~≈∿≋", DUST = "·.·:˙", c = document.createElement("canvas"); c.width = c.height = 64;
  const g = c.getContext("2d");
  let link = document.querySelector('link[rel="icon"][type="image/svg+xml"]') || document.querySelector('link[rel="icon"]');
  if (!link) { link = document.createElement("link"); link.rel = "icon"; document.head.appendChild(link); }
  link.type = "image/png";
  document.querySelectorAll('link[rel="icon"]').forEach(l => { if (l !== link) l.remove(); });   // one icon, so the browser follows this one
  const look = [0, 0], aim = [0, 0];
  let nextDart = 0, blinkAt = performance.now() + 2500, lastIcon = "";
  addEventListener("pointermove", e => { aim[0] = (e.clientX / innerWidth - 0.5) * 2; aim[1] = (e.clientY / innerHeight - 0.5) * 2; nextDart = performance.now() + 4000; }, { passive: true });
  function drawEye(now) {
    if (now > nextDart) { aim[0] = Math.random() * 2 - 1; aim[1] = Math.random() * 1.6 - 0.8; nextDart = now + 1400 + Math.random() * 2600; }
    look[0] += (aim[0] - look[0]) * 0.35; look[1] += (aim[1] - look[1]) * 0.35;
    const bu = (now - blinkAt) / 190, lid = bu > 0 && bu < 2 ? Math.sin(Math.min(1, bu / 2) * Math.PI) : 0;   // 0 open .. 1 shut
    if (bu >= 2) blinkAt = now + 2600 + Math.random() * 4200 * (Math.random() < 0.2 ? 0.15 : 1);           // now and then a double blink
    g.clearRect(0, 0, 64, 64);
    const lens = g.createRadialGradient(26, 22, 2, 32, 32, 31); lens.addColorStop(0, "#fbf8f2"); lens.addColorStop(0.7, "#ebe5da"); lens.addColorStop(1, "#d9d2c4");
    g.fillStyle = lens; g.beginPath(); g.arc(32, 32, 30, 0, 7); g.fill();
    g.save(); g.beginPath(); g.arc(32, 32, 29, 0, 7); g.clip();
    const rx = 32 + look[0] * 5, ry = 32 + look[1] * 5, cx = 32 + look[0] * 9, cy = 32 + look[1] * 9;
    const am = g.createRadialGradient(rx - 4, ry - 4, 1, rx, ry, 19); am.addColorStop(0, "#f7cf5a"); am.addColorStop(1, "#d6921a");
    g.fillStyle = am; g.beginPath(); g.arc(rx, ry, 18.5, 0, 7); g.fill();
    const rd = g.createRadialGradient(cx - 2, cy - 3, 0.5, cx, cy, 9); rd.addColorStop(0, "#f06a52"); rd.addColorStop(1, "#ad2416");
    g.fillStyle = rd; g.beginPath(); g.arc(cx, cy, 9, 0, 7); g.fill();
    if (lid > 0.01) {                                                   // the lids close from above and below, meeting a little low
      const top = 2 + lid * 34, bot = 62 - lid * 26;
      g.fillStyle = "#d8cfbf"; g.fillRect(0, 0, 64, top); g.fillRect(0, bot, 64, 64 - bot);
      g.strokeStyle = "#16141b"; g.lineWidth = 2.2; g.beginPath(); g.moveTo(2, top); g.quadraticCurveTo(32, top + 6 * (1 - lid), 62, top); g.stroke();
    }
    const gl = g.createRadialGradient(20, 17, 0, 20, 17, 29); gl.addColorStop(0, "rgba(255,255,255,.8)"); gl.addColorStop(0.35, "rgba(255,255,255,.2)"); gl.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = gl; g.fillRect(0, 0, 64, 64);
    g.restore();
    g.strokeStyle = "rgba(22,20,27,.32)"; g.lineWidth = 1.4; g.beginPath(); g.arc(32, 32, 30.3, 0, 7); g.stroke();
    const url = c.toDataURL("image/png");
    if (url !== lastIcon) { link.href = url; lastIcon = url; }
  }
  // the title: a crest of foam moves out from the eye (the left of the tab), leaving dust where it passed; then it comes back and builds
  const T0 = performance.now() + 3200, CYCLE = 9800;
  const pick = (s, i, k) => s[(i * 7 + k * 13 + (i * k) % 5) % s.length];
  function titleAt(now) {
    const u = now - T0;
    if (u < 0) return NAME;
    const ph = u % CYCLE, cyc = Math.floor(u / CYCLE), out = [];
    // 0–2.6 s erode, 2.6–4.4 s hold in dust, 4.4–7 s rebuild, then the word for a while
    const erode = ph < 2600, hold = ph >= 2600 && ph < 4400, build = ph >= 4400 && ph < 7000;
    const w = erode ? -2 + (ph / 2600) * (NAME.length + 5) : build ? -2 + ((ph - 4400) / 2600) * (NAME.length + 5) : 99;
    for (let i = 0; i < NAME.length; i++) {
      const d = w - i, tick = Math.floor(now / 140);
      if (!erode && !hold && !build) out.push(NAME[i]);
      else if (hold) out.push(pick(DUST, i, cyc + Math.floor(now / 600)));
      else if (d >= -0.5 && d < 1.5) out.push(pick(FOAM, i, tick));                    // the crest
      else if (d >= 1.5 && d < 3) out.push(erode ? pick("~-·", i, tick) : NAME[i]);      // its wake
      else if (d >= 3) out.push(erode ? pick(DUST, i, cyc) : NAME[i]);                    // passed: dust, or the letter rebuilt
      else out.push(erode ? NAME[i] : pick(DUST, i, cyc));                                // not reached yet
    }
    return out.join("");
  }
  let lastTitle = "";
  function tick() {
    const now = performance.now();
    drawEye(now);
    const t = titleAt(now);
    if (t !== lastTitle) { document.title = t; lastTitle = t; }
  }
  tick(); setInterval(tick, 110);
})();
