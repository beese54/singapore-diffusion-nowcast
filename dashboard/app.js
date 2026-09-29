'use strict';
// Singapore Rain Nowcaster dashboard. Static: everything comes from
// data/data.json and greyscale sprite sheets written by scripts/build_dashboard.py.
//
// Raster budget (Pattern BB): every radar canvas has a backing store of exactly
// 217 x 120 px, the radar's native grid, so a draw is a 1:1 putImageData with no
// resampling; CSS scales it on the GPU. No resize handler exists on purpose.
// Players run a requestAnimationFrame loop ONLY while playing and visible, and a
// panel redraws only when the frame it shows actually changes.

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const LOGMAX = Math.log1p(100);
const LEVELS = [0.5, 1, 2, 5, 10, 20, 30, 50, 100];
const COLORS = ['#c6e8ff', '#7cc3f5', '#2f8fd8', '#3fbf4f', '#f2e03c', '#f59a23', '#e8321e', '#9b1bb5'];
const SVGNS = 'http://www.w3.org/2000/svg';
// Deep links for sharing a moment: ?hero=17:15 (static hero frame),
// ?case=27sep&lead=30&t=11:10 (explorer at that target time).
const Q = new URLSearchParams(location.search);

const $ = (id) => document.getElementById(id);
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

// ── colour lookup tables: byte -> RGBA ─────────────────────────────────────
// Thresholds compare BYTES encoded with the exporter's own rounding, so a value
// stored exactly at a level (e.g. NEA's 0.5 mm/hr) always lands in that level.
const byteOf = (mm) => Math.round(255 * Math.log1p(mm) / LOGMAX);
const RAIN_LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 4);
  const thr = LEVELS.map(byteOf);
  for (let b = 0; b < 256; b++) {
    let k = -1;
    for (let i = 0; i < 8; i++) if (b >= thr[i]) k = i;
    if (k < 0) continue;                              // below 0.5 mm/hr: transparent
    const [r, g, bl] = hex(COLORS[k]);
    lut.set([r, g, bl, 255], b * 4);
  }
  return lut;
})();
const PROB_STOPS = [[0.0, '#fde68a'], [0.25, '#fb923c'], [0.5, '#ef4444'], [0.75, '#c026d3'], [1.0, '#6d28d9']];
const probColor = (p) => {
  for (let i = 1; i < PROB_STOPS.length; i++) {
    const [p1, c1] = PROB_STOPS[i]; const [p0, c0] = PROB_STOPS[i - 1];
    if (p <= p1) {
      const t = (p - p0) / (p1 - p0), a = hex(c0), b = hex(c1);
      return a.map((v, j) => Math.round(v + (b[j] - v) * t));
    }
  }
  return hex(PROB_STOPS[PROB_STOPS.length - 1][1]);
};
const PROB_LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 4);
  for (let b = 3; b < 256; b++) {                    // p < ~1%: transparent
    const p = b / 255;
    // 1 of 8 futures (p = 0.125) stays faint; agreement is what should stand out
    lut.set([...probColor(p), Math.round(255 * Math.min(1, 0.12 + 0.88 * p))], b * 4);
  }
  return lut;
})();

// ── sprite sheets ──────────────────────────────────────────────────────────
// Resolve on onload (enough for drawImage); never await img.decode() in the
// chain -- in a background tab it may never settle (Pattern BB).
const sheetCache = new Map();
function loadSheet(spec) {
  if (sheetCache.has(spec.file)) return sheetCache.get(spec.file);
  const p = new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas');
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(img, 0, 0);
      const rgba = ctx.getImageData(0, 0, c.width, c.height).data;
      const grey = new Uint8Array(c.width * c.height);
      for (let i = 0; i < grey.length; i++) grey[i] = rgba[i * 4];
      resolve({ ...spec, sheetW: c.width, grey });
    };
    // drop a failed load from the cache so the next request retries it
    img.onerror = () => { sheetCache.delete(spec.file); reject(new Error('failed to load ' + spec.file)); };
    img.src = spec.file;
  });
  sheetCache.set(spec.file, p);
  return p;
}

// ── a radar panel: canvas + SVG overlay in radar-pixel coordinates ─────────
class Panel {
  constructor(el, grid, coast) {
    this.w = grid.w; this.h = grid.h;
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.w; this.canvas.height = this.h;   // native size, see header
    this.ctx = this.canvas.getContext('2d');
    this.img = this.ctx.createImageData(this.w, this.h);
    el.appendChild(this.canvas);
    this.svg = document.createElementNS(SVGNS, 'svg');
    this.svg.setAttribute('viewBox', `0 0 ${this.w} ${this.h}`);
    this.svg.setAttribute('preserveAspectRatio', 'none');
    this.svg.setAttribute('aria-hidden', 'true');
    const g = document.createElementNS(SVGNS, 'g');
    g.setAttribute('transform', 'translate(0.5 0.5)');         // pixel centres
    for (const d of coast || []) {
      const p = document.createElementNS(SVGNS, 'path');
      p.setAttribute('d', d); p.setAttribute('fill', 'none');
      p.setAttribute('stroke', 'var(--coast)'); p.setAttribute('stroke-width', '0.6');
      p.setAttribute('vector-effect', 'non-scaling-stroke');
      g.appendChild(p);
    }
    this.marks = document.createElementNS(SVGNS, 'g');
    g.appendChild(this.marks);
    this.svg.appendChild(g);
    el.appendChild(this.svg);
    this.key = null;
  }
  draw(sheet, k, lut) {
    const key = sheet.file + '#' + k + (lut === PROB_LUT ? 'p' : 'r');
    if (key === this.key) return;                              // dirty guard
    this.key = key;
    const { w, h, cols, sheetW, grey } = sheet;
    const r0 = Math.floor(k / cols) * h, c0 = (k % cols) * w;
    const out = this.img.data;
    for (let y = 0; y < h; y++) {
      const src = (r0 + y) * sheetW + c0;
      for (let x = 0; x < w; x++) {
        const b = grey[src + x] * 4, o = (y * w + x) * 4;
        out[o] = lut[b]; out[o + 1] = lut[b + 1]; out[o + 2] = lut[b + 2]; out[o + 3] = lut[b + 3];
      }
    }
    this.ctx.putImageData(this.img, 0, 0);
  }
  setMarks(items) {
    // flood markers get a white halo so they read on top of red/purple rain
    items = items.flatMap((m) => (m.halo ? [{ ...m, stroke: '#ffffff', sw: (m.sw || 1) + 2.5, title: null }, m] : [m]));
    this.marks.replaceChildren(...items.map((m) => {
      const e = document.createElementNS(SVGNS, m.shape === 'rect' ? 'rect' : 'circle');
      if (m.shape === 'rect') {
        e.setAttribute('x', m.x); e.setAttribute('y', m.y); e.setAttribute('width', m.w); e.setAttribute('height', m.h);
      } else {
        e.setAttribute('cx', m.x); e.setAttribute('cy', m.y); e.setAttribute('r', m.r || 1.6);
      }
      e.setAttribute('fill', m.fill || 'none');
      e.setAttribute('stroke', m.stroke || '#fff');
      e.setAttribute('stroke-width', m.sw || 1);
      e.setAttribute('vector-effect', 'non-scaling-stroke');
      if (m.title) { const t = document.createElementNS(SVGNS, 'title'); t.textContent = m.title; e.appendChild(t); }
      return e;
    }));
  }
}

// ── player: a rAF loop that exists only while playing ──────────────────────
function makePlayer({ fps, button, step, loop = true, visibleEl }) {
  let raf = 0, last = 0, acc = 0, playing = false, onScreen = true;
  const frame = (ts) => {
    if (!playing) return;
    if (last) acc += ts - last;
    last = ts;
    const dt = 1000 / fps;
    while (acc >= dt) {
      acc -= dt;
      if (!step() && !loop) { stop(); return; }
    }
    raf = requestAnimationFrame(frame);
  };
  const start = () => {
    if (playing) return;
    playing = true; last = 0; acc = 0;
    button.textContent = '❚❚'; button.setAttribute('aria-label', 'Pause');
    raf = requestAnimationFrame(frame);
  };
  const stop = () => {
    playing = false; cancelAnimationFrame(raf);
    button.textContent = '▶'; button.setAttribute('aria-label', 'Play');
  };
  button.addEventListener('click', () => (playing ? stop() : start()));
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  if (visibleEl && 'IntersectionObserver' in window) {
    new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; if (!onScreen) stop(); }).observe(visibleEl);
  }
  return { start, stop, get playing() { return playing; } };
}

// ── helpers ────────────────────────────────────────────────────────────────
const fmt = (v, d = 3) => (v >= 0 ? '+' : '') + v.toFixed(d);
const pct = (v) => Math.round(v * 100) + '%';
function segmented(el, options, current, onPick) {
  el.replaceChildren(...options.map(([value, label]) => {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = label; b.setAttribute('role', 'radio');
    b.setAttribute('aria-checked', String(value === current));
    b.addEventListener('click', () => {
      for (const x of el.children) x.setAttribute('aria-checked', 'false');
      b.setAttribute('aria-checked', 'true'); onPick(value);
    });
    return b;
  }));
}
function svgEl(tag, attrs, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}
const toMin = (hhmm) => { const [h, m] = hhmm.split(':').map(Number); return h * 60 + m; };

// ── sections ───────────────────────────────────────────────────────────────
async function hero(D) {
  const c = D.cases['22sep'];
  const panel = new Panel($('hero-panel'), D.grid, D.coast);
  const sheet = await loadSheet(c.obs);
  const flash = c.reports.filter((r) => r.type === 'FLASH_FLOOD');
  let k = c.obs_times.indexOf('16:50');
  const show = () => {
    panel.draw(sheet, k, RAIN_LUT);
    const now = toMin(c.obs_times[k]);
    panel.setMarks(flash.filter((r) => toMin(r.time) <= now)
      .map((r) => ({ x: r.x, y: r.y, r: 2.6, stroke: '#ff2d20', sw: 2, halo: true, title: `${r.name} ${r.time}` })));
    $('hero-time').textContent = c.obs_times[k];
  };
  show();
  const btn = $('hero-play');
  if (REDUCED) { btn.hidden = true; return; }     // static frame, no loop
  const player = makePlayer({ fps: 6, button: btn, visibleEl: $('hero-panel'),
    step: () => { k = (k + 1) % c.obs.n; show(); return true; } });
  const shared = c.obs_times.indexOf(Q.get('hero'));
  if (shared >= 0) { k = shared; show(); player.stop(); return; }   // shared moment: hold it
  k = 0; show(); player.start();
}

async function denoise(D) {
  const d = D.denoise;
  $('dn-issued').textContent = `(forecast issued ${d.issued} SGT)`;
  $('dn-target').textContent = `(${d.target} SGT)`;
  const ctxSheet = await loadSheet(d.context);
  const strip = $('dn-context');
  for (let i = 0; i < d.context.n; i++) {
    const fig = document.createElement('figure');
    const div = document.createElement('div'); div.className = 'panel';
    fig.appendChild(div);
    const cap = document.createElement('figcaption');
    const ago = (d.context.n - 1 - i) * 5;
    cap.textContent = ago ? `−${ago} min` : 'latest'; fig.appendChild(cap);
    strip.appendChild(fig);
    new Panel(div, D.grid, D.coast).draw(ctxSheet, i, RAIN_LUT);
  }
  const [xt, x0, ob] = await Promise.all([loadSheet(d.x_t), loadSheet(d.x0), loadSheet(d.observed)]);
  const pX = new Panel($('dn-xt'), D.grid, D.coast), p0 = new Panel($('dn-x0'), D.grid, D.coast);
  new Panel($('dn-obs'), D.grid, D.coast).draw(ob, 0, RAIN_LUT);
  const slider = $('dn-step'), out = $('dn-step-out');
  const show = (s) => { pX.draw(xt, s, RAIN_LUT); p0.draw(x0, s, RAIN_LUT); out.textContent = s + 1; slider.value = s; };
  slider.addEventListener('input', () => { player.stop(); show(+slider.value); });
  const player = makePlayer({ fps: 8, button: $('dn-play'), loop: false, visibleEl: $('dn-xt'),
    step: () => { const s = +slider.value; if (s >= 49) return false; show(s + 1); return true; } });
  // capture phase: runs before the player's own toggle, so pressing play at
  // the last step rewinds and plays again instead of stopping immediately
  $('dn-play').addEventListener('click', () => { if (!player.playing && +slider.value >= 49) show(0); }, true);
  show(REDUCED ? 49 : 0);
}

async function explorer(D) {
  const state = { c: D.cases[Q.get('case')] ? Q.get('case') : '27sep',
                  lead: ['30', '60', '90'].includes(Q.get('lead')) ? Q.get('lead') : '30', k: 0 };
  const P = {
    obs: new Panel($('ex-obs'), D.grid, D.coast), per: new Panel($('ex-per'), D.grid, D.coast),
    mean: new Panel($('ex-mean'), D.grid, D.coast), p10: new Panel($('ex-p10'), D.grid, D.coast),
  };
  const slider = $('ex-t');
  segmented($('case-seg'), [['27sep', '27 Sep · Pasir Panjang'], ['22sep', '22 Sep · King\'s Road']], state.c,
    (v) => { state.c = v; state.k = firstFlagOr0(); render(true); });
  segmented($('lead-seg'), [['30', '30 min'], ['60', '60 min'], ['90', '90 min']], state.lead,
    (v) => { state.lead = v; render(true); });
  $('show-prone').addEventListener('change', () => render(false));

  // start where the story is: the first forecast that flagged the site
  function firstFlagOr0() {
    const L = D.cases[state.c].leads[state.lead];
    const i = L.site_p10.findIndex((p) => p >= 0.25);
    return i < 0 ? 0 : i;
  }
  state.k = firstFlagOr0();
  if (Q.has('t')) {
    const i = D.cases[state.c].leads[state.lead].targets.indexOf(Q.get('t'));
    if (i >= 0) state.k = i;
  }

  async function render(resetSlider) {
    const c = D.cases[state.c], L = c.leads[state.lead];
    const n = L.targets.length;
    if (resetSlider) { slider.max = n - 1; state.k = Math.min(state.k, n - 1); }
    slider.value = state.k;
    const [obs, mean, p10] = await Promise.all([loadSheet(c.obs), loadSheet(L.mean), loadSheet(L.p10)]);
    const k = state.k;
    P.obs.draw(obs, L.target_obs_index[k], RAIN_LUT);
    P.per.draw(obs, L.persistence_obs_index[k], RAIN_LUT);
    P.mean.draw(mean, k, RAIN_LUT);
    P.p10.draw(p10, k, PROB_LUT);
    const s = c.site;
    const marks = [{ shape: 'rect', x: s.j - s.r - 0.5, y: s.i - s.r - 0.5, w: 2 * s.r + 1, h: 2 * s.r + 1, stroke: '#ffffff', sw: 1.5, title: s.name }];
    if ($('show-prone').checked) {
      for (const f of D.flood_prone) marks.push({ x: f.x, y: f.y, r: 1.3, stroke: '#9fe7ff', sw: 1, title: `PUB flood-prone: ${f.name}` });
    }
    for (const r of c.reports.filter((r) => r.type === 'FLASH_FLOOD')) {
      marks.push({ x: r.x, y: r.y, r: 2.6, stroke: '#ff2d20', sw: 2, halo: true, title: `Flash flood ${r.time}: ${r.name}` });
    }
    Object.values(P).forEach((p) => p.setMarks(marks));
    const target = L.targets[k], issued = L.issued[k];
    $('ex-t-out').textContent = `${target} SGT (made at ${issued})`;
    const obsNow = c.obs_site[L.target_obs_index[k]];
    const naiveNow = c.obs_site[L.persistence_obs_index[k]];
    const flood = c.reports.find((r) => r.type === 'FLASH_FLOOD');
    $('ex-sentence').innerHTML =
      `At <b>${issued}</b>, forecasting <b>${target}</b> SGT (${L.minutes_after_last_frame} minutes ahead), ` +
      `for the flood site <b>${s.name}</b>` + (flood ? `, where a flash flood was reported at <b>${flood.time}</b>` : '') + ':';
    // The verdict, in plain words: heavy rain = at least 10 mm/hr somewhere in
    // the 2 x 2 km box; the model "warns" when 2 or more of 8 futures show it.
    const HEAVY = 10, WARN = 0.25;
    const real = obsNow >= HEAVY, modelWarn = L.site_p10[k] >= WARN, naiveWarn = naiveNow >= HEAVY;
    const chip = (label, said, right) =>
      `<span class="chip ${right ? 'ok' : 'no'}"><span class="mark">${right ? '✓' : '✗'}</span><b>${label}</b> ${said}</span>`;
    $('verdict').innerHTML =
      chip('AI model:', modelWarn ? `warns of heavy rain (${Math.round(L.site_p10[k] * 8)} of 8 futures)`
                                  : `no warning (${Math.round(L.site_p10[k] * 8)} of 8 futures)`, modelWarn === real) +
      chip('Naive forecast:', naiveWarn ? `heavy rain (it is raining ${naiveNow} mm/hr now)`
                                        : `no heavy rain (${naiveNow} mm/hr now)`, naiveWarn === real) +
      `<span class="chip truth"><b>What happened:</b> ${real ? `heavy rain, ${obsNow} mm/hr` : `no heavy rain (${obsNow} mm/hr)`}</span>`;
    moments(c, L);
    chart(c, L, k);
  }

  // one-click jumps to the moments that tell the story
  function moments(c, L) {
    const flood = c.reports.find((r) => r.type === 'FLASH_FLOOD');
    const obsAt = L.target_obs_index.map((i) => c.obs_site[i]);
    const first = L.site_p10.findIndex((p) => p >= 0.25);
    const peak = obsAt.indexOf(Math.max(...obsAt));
    const atFlood = flood ? L.targets.findIndex((t) => toMin(t) >= toMin(flood.time)) : -1;
    const items = [
      [first, first >= 0 ? `First warning (made ${L.issued[first]})` : 'No warning at this lead'],
      [peak, `Heaviest rain (${L.targets[peak]})`],
      [atFlood, flood ? `Flood reported (${flood.time})` : ''],
    ].filter(([i, label]) => label);
    $('moments').replaceChildren(...items.map(([i, label]) => {
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = label; b.disabled = i < 0;
      b.setAttribute('aria-pressed', String(i === state.k));
      b.addEventListener('click', () => { player.stop(); state.k = i; render(false); });
      return b;
    }));
  }
  slider.addEventListener('input', () => { player.stop(); state.k = +slider.value; render(false); });
  const player = makePlayer({ fps: 2, button: $('ex-play'), visibleEl: $('ex-obs'),
    step: () => { const n = +slider.max + 1; state.k = (state.k + 1) % n; render(false); return true; } });
  await render(true);

  // legend
  const rainBar = COLORS.map((c) => `<span style="background:${c}"></span>`).join('');
  const probBar = [0.1, 0.3, 0.5, 0.7, 0.9].map((p) => `<span style="background:rgb(${probColor(p)})"></span>`).join('');
  $('legend').innerHTML =
    `<span>Rain (mm/hr): 0.5<span class="bar">${rainBar}</span>100</span>` +
    `<span>Chance of ≥10 mm/hr: 0%<span class="bar">${probBar}</span>100%</span>` +
    `<span>□ flood site (2 × 2 km) · <span style="color:#ff3b30">○</span> flash flood reported · <span style="color:#3aa9c8">○</span> PUB flood-prone location</span>`;
}

function chart(c, L, k) {
  const W = 720, H = 240, m = { l: 40, r: 44, t: 26, b: 26 };
  const t0 = toMin(c.obs_times[0]), t1 = toMin(c.obs_times[c.obs_times.length - 1]);
  const X = (t) => m.l + (toMin(t) - t0) / (t1 - t0) * (W - m.l - m.r);
  const ymax = Math.max(40, ...c.obs_site, ...L.site_max) * 1.08;
  const Y = (v) => H - m.b - v / ymax * (H - m.t - m.b);
  const YP = (p) => H - m.b - p * (H - m.t - m.b);
  const s = svgEl('svg', { viewBox: `0 0 ${W} ${H}` });
  // gridlines + axes
  for (let v = 0; v <= ymax; v += 20) {
    s.appendChild(svgEl('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: 'var(--line)' }));
    s.appendChild(svgEl('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, v));
  }
  for (const p of [0, 0.5, 1]) s.appendChild(svgEl('text', { x: W - m.r + 6, y: YP(p) + 4 }, pct(p)));
  for (let t = Math.ceil(t0 / 60) * 60; t <= t1; t += 60) {
    const hh = String(Math.floor(t / 60)).padStart(2, '0') + ':00';
    s.appendChild(svgEl('text', { x: X(hh), y: H - 8, 'text-anchor': 'middle' }, hh));
  }
  s.appendChild(svgEl('text', { x: m.l - 6, y: 12, 'text-anchor': 'end' }, 'mm/hr'));
  s.appendChild(svgEl('text', { x: W - m.r + 6, y: 12 }, 'P(≥10)'));
  // probability bars
  L.targets.forEach((t, i) => {
    const h = YP(0) - YP(L.site_p10[i]);
    s.appendChild(svgEl('rect', { x: X(t) - 3, y: YP(L.site_p10[i]), width: 6, height: Math.max(h, 0),
      fill: `rgb(${probColor(L.site_p10[i])})`, opacity: i === k ? 1 : 0.55 }));
  });
  // ensemble band + median
  const band = L.targets.map((t, i) => `${X(t)},${Y(L.site_max[i])}`).join(' ') + ' ' +
    L.targets.map((t, i) => `${X(t)},${Y(L.site_min[i])}`).reverse().join(' ');
  s.appendChild(svgEl('polygon', { points: band, fill: 'var(--accent)', opacity: 0.15 }));
  s.appendChild(svgEl('polyline', { points: L.targets.map((t, i) => `${X(t)},${Y(L.site_median[i])}`).join(' '),
    fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2 }));
  // observed (step)
  let d = '';
  c.obs_times.forEach((t, i) => { d += (i ? ` L${X(t)},${Y(c.obs_site[i - 1])} L` : 'M') + `${X(t)},${Y(c.obs_site[i])}`; });
  s.appendChild(svgEl('path', { d, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.6 }));
  // flash-flood reports and the cursor
  for (const r of c.reports.filter((r) => r.type === 'FLASH_FLOOD')) {
    s.appendChild(svgEl('line', { x1: X(r.time), x2: X(r.time), y1: m.t, y2: H - m.b, stroke: '#ff3b30', 'stroke-dasharray': '3 3' }));
  }
  s.appendChild(svgEl('line', { x1: X(L.targets[k]), x2: X(L.targets[k]), y1: m.t, y2: H - m.b, stroke: 'var(--accent)', 'stroke-width': 1.5 }));
  s.appendChild(svgEl('line', { x1: X(L.issued[k]), x2: X(L.issued[k]), y1: m.t, y2: H - m.b, stroke: 'var(--muted)', 'stroke-dasharray': '2 4' }));
  $('ex-chart').replaceChildren(s);
  $('ex-chart-cap').innerHTML = `<b>How to read this chart:</b> the bold line is the rain that actually fell at ${c.site.name}. ` +
    `The coloured bars are the model's warning for each time: how many of its 8 futures showed heavy rain there (taller = more sure). ` +
    `The blue line and band are the rain amounts it expected. The solid blue marker is the forecast you selected; the dotted grey one is ` +
    `when that forecast was made; the red dashed line is the flash-flood report. <b>Look for bars that rise before the bold line does:</b> ` +
    `that is a warning given in advance.`;
}

function evidence(D) {
  const R = D.results;
  $('ev-period').textContent = `${R.period[0].slice(0, 10)} to ${R.period[1].slice(0, 10)}`;
  const h30 = R.headline['30'];
  $('stat-crps').textContent = Math.round(h30.crps_skill * 100) + '%';
  $('stat-crps-ci').textContent = `likely range ${Math.round(h30.crps_ci[0] * 100)}–${Math.round(h30.crps_ci[1] * 100)}% (CRPS skill)`;
  // colour by the numbers as DISPLAYED (2 dp): 0.558 vs 0.558 is a tie, not a win
  const cls = (a, b) => { const x = +(+a).toFixed(2), y = +(+b).toFixed(2); return x > y ? 'good' : x < y ? 'bad' : ''; };
  const h = R.headline;
  $('ev-summary').innerHTML =
    `<li><b>30 minutes ahead, the model beats the naive forecast overall</b>: its probability forecasts have ` +
    `${Math.round(h['30'].crps_skill * 100)}% lower error (likely range ${Math.round(h['30'].crps_ci[0] * 100)}–${Math.round(h['30'].crps_ci[1] * 100)}%), ` +
    `and it puts light rain in the right place more often (score ${h['30'].fss2[0].toFixed(2)} vs ${h['30'].fss2[1].toFixed(2)}).</li>` +
    `<li><b>Heavy rain is not solved:</b> for downpours of 10 mm/hr and more, the naive forecast still places rain better at every lead ` +
    `(${h['30'].fss10[0].toFixed(2)} vs ${h['30'].fss10[1].toFixed(2)} at 30 minutes). This is the main problem left.</li>` +
    `<li><b>60 and 90 minutes ahead</b>, the model only matches or slightly beats the naive forecast, and only for light rain.</li>`;
  let html = '<thead><tr><th>How far ahead</th><th>Overall accuracy<br><small>% lower error than naive (CRPS skill, 95% range)</small></th>' +
    '<th>Light rain in the right place<br><small>score, model / naive (FSS ≥2 mm/hr)</small></th>' +
    '<th>Heavy rain in the right place<br><small>score, model / naive (FSS ≥10 mm/hr)</small></th></tr></thead><tbody>';
  for (const L of ['30', '60', '90']) {
    const r = R.headline[L];
    html += `<tr><td>${L} min</td><td class="${r.crps_ci[0] > 0 ? 'good' : ''}">${Math.round(r.crps_skill * 100)}% <small>(${Math.round(r.crps_ci[0] * 100)}–${Math.round(r.crps_ci[1] * 100)}%)</small></td>` +
      `<td class="${cls(r.fss2[0], r.fss2[1])}">${r.fss2[0].toFixed(2)} / ${r.fss2[1].toFixed(2)}</td>` +
      `<td class="${cls(r.fss10[0], r.fss10[1])}">${r.fss10[0].toFixed(2)} / ${r.fss10[1].toFixed(2)}</td></tr>`;
  }
  $('ev-table').innerHTML = html + '</tbody>';
  $('ev-fss').replaceChildren(...['30', '60', '90'].map((L) => {
    const card = document.createElement('div'); card.className = 'fss-card';
    let t = `<h4>${L} min lead</h4><table><thead><tr><th>Threshold</th><th>6 km</th><th>12 km</th><th>24 km</th></tr></thead><tbody>`;
    for (const thr of ['0.5', '2.0', '10.0']) {
      const f = R.fss_by_scale[L][thr];
      t += `<tr><td>${parseFloat(thr)} mm/hr</td>` + ['6.1 km', '11.9 km', '23.5 km'].map((k) =>
        `<td class="${cls(f.model[k], f.persistence[k])}">${f.model[k].toFixed(2)}<small> / ${f.persistence[k].toFixed(2)}</small></td>`).join('') + '</tr>';
    }
    card.innerHTML = t + '</tbody></table>';
    return card;
  }));
  let ft = '<thead><tr><th>How far ahead</th><th>Floods during the test period<br><small>18 &amp; 22 Sep: model / naive</small></th>' +
    '<th>Floods after it<br><small>27 Sep: model / naive</small></th><th>How often it raises an alarm on ordinary days<br><small>model / naive</small></th></tr></thead><tbody>';
  for (const L of ['30', '60', '90']) {
    const f = R.flood_events[L], a = f.test, b = f.after_test;
    ft += `<tr><td>${L} min</td><td class="${cls(a.model_hits, a.persistence_hits)}">${a.model_hits}/${a.n} vs ${a.persistence_hits}/${a.n}</td>` +
      `<td class="${cls(b.model_hits, b.persistence_hits)}">${b.model_hits}/${b.n} vs ${b.persistence_hits}/${b.n}</td>` +
      `<td>${(b.model_alarm_rate_ordinary * 100).toFixed(1)}% / ${(b.persistence_alarm_rate_ordinary * 100).toFixed(1)}%</td></tr>`;
  }
  $('ev-flood').innerHTML = ft + '</tbody>';

  // "When it warns, is it right?" -- results/warning_skill.json
  const W = R.warning, pc = (v) => Math.round(v * 100) + '%';
  const rng = (c) => `<small>(${pc(c[0])}–${pc(c[1])})</small>`;
  let wt = '<thead><tr><th>How far ahead</th><th>Catches this share of downpours<br><small>AI model (95% range) · naive</small></th>' +
    '<th>Right when it warns<br><small>AI model (95% range) · naive</small></th><th>Compared with guessing</th></tr></thead><tbody>';
  for (const L of ['30', '60', '90']) {
    const w = W[L];
    wt += `<tr><td>${L} min</td><td class="${cls(w.model.catch_rate, w.naive.catch_rate)}">${pc(w.model.catch_rate)} ${rng(w.model.catch_rate_ci)} · ${pc(w.naive.catch_rate)}</td>` +
      `<td class="${cls(w.model.precision, w.naive.precision)}">${pc(w.model.precision)} ${rng(w.model.precision_ci)} · ${pc(w.naive.precision)}</td>` +
      `<td>${Math.round(w.model.times_chance)}× more often right than chance</td></tr>`;
  }
  $('ev-warn').innerHTML = wt + '</tbody>';
  const w30 = W['30'], only = w30.model_only_warnings;
  $('ev-warn-note').innerHTML =
    `Heavy rain falls in only about ${(w30.base_rate * 100).toFixed(1)}% of squares at any moment, so a warning that is right ` +
    `1 time in 3 is far better than chance. <b>The model's distinctive value</b> is warning where the radar shows no heavy rain ` +
    `yet: at 30 minutes it gave ${only.warnings.toLocaleString()} such warnings, ${only.came_true} came true — about ` +
    `${pc(only.share_of_all_heavy_rain)} of all downpours flagged before they arrived, which the naive forecast can never do — ` +
    `but ${pc(1 - only.precision)} of those early warnings were false alarms. At 60 and 90 minutes it is no better than the naive forecast.`;
  const W60 = R.warning60;
  if (W60) $('ev-warn-note').innerHTML +=
    `<br><br><b>60 minutes ahead, use plain extrapolation for heavy-rain warnings.</b> Simply moving the latest radar map along ` +
    `its recent motion catches ${pc(W60.extrap.catch_rate)} of downpours and is right ${pc(W60.extrap.precision)} of the time. ` +
    `Re-tuning the AI model's warning rule (warn when ${W60.rule.k} of 8 futures show ${W60.rule.X} mm/hr) lifts its catch from ` +
    `${pc(W60.uncal.catch_rate)} to ${pc(W60.cal.catch_rate)}, right ${pc(W60.cal.precision)} of the time — better, but still not ` +
    `better than extrapolation, and the gain was not stable across random draws. Combining the two catches ${pc(W60.hybrid.catch_rate)} ` +
    `but is right only ${pc(W60.hybrid.precision)} of the time.`;
}

function live() {
  // Measured 2026-09-26 (docs/RESULTS.md §5): 19.6 s total, 1.7 s loading.
  const parts = [['Load', 1.7, '#5b6875'], ['30-min · 8 futures', 6.1, '#0a6e8a'],
    ['60-min', 5.4, '#1289a8'], ['90-min', 5.4, '#27a4c4'], ['Save', 19.6 - 1.7 - 6.1 - 5.4 - 5.4, '#8a97a3']];
  $('live-timeline').replaceChildren(...parts.map(([label, s, col]) => {
    const d = document.createElement('div');
    d.style.flex = String(s); d.style.background = col;
    d.textContent = `${label} ${s.toFixed(1)} s`;
    d.title = `${label === 'Load' ? 'Load models and radar' : label}: ${s.toFixed(1)} s`;
    return d;
  }));
}

function radar240(D) {
  const frames = D.radar240 || [];
  if (!frames.length) { $('r240-fig').hidden = true; $('r240-fig').parentElement.classList.add('one'); return; }
  const holder = $('r240');
  const img = document.createElement('img');
  img.alt = 'NEA 240 km rain radar, 22 Sep 2026';
  holder.appendChild(img);
  let k = 0;
  const show = () => { img.src = frames[k].file; $('r240-time').textContent = frames[k].time + ' SGT ·'; };
  show();
  if (REDUCED) { $('r240-play').hidden = true; return; }
  makePlayer({ fps: 3, button: $('r240-play'), visibleEl: holder,
    step: () => { k = (k + 1) % frames.length; show(); return true; } });
}

(async function main() {
  let D;
  try {
    D = await (await fetch('data/data.json')).json();
  } catch (e) {
    document.querySelector('main').insertAdjacentHTML('afterbegin',
      '<p class="wrap context">Could not load data/data.json. Serve this folder over HTTP ' +
      '(e.g. <code>python -m http.server</code>), not as a file:// page.</p>');
    return;
  }
  live();
  evidence(D);
  await Promise.all([hero(D), denoise(D), explorer(D)]);
  radar240(D);
})();
