/* ==========================================================================
   NextBest console
   No framework and no chart library: every visual is hand-built SVG or CSS
   grid, so colours come from the same CSS custom properties as the chrome and
   a theme switch repaints the charts for free.
   ====================================================================== */

const S = { view: 'command', filters: null, overview: null, cust: { offset: 0, limit: 40 } };

const $ = (sel, root = document) => root.querySelector(sel);
const el = (t, cls, html) => { const n = document.createElement(t); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; };

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path} -> ${r.status} ${await r.text()}`);
  return r.json();
}
const post = (path, body) => api(path, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});

/* ------------------------------------------------------------- formatting */
const nf = new Intl.NumberFormat('en-US');
const int = n => nf.format(Math.round(n));
const money = n => (n < 0 ? '-' : '') + '$' + nf.format(Math.round(Math.abs(n)));
const money2 = n => (n < 0 ? '-' : '') + '$' + Math.abs(n).toFixed(2);
const pp = n => (n >= 0 ? '+' : '') + n.toFixed(2) + 'pp';
const pct = n => n.toFixed(1) + '%';
const cls = n => (n > 0 ? 'pos' : n < 0 ? 'neg' : '');

/* ----------------------------------------------------------------- charts */
const SVGNS = 'http://www.w3.org/2000/svg';
function sv(tag, attrs = {}) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

/** Multi-series line chart. series: [{points:[{x,y}], color, dash, label}] */
function lineChart(series, opts = {}) {
  const W = opts.width || 660, H = opts.height || 240;
  const m = { t: 12, r: 14, b: 26, l: 52 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;

  const all = series.flatMap(s => s.points);
  if (!all.length) return el('div', 'empty', 'No data');
  const xs = all.map(p => p.x), ys = all.map(p => p.y);
  const x0 = opts.x0 ?? Math.min(...xs), x1 = opts.x1 ?? Math.max(...xs);
  let y0 = opts.y0 ?? Math.min(0, Math.min(...ys)), y1 = opts.y1 ?? Math.max(...ys);
  if (y1 === y0) y1 = y0 + 1;
  const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;

  const X = v => m.l + ((v - x0) / (x1 - x0 || 1)) * iw;
  const Y = v => m.t + ih - ((v - y0) / (y1 - y0 || 1)) * ih;

  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: H, role: 'img' });

  // horizontal gridlines + y labels
  for (let i = 0; i <= 4; i++) {
    const v = y0 + ((y1 - y0) * i) / 4, y = Y(v);
    svg.appendChild(sv('line', { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: 'var(--line)', 'stroke-width': 1 }));
    const t = sv('text', { x: m.l - 8, y: y + 3.2, 'text-anchor': 'end', class: 'axis-text' });
    t.textContent = opts.yfmt ? opts.yfmt(v) : Math.round(v);
    svg.appendChild(t);
  }
  // x labels
  for (let i = 0; i <= 4; i++) {
    const v = x0 + ((x1 - x0) * i) / 4;
    const t = sv('text', { x: X(v), y: H - 8, 'text-anchor': 'middle', class: 'axis-text' });
    t.textContent = opts.xfmt ? opts.xfmt(v) : v.toFixed(1);
    svg.appendChild(t);
  }
  // zero line if the range straddles it
  if (y0 < 0 && y1 > 0) {
    svg.appendChild(sv('line', { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), stroke: 'var(--line-2)', 'stroke-width': 1 }));
  }

  for (const s of series) {
    if (!s.points.length) continue;
    const d = s.points.map((p, i) => `${i ? 'L' : 'M'}${X(p.x).toFixed(1)} ${Y(p.y).toFixed(1)}`).join(' ');
    if (s.fill) {
      const base = Y(Math.max(y0, 0));
      const area = `${d} L${X(s.points.at(-1).x).toFixed(1)} ${base} L${X(s.points[0].x).toFixed(1)} ${base} Z`;
      svg.appendChild(sv('path', { d: area, fill: s.color, opacity: 0.11 }));
    }
    svg.appendChild(sv('path', {
      d, fill: 'none', stroke: s.color, 'stroke-width': s.width || 2,
      'stroke-linecap': 'round', 'stroke-linejoin': 'round',
      ...(s.dash ? { 'stroke-dasharray': s.dash } : {}),
    }));
    if (s.marker) {
      svg.appendChild(sv('circle', { cx: X(s.marker.x), cy: Y(s.marker.y), r: 4, fill: s.color }));
      svg.appendChild(sv('circle', { cx: X(s.marker.x), cy: Y(s.marker.y), r: 8.5, fill: 'none', stroke: s.color, opacity: .4 }));
    }
  }
  return svg;
}

/** Diverging category bars (decile uplift). */
function divergingBars(items, opts = {}) {
  const W = opts.width || 660, H = opts.height || 210;
  const m = { t: 12, r: 14, b: 30, l: 46 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const vals = items.map(d => d.value);
  const mx = Math.max(0.001, Math.max(...vals.map(Math.abs))) * 1.15;
  const Y = v => m.t + ih / 2 - (v / mx) * (ih / 2);
  const bw = (iw / items.length) * 0.62;

  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: H, role: 'img' });
  for (const s of [-1, -0.5, 0.5, 1]) {
    const y = Y(mx * s * 0.9);
    svg.appendChild(sv('line', { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: 'var(--line)', 'stroke-width': 1 }));
  }
  svg.appendChild(sv('line', { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), stroke: 'var(--line-2)', 'stroke-width': 1.2 }));

  items.forEach((d, i) => {
    const cx = m.l + (iw / items.length) * (i + 0.5);
    const y = d.value >= 0 ? Y(d.value) : Y(0);
    const h = Math.max(1.5, Math.abs(Y(d.value) - Y(0)));
    svg.appendChild(sv('rect', {
      x: cx - bw / 2, y, width: bw, height: h, rx: 2,
      fill: d.value >= 0 ? 'var(--jade)' : 'var(--amber)', opacity: .9,
    }));
    const lab = sv('text', { x: cx, y: H - 10, 'text-anchor': 'middle', class: 'axis-text' });
    lab.textContent = d.label;
    svg.appendChild(lab);
    const val = sv('text', {
      x: cx, y: d.value >= 0 ? y - 5 : y + h + 10, 'text-anchor': 'middle', class: 'axis-text',
      fill: d.value >= 0 ? 'var(--jade)' : 'var(--amber)',
    });
    val.textContent = d.value.toFixed(1);
    svg.appendChild(val);
  });
  // y axis ticks
  [mx, 0, -mx].forEach(v => {
    const t = sv('text', { x: m.l - 8, y: Y(v) + 3.2, 'text-anchor': 'end', class: 'axis-text' });
    t.textContent = v.toFixed(1);
    svg.appendChild(t);
  });
  return svg;
}

/** Simple histogram. */
function histogram(items, opts = {}) {
  const W = opts.width || 660, H = opts.height || 170;
  const m = { t: 10, r: 12, b: 26, l: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const mx = Math.max(...items.map(d => d.n)) || 1;
  const xs = items.map(d => d.x);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const X = v => m.l + ((v - x0) / (x1 - x0 || 1)) * iw;
  const bw = Math.max(2, (iw / items.length) * 0.86);

  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: H, role: 'img' });
  svg.appendChild(sv('line', { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, stroke: 'var(--line)' }));
  items.forEach(d => {
    const h = (d.n / mx) * ih;
    svg.appendChild(sv('rect', {
      x: X(d.x) - bw / 2, y: m.t + ih - h, width: bw, height: h, rx: 1.5,
      fill: d.x >= 0 ? 'var(--jade)' : 'var(--amber)', opacity: .82,
    }));
  });
  if (x0 < 0 && x1 > 0) {
    svg.appendChild(sv('line', { x1: X(0), x2: X(0), y1: m.t, y2: m.t + ih, stroke: 'var(--line-2)', 'stroke-dasharray': '3 3' }));
  }
  for (let i = 0; i <= 4; i++) {
    const v = x0 + ((x1 - x0) * i) / 4;
    const t = sv('text', { x: X(v), y: H - 8, 'text-anchor': 'middle', class: 'axis-text' });
    t.textContent = v.toFixed(1);
    svg.appendChild(t);
  }
  return svg;
}

/** Horizontal contribution bars, signed. */
function hBars(items, opts = {}) {
  const wrap = el('div');
  wrap.style.cssText = 'display:flex;flex-direction:column;gap:7px';
  const mx = Math.max(...items.map(d => Math.abs(d.value)), 0.0001);
  for (const d of items) {
    const row = el('div');
    row.style.cssText = 'display:grid;grid-template-columns:1fr 96px;gap:10px;align-items:center';
    const left = el('div');
    left.style.cssText = 'min-width:0';
    const lab = el('div', null, d.label);
    lab.style.cssText = 'font-size:12px;color:var(--ink-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:3px';
    const track = el('div');
    track.style.cssText = 'height:6px;background:var(--panel-3);border-radius:3px;position:relative;overflow:hidden';
    const bar = el('div');
    const w = (Math.abs(d.value) / mx) * (opts.signed ? 50 : 100);
    bar.style.cssText = `position:absolute;top:0;height:100%;border-radius:3px;background:${d.value >= 0 ? 'var(--jade)' : 'var(--amber)'};width:${w}%;` +
      (opts.signed ? (d.value >= 0 ? 'left:50%' : `right:50%`) : 'left:0');
    track.appendChild(bar);
    if (opts.signed) {
      const mid = el('div');
      mid.style.cssText = 'position:absolute;left:50%;top:0;width:1px;height:100%;background:var(--line-2)';
      track.appendChild(mid);
    }
    left.append(lab, track);
    const val = el('div', 'num', opts.fmt ? opts.fmt(d.value) : d.value.toFixed(2));
    val.style.cssText = `font-size:12px;text-align:right;color:${d.value >= 0 ? 'var(--jade)' : 'var(--amber)'}`;
    row.append(left, val);
    wrap.appendChild(row);
  }
  return wrap;
}

/** Stacked proportion bar for the quadrant mix. */
function stackBar(items) {
  const total = items.reduce((a, b) => a + b.count, 0) || 1;
  const wrap = el('div');
  wrap.style.cssText = 'display:flex;height:26px;border-radius:6px;overflow:hidden;border:1px solid var(--line)';
  const colors = { persuadable: 'var(--jade)', sure_thing: 'var(--steel)', lost_cause: 'var(--grey)', sleeping_dog: 'var(--amber)' };
  for (const it of items) {
    if (!it.count) continue;
    const seg = el('div');
    const share = (it.count / total) * 100;
    seg.style.cssText = `width:${share}%;background:${colors[it.key]};opacity:.85`;
    seg.title = `${it.label}: ${int(it.count)} (${share.toFixed(1)}%)`;
    wrap.appendChild(seg);
  }
  return wrap;
}

function panel(title, hint, bodyNode, bodyCls) {
  const p = el('div', 'panel');
  const head = el('div', 'panel-head');
  head.append(el('h3', null, title), el('span', 'hint', hint || ''));
  const body = el('div', 'panel-body' + (bodyCls ? ' ' + bodyCls : ''));
  if (bodyNode) body.appendChild(bodyNode);
  p.append(head, body);
  return { panel: p, body };
}

function kpi(label, value, foot, tone) {
  const k = el('div', 'kpi');
  k.append(el('span', 'kpi-label', label), el('div', 'kpi-value' + (tone ? ' ' + tone : ''), value));
  if (foot) k.appendChild(el('div', 'kpi-foot', foot));
  return k;
}

function table(cols, rows, opts = {}) {
  const wrap = el('div', 'table-wrap');
  const t = el('table');
  const thead = el('thead');
  const tr = el('tr');
  cols.forEach(c => { const th = el('th', c.right ? 'r' : '', c.label); tr.appendChild(th); });
  thead.appendChild(tr);
  const tb = el('tbody');
  rows.forEach(r => {
    const row = el('tr', opts.onRow ? 'clickable' : '');
    cols.forEach(c => {
      const td = el('td', (c.right ? 'r ' : '') + (c.strong ? 'strong ' : '') + (c.cls ? c.cls(r) : ''));
      const v = c.render ? c.render(r) : r[c.key];
      if (v instanceof Node) td.appendChild(v); else td.innerHTML = v ?? '';
      row.appendChild(td);
    });
    if (opts.onRow) row.onclick = () => opts.onRow(r);
    tb.appendChild(row);
  });
  t.append(thead, tb);
  wrap.appendChild(t);
  return rows.length ? wrap : el('div', 'empty', opts.empty || 'Nothing to show');
}

const qpill = (key, label) => `<span class="pill ${key}">${label}</span>`;

/* ================================================================== VIEWS */
const VIEWS = {};

/* ---------------------------------------------------- 1. Campaign Command */
VIEWS.command = {
  title: 'Campaign Command',
  sub: 'Incremental performance of the current champion against the propensity baseline it replaces.',
  async render(c) {
    const d = await api('/api/overview');
    S.overview = d;

    const k = d.kpi;
    const row = el('div', 'kpi-row');
    row.append(
      kpi('Addressable base', int(k.addressable), `of ${int(d.n_customers)} scored`),
      kpi('Expected incremental profit', money(k.expected_incremental_profit), `on ${money(k.expected_spend)} spend`, 'pos'),
      kpi('Incremental conversions', int(k.expected_incremental_conversions), 'above the do-nothing counterfactual'),
      kpi('Suppressed', int(k.suppressed), 'negative or zero uplift', 'neg'),
      kpi('Mean uplift', pp(k.mean_uplift_pp), 'typical contact, whole base'),
    );
    c.appendChild(row);

    // ---- Qini comparison
    const champ = d.qini.champion, prop = d.qini.propensity;
    const qini = lineChart([
      { points: champ.map(p => ({ x: p.x, y: p.random })), color: 'var(--ink-3)', dash: '3 4', width: 1.4 },
      { points: prop.map(p => ({ x: p.x, y: p.model })), color: 'var(--peri)', dash: '6 4', width: 1.9 },
      { points: champ.map(p => ({ x: p.x, y: p.model })), color: 'var(--jade)', width: 2.4, fill: true },
    ], { height: 250, xfmt: v => (v * 100).toFixed(0) + '%', yfmt: v => int(v) });

    const qBox = el('div');
    qBox.appendChild(qini);
    const leg = el('div', 'legend');
    leg.style.marginTop = '10px';
    leg.innerHTML = `
      <span><i style="background:var(--jade)"></i>${d.champion.name} — Qini ${int(d.headline.champion_qini)}</span>
      <span><i style="background:var(--peri)"></i>Propensity baseline — Qini ${int(d.headline.propensity_qini)}</span>
      <span><i style="background:var(--ink-3)"></i>Random targeting</span>`;
    qBox.appendChild(leg);

    const p1 = panel('Qini curve — cumulative incremental conversions', 'x = share of base targeted, ranked by score', qBox);

    // ---- quadrants
    const qWrap = el('div');
    const grid = el('div', 'quad-grid');
    const meta = {
      persuadable: 'Buy only if contacted',
      sure_thing: 'Buy either way',
      lost_cause: 'Never buy',
      sleeping_dog: 'Contact drives them away',
    };
    for (const q of d.quadrants) {
      const cell = el('div', 'quad-cell ' + q.key);
      cell.innerHTML = `<h4>${q.label}</h4><div class="n">${int(q.count)}</div>
        <div class="s">${pct(q.share)} · ${meta[q.key]}</div>`;
      grid.appendChild(cell);
    }
    qWrap.appendChild(grid);
    const p2 = panel('Population by uplift archetype', 'from the mean effect across offers', qWrap);

    const g1 = el('div', 'grid g-2-1');
    g1.append(p1.panel, p2.panel);
    c.appendChild(g1);

    // ---- deciles side by side
    const dc = el('div');
    dc.appendChild(divergingBars(d.deciles.champion.map(x => ({ label: 'D' + x.decile, value: x.uplift })), { height: 210 }));
    const dp = el('div');
    dp.appendChild(divergingBars(d.deciles.propensity.map(x => ({ label: 'D' + x.decile, value: x.uplift })), { height: 210 }));

    const pd1 = panel(`Observed uplift by decile — ${d.champion.name}`, 'treated minus control response, held-out set', dc);
    const pd2 = panel('Observed uplift by decile — propensity baseline', 'the model most teams actually ship', dp);
    const g2 = el('div', 'grid g-2');
    g2.append(pd1.panel, pd2.panel);
    c.appendChild(g2);

    const note = el('div', 'note');
    note.innerHTML = `<b>Read the two charts together.</b> The uplift model produces a monotone staircase that ends
      <span class="neg">${pp(d.deciles.champion.at(-1).uplift)}</span> — the bottom decile is actively harmed by contact.
      The propensity baseline's deciles are unordered, because ranking by <i>who will buy</i> says almost nothing about
      <i>who will buy because you asked</i>. Its correlation with true uplift is
      <b>${d.headline.propensity_corr.toFixed(3)}</b> against <b>${d.headline.champion_corr.toFixed(3)}</b> for the champion.`;
    c.appendChild(note);

    // ---- profit curve
    const pr = d.profit.champion;
    const prChart = lineChart([{
      points: pr.points.map(p => ({ x: p.x, y: p.profit })), color: 'var(--jade)', width: 2.4, fill: true,
      marker: { x: pr.optimal_fraction, y: pr.optimal_profit },
    }], { height: 220, xfmt: v => (v * 100).toFixed(0) + '%', yfmt: v => money(v) });
    const prBox = el('div');
    prBox.appendChild(prChart);
    prBox.appendChild(el('div', 'note', `<b>Optimal cutoff: ${(pr.optimal_fraction * 100).toFixed(1)}% of the base.</b>
      Contacting exactly that many people returns <b>${money(pr.optimal_profit)}</b> on the held-out sample.
      Contacting everyone returns ${money(pr.profit_at_full)} — the difference is margin handed to customers who
      would have converted anyway.`));
    c.appendChild(panel('Profit curve', `margin ${money2(d.economics.avg_margin_per_conversion)}/conversion · offer ${money2(d.economics.avg_offer_cost)} · contact $0.45`, prBox).panel);
  },
};

/* ---------------------------------------------------- 2. Audience Builder */
VIEWS.audience = {
  title: 'Audience Builder',
  sub: 'Set a budget and the optimiser fills it with the customers whose incremental profit per dollar is highest.',
  async render(c) {
    const f = S.filters;
    const state = {
      budget: 25000,
      offers: f.offers.map(o => o.key),
      segments: [], categorys: [], min_uplift_pp: 0,
    };

    // ---------- controls
    const ctr = el('div');
    const budgetRow = el('div');
    budgetRow.style.cssText = 'display:grid;grid-template-columns:1fr 190px;gap:20px;align-items:center;margin-bottom:18px';
    const slider = el('div');
    const sl = document.createElement('input');
    sl.type = 'range'; sl.min = 500; sl.max = 120000; sl.step = 500; sl.value = state.budget;
    const slLab = el('div');
    slLab.style.cssText = 'display:flex;justify-content:space-between;font-size:11px;color:var(--ink-3);margin-bottom:7px';
    slLab.innerHTML = `<span class="mono" style="letter-spacing:.12em;text-transform:uppercase;font-size:10px">Campaign budget</span><span id="budgetHint">drag to reallocate</span>`;
    slider.append(slLab, sl);
    const budgetVal = el('div');
    budgetVal.style.cssText = 'font-family:Archivo;font-size:30px;font-weight:700;letter-spacing:-.03em;text-align:right;font-variant-numeric:tabular-nums';
    budgetVal.textContent = money(state.budget);
    budgetRow.append(slider, budgetVal);

    const chipRow = (label, items, sel, onToggle) => {
      const fld = el('div', 'field');
      fld.appendChild(el('label', null, label));
      const chips = el('div', 'chips');
      items.forEach(it => {
        const b = el('button', 'chip' + (sel.includes(it.key) ? ' on' : ''), it.label);
        b.onclick = () => { onToggle(it.key); b.classList.toggle('on'); schedule(); };
        chips.appendChild(b);
      });
      fld.appendChild(chips);
      return fld;
    };

    const toggler = arr => k => { const i = arr.indexOf(k); i < 0 ? arr.push(k) : arr.splice(i, 1); };
    const controls = el('div', 'controls');
    controls.style.marginBottom = '4px';
    controls.append(
      chipRow('Offers in play', f.offers.map(o => ({ key: o.key, label: o.label })), state.offers, toggler(state.offers)),
      chipRow('Segments (all if none)', f.segments.map(s => ({ key: s, label: s })), state.segments, toggler(state.segments)),
      chipRow('Categorys (all if none)', f.categorys.map(v => ({ key: v, label: v })), state.categorys, toggler(state.categorys)),
    );
    ctr.append(budgetRow, controls);
    const pCtl = panel('Campaign parameters', 'every change re-runs the allocator over the full base', ctr);
    c.appendChild(pCtl.panel);

    // ---------- results
    const out = el('div');
    out.style.cssText = 'display:flex;flex-direction:column;gap:18px';
    c.appendChild(out);

    let timer = null;
    const schedule = () => { clearTimeout(timer); timer = setTimeout(run, 160); };
    sl.oninput = () => { state.budget = +sl.value; budgetVal.textContent = money(state.budget); schedule(); };

    async function run() {
      const r = await post('/api/audience', {
        budget: state.budget,
        offers: state.offers.length ? state.offers : null,
        segments: state.segments.length ? state.segments : null,
        categorys: state.categorys.length ? state.categorys : null,
        min_uplift_pp: state.min_uplift_pp,
        preview: 25,
      });
      out.innerHTML = '';

      const roi = r.spend > 0 ? r.expected_profit / r.spend : 0;
      const krow = el('div', 'kpi-row');
      krow.append(
        kpi('Customers selected', int(r.n_selected), `from a pool of ${int(r.pool_size)}`),
        kpi('Budget used', money(r.spend), `of ${money(r.budget)} available`),
        kpi('Expected incremental profit', money(r.expected_profit), `${roi.toFixed(2)}x return on spend`, 'pos'),
        kpi('Incremental conversions', int(r.expected_incremental_conversions), 'that would not have happened'),
        kpi('Suppressed', int(r.n_suppressed), 'negative uplift — never contact', 'neg'),
      );
      out.appendChild(krow);

      // quadrant mix + comparison
      const mixBox = el('div');
      mixBox.appendChild(stackBar(r.quadrant_mix));
      const legend = el('div', 'legend');
      legend.style.marginTop = '10px';
      const colors = { persuadable: 'var(--jade)', sure_thing: 'var(--steel)', lost_cause: 'var(--grey)', sleeping_dog: 'var(--amber)' };
      legend.innerHTML = r.quadrant_mix.map(q =>
        `<span><i style="background:${colors[q.key]}"></i>${q.label} · ${int(q.count)}</span>`).join('');
      mixBox.appendChild(legend);

      const dogs = r.quadrant_mix.find(q => q.key === 'sleeping_dog');
      const nDogs = dogs ? dogs.count : 0;
      const nb = el('div', 'note');
      nb.style.marginTop = '14px';
      nb.innerHTML = nDogs === 0
        ? `<b>${int(r.n_suppressed)} customers suppressed before the budget was applied.</b>
           Non-positive uplift means negative expected value at <i>any</i> budget, so these are excluded outright
           rather than competing for spend.`
        : `<b>${int(nDogs)} customers here are classified as sleeping dogs — and that is not a leak.</b>
           The quadrant describes how someone reacts to a <i>typical</i> contact. A customer who resents discount mail
           can still respond positively to one specific offer, and the optimiser only ever selects them paired with
           that offer. ${int(r.n_suppressed)} customers with no positive offer at all were suppressed outright.`;
      mixBox.appendChild(nb);

      const cmp = el('div');
      const naiveProfit = r.naive.profit, naiveSpend = r.naive.spend;
      cmp.appendChild(table(
        [
          { label: 'Strategy', key: 'name', strong: true },
          { label: 'Contacted', key: 'n', right: true },
          { label: 'Spend', key: 'spend', right: true },
          { label: 'Expected profit', key: 'profit', right: true, cls: () => 'strong' },
        ],
        [
          { name: 'Contact everyone in filter', n: int(r.naive.n), spend: money(naiveSpend), profit: money(naiveProfit) },
          { name: `Optimised at ${money(r.budget)}`, n: int(r.n_selected), spend: money(r.spend), profit: `<span class="pos">${money(r.expected_profit)}</span>` },
        ],
      ));
      const g = el('div', 'grid g-2');
      g.append(panel('Audience composition', 'who the budget actually bought', mixBox).panel,
               panel('Versus contacting everyone', 'same pool, no budget discipline', cmp).panel);
      out.appendChild(g);

      // preview table
      const prev = table(
        [
          { label: 'Customer', key: 'customer_id', strong: true },
          { label: 'Segment', key: 'segment' },
          { label: 'Quadrant', render: r2 => qpill(r2.quadrant, r2.quadrant_label) },
          { label: 'Best offer', key: 'offer_label' },
          { label: 'Uplift', right: true, render: r2 => `<span class="${cls(r2.uplift_pp)}">${pp(r2.uplift_pp)}</span>` },
          { label: 'Exp. profit', right: true, render: r2 => money2(r2.expected_profit) },
        ],
        r.preview, { onRow: row => openDrawer(row.customer_id), empty: 'Budget too small to select anyone' },
      );
      out.appendChild(panel(`Target list — top ${r.preview.length} of ${int(r.n_selected)}`, 'click a row for the full explanation', prev, 'tight').panel);
    }

    await run();
  },
};

/* ---------------------------------------------------------- 3. Customers */
VIEWS.customers = {
  title: 'Customers',
  sub: 'Every scored customer, their assigned offer, and why the model chose it.',
  async render(c) {
    const f = S.filters;
    S.cust.offset = 0;
    const st = { quadrant: 'all', segment: 'all', category: 'all', sort: 'expected_profit', q: '' };

    const ctr = el('div', 'controls');
    const mk = (label, opts, key) => {
      const fld = el('div', 'field');
      fld.appendChild(el('label', null, label));
      const s = document.createElement('select');
      opts.forEach(o => { const op = document.createElement('option'); op.value = o.v; op.textContent = o.l; s.appendChild(op); });
      s.value = st[key];
      s.onchange = () => { st[key] = s.value; S.cust.offset = 0; load(); };
      fld.appendChild(s);
      return fld;
    };
    ctr.append(
      mk('Quadrant', [{ v: 'all', l: 'All quadrants' }, ...f.quadrants.map(q => ({ v: q.key, l: q.label }))], 'quadrant'),
      mk('Segment', [{ v: 'all', l: 'All segments' }, ...f.segments.map(s => ({ v: s, l: s }))], 'segment'),
      mk('Category', [{ v: 'all', l: 'All categorys' }, ...f.categorys.map(v => ({ v, l: v }))], 'category'),
      mk('Sort by', [
        { v: 'expected_profit', l: 'Expected profit' },
        { v: 'offer_uplift', l: 'Uplift (best offer)' },
        { v: 'typical_uplift', l: 'Uplift (typical contact)' },
        { v: 'monetary_12m', l: 'Spend (12m)' },
        { v: 'p_control', l: 'Baseline demand' },
      ], 'sort'),
    );
    const searchFld = el('div', 'field');
    searchFld.appendChild(el('label', null, 'Customer ID'));
    const search = document.createElement('input');
    search.type = 'text'; search.placeholder = 'e.g. 4211';
    search.oninput = () => { st.q = search.value.trim(); S.cust.offset = 0; clearTimeout(search._t); search._t = setTimeout(load, 220); };
    searchFld.appendChild(search);
    ctr.appendChild(searchFld);
    c.appendChild(panel('Filters', '', ctr).panel);

    const host = el('div');
    c.appendChild(host);

    async function load() {
      const qs = new URLSearchParams({
        limit: S.cust.limit, offset: S.cust.offset, sort: st.sort,
        ...(st.quadrant !== 'all' ? { quadrant: st.quadrant } : {}),
        ...(st.segment !== 'all' ? { segment: st.segment } : {}),
        ...(st.category !== 'all' ? { category: st.category } : {}),
        ...(st.q ? { q: st.q } : {}),
      });
      const d = await api('/api/customers?' + qs);
      host.innerHTML = '';

      const t = table([
        { label: 'ID', key: 'customer_id', strong: true },
        { label: 'Category', key: 'category' },
        { label: 'Segment', key: 'segment' },
        { label: 'Quadrant', render: r => qpill(r.quadrant, r.quadrant_label) },
        { label: 'Best offer', key: 'offer_label' },
        { label: 'Uplift', right: true, render: r => `<span class="${cls(r.uplift_pp)}">${pp(r.uplift_pp)}</span>` },
        { label: 'Baseline', right: true, render: r => r.p_control_pct.toFixed(1) + '%' },
        { label: 'Exp. profit', right: true, render: r => `<span class="${cls(r.expected_profit)}">${money2(r.expected_profit)}</span>` },
        { label: 'Spend 12m', right: true, render: r => money(r.monetary_12m) },
        { label: 'Recency', right: true, render: r => int(r.recency_days) + 'd' },
      ], d.rows, { onRow: r => openDrawer(r.customer_id), empty: 'No customers match these filters' });

      const foot = el('div');
      foot.style.cssText = 'display:flex;justify-content:space-between;align-items:center;padding:12px 4px 0;font-size:12px;color:var(--ink-3)';
      const info = el('span', 'num', `${int(d.offset + 1)}–${int(Math.min(d.offset + d.limit, d.total))} of ${int(d.total)}`);
      const btns = el('div');
      btns.style.cssText = 'display:flex;gap:8px';
      const prev = el('button', 'btn ghost', 'Previous');
      const next = el('button', 'btn ghost', 'Next');
      prev.disabled = d.offset === 0; next.disabled = d.offset + d.limit >= d.total;
      prev.style.opacity = prev.disabled ? .4 : 1; next.style.opacity = next.disabled ? .4 : 1;
      prev.onclick = () => { S.cust.offset = Math.max(0, S.cust.offset - S.cust.limit); load(); };
      next.onclick = () => { S.cust.offset += S.cust.limit; load(); };
      btns.append(prev, next);
      foot.append(info, btns);

      const p = panel('Scored customers', 'click any row for the uplift explanation', t, 'tight');
      p.body.appendChild(foot);
      host.appendChild(p.panel);
    }
    await load();
  },
};

/* -------------------------------------------------------- 4. Offer Matrix */
VIEWS.offers = {
  title: 'Offer Matrix',
  sub: 'Mean uplift by segment and offer. The winning offer is not the same for every segment — that is the whole point of running five models.',
  async render(c) {
    const d = await api('/api/offer-matrix');
    const keys = d.offers.map(o => o.key);
    const all = d.matrix.flatMap(r => keys.map(k => r[k]));
    const mx = Math.max(...all.map(Math.abs), 0.01);

    const grid = el('div');
    grid.style.cssText = `display:grid;grid-template-columns:190px repeat(${keys.length},1fr);gap:5px;min-width:640px`;
    grid.appendChild(el('div'));
    d.offers.forEach(o => {
      const h = el('div', null, o.label);
      h.style.cssText = 'font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);font-family:"IBM Plex Mono",monospace;text-align:center;padding:0 2px 4px';
      grid.appendChild(h);
    });

    d.matrix.forEach(row => {
      const lab = el('div');
      lab.style.cssText = 'font-size:12.5px;display:flex;align-items:center;gap:8px;color:var(--ink)';
      lab.innerHTML = `<span>${row.segment}</span><span class="tag">${int(row.n)}</span>`;
      grid.appendChild(lab);

      const best = keys.reduce((a, b) => (row[b] > row[a] ? b : a), keys[0]);
      keys.forEach(k => {
        const v = row[k];
        const a = Math.min(1, Math.abs(v) / mx);
        const cell = el('div', 'heat', pp(v));
        cell.style.background = v >= 0
          ? `color-mix(in srgb, var(--jade) ${(a * 78).toFixed(0)}%, transparent)`
          : `color-mix(in srgb, var(--amber) ${(a * 78).toFixed(0)}%, transparent)`;
        cell.style.color = a > .5 ? 'var(--bg)' : 'var(--ink)';
        if (k === best) cell.style.boxShadow = 'inset 0 0 0 1.5px var(--jade)';
        cell.title = `${row.segment} × ${k}: ${pp(v)}${k === best ? ' (best)' : ''}`;
        grid.appendChild(cell);
      });
    });

    const wrap = el('div');
    wrap.style.overflowX = 'auto';
    wrap.appendChild(grid);
    const box = el('div');
    box.appendChild(wrap);
    box.appendChild(el('div', 'note', `<b>The outlined cell in each row is that segment's best offer.</b>
      A single-offer campaign is forced to pick one column for everybody; running one uplift model per offer lets each
      segment take its own maximum. That difference is free margin.`));
    c.appendChild(panel('Mean uplift by segment × offer', 'diverging scale — orange is negative uplift', box).panel);
  },
};

/* ------------------------------------------------------- 5. Model bake-off */
VIEWS.models = {
  title: 'Model Bake-off',
  sub: 'Five estimators on the same held-out split. Because the data is simulated we also know each customer\'s true effect, so the ranking rests on measured error rather than a plausible-looking curve.',
  async render(c) {
    const d = await api('/api/bakeoff');

    const rows = d.bakeoff.map(b => ({ ...b, isChamp: b.key === d.champion.key }));
    const t = table([
      { label: 'Model', strong: true, render: r => r.name + (r.isChamp ? ' <span class="tag" style="color:var(--jade);border-color:var(--jade)">champion</span>' : '') },
      { label: 'Type', render: r => r.is_uplift ? 'Uplift' : '<span style="color:var(--amber)">Propensity</span>' },
      { label: 'Qini', right: true, render: r => `<span class="${cls(r.qini)}">${int(r.qini)}</span>` },
      { label: '95% CI', right: true, render: r => `<span class="num" style="font-size:11.5px;color:var(--ink-3)">${int(r.qini_ci.lo)} … ${int(r.qini_ci.hi)}</span>` },
      { label: 'Corr vs truth', right: true, render: r => `<span class="${cls(r.ground_truth.corr)}">${r.ground_truth.corr.toFixed(3)}</span>` },
      { label: 'MAE', right: true, render: r => r.ground_truth.mae.toFixed(4) },
      { label: 'Top decile', right: true, render: r => `<span class="${cls(r.top_decile_uplift)}">${pp(r.top_decile_uplift)}</span>` },
      { label: 'Bottom decile', right: true, render: r => `<span class="${cls(r.bottom_decile_uplift)}">${pp(r.bottom_decile_uplift)}</span>` },
      { label: 'Fit', right: true, render: r => r.fit_seconds + 's' },
    ], rows);
    c.appendChild(panel('Estimator comparison', 'held-out 30% split, bootstrap CI over 40 resamples', t, 'tight').panel);

    const note = el('div', 'note');
    note.innerHTML = `<b>Correlation against ground truth is the column that matters.</b> Qini rewards a good ranking,
      but two models can share a Qini and disagree wildly about magnitudes. The propensity baseline shows the failure
      this project exists to demonstrate: a respectable-looking classifier whose correlation with true uplift is
      approximately zero. It is not a worse uplift model — it is not an uplift model.`;
    c.appendChild(note);

    const ot = table([
      { label: 'Offer', key: 'label', strong: true },
      { label: 'Qini', right: true, render: r => int(r.qini) },
      { label: 'Corr vs truth', right: true, render: r => `<span class="${cls(r.corr_vs_truth)}">${r.corr_vs_truth.toFixed(3)}</span>` },
      { label: 'True mean uplift', right: true, render: r => pp(r.mean_uplift_pp) },
      { label: 'Training rows', right: true, render: r => int(r.n_train) },
    ], d.offers);
    c.appendChild(panel('Per-offer models', `one ${d.champion.name} per treatment arm, each trained against the shared control group`, ot, 'tight').panel);

    // ---- the same estimators on a genuine RCT
    const rd = await api('/api/real-data');
    if (!rd.available) {
      c.appendChild(el('div', 'note', `<b>Real-data validation not run yet.</b>
        <code class="mono">python -m nextbest.realdata</code> re-runs these five estimators on the Hillstrom
        randomised email experiment and this panel fills in.`));
      return;
    }

    const rt = table([
      { label: 'Model', strong: true, render: r => r.name + (r.key === rd.champion.key ? ' <span class="tag" style="color:var(--jade);border-color:var(--jade)">best</span>' : '') },
      { label: 'Type', render: r => r.is_uplift ? 'Uplift' : '<span style="color:var(--amber)">Propensity</span>' },
      { label: 'Qini', right: true, render: r => `<span class="${cls(r.qini)}">${int(r.qini)}</span>` },
      { label: '95% CI', right: true, render: r => {
        const sig = r.qini_ci.lo > 0;
        return `<span class="num" style="font-size:11.5px;color:${sig ? 'var(--jade)' : 'var(--ink-3)'}">${int(r.qini_ci.lo)} … ${int(r.qini_ci.hi)}</span>`;
      } },
      { label: 'Beats random?', right: true, render: r => r.qini_ci.lo > 0
        ? '<span class="pos">yes</span>'
        : '<span style="color:var(--ink-3)">not proven</span>' },
      { label: 'Top decile', right: true, render: r => `<span class="${cls(r.top_decile_uplift)}">${pp(r.top_decile_uplift)}</span>` },
      { label: 'Bottom decile', right: true, render: r => `<span class="${cls(r.bottom_decile_uplift)}">${pp(r.bottom_decile_uplift)}</span>` },
    ], rd.results);

    const rbox = el('div');
    rbox.appendChild(rt);

    const armT = table([
      { label: 'Arm', key: 'arm', strong: true },
      { label: 'Customers', right: true, render: r => int(r.n) },
      { label: `${rd.outcome} rate`, right: true, render: r => r.rate.toFixed(2) + '%' },
      { label: 'Lift vs control', right: true, render: r => r.arm === 'No E-Mail'
        ? '<span style="color:var(--ink-3)">baseline</span>'
        : `<span class="${cls(r.lift_pp)}">${pp(r.lift_pp)}</span>` },
    ], rd.arms);
    rbox.appendChild(el('div', 'sec-title', 'Observed experiment arms'));
    rbox.appendChild(armT);

    rbox.appendChild(el('div', 'note warn', `<b>No ground-truth column exists here, and that is the lesson.</b>
      On the simulated data every model can be scored against each customer's true effect. On this genuine experiment
      only the ranking can be judged — which is why the confidence interval carries the verdict. A Qini whose interval
      contains zero has not been shown to beat shuffling the list, however respectable the point estimate looks.`));

    c.appendChild(panel(
      'Real randomised experiment — Hillstrom MineThatData',
      `${int(rd.n)} customers · ${rd.treated_share}% treated · ${rd.outcome} rate ${rd.outcome_rate}% · bootstrap CI over 40 resamples`,
      rbox, 'tight').panel);
  },
};

/* --------------------------------------------------------- 6. Model Health */
VIEWS.health = {
  title: 'Model Health',
  sub: 'What drives the uplift score, how the predictions are distributed, and whether the randomisation actually held.',
  async render(c) {
    const d = await api('/api/model-health');

    const imp = el('div');
    imp.appendChild(hBars(
      d.importance.slice(0, 10).map(i => ({ label: i.label, value: i.importance })),
      { fmt: v => v.toFixed(2) + 'pp' },
    ));
    const impBox = el('div');
    impBox.appendChild(imp);
    impBox.appendChild(el('div', 'note', `Mean absolute change in predicted uplift when each feature is reset to the
      population median. This is a one-at-a-time sensitivity, not a Shapley value — it ignores interactions, and it is
      labelled as such rather than dressed up as SHAP.`));

    const hist = el('div');
    hist.appendChild(histogram(d.distribution, { height: 190 }));
    hist.appendChild(el('div', 'legend', `<span><i style="background:var(--jade)"></i>positive uplift</span><span><i style="background:var(--amber)"></i>negative uplift — suppress</span>`));

    const g = el('div', 'grid g-2');
    g.append(panel('Uplift drivers', 'global view, sampled', impBox).panel,
             panel('Predicted uplift distribution', 'percentage points, whole base', hist).panel);
    c.appendChild(g);

    const arm = table([
      { label: 'Arm', key: 'arm', strong: true },
      { label: 'Customers', right: true, render: r => int(r.n) },
      { label: 'Conversion rate', right: true, render: r => r.conversion_rate.toFixed(2) + '%' },
      { label: 'Lift vs control', right: true, render: r => {
        const ctrl = d.arm_rates.find(a => a.arm === 'control').conversion_rate;
        const v = r.conversion_rate - ctrl;
        return r.arm === 'control' ? '<span style="color:var(--ink-3)">baseline</span>' : `<span class="${cls(v)}">${pp(v)}</span>`;
      } },
    ], d.arm_rates);

    const gt = d.champion_ground_truth;
    const meta = el('div', 'kv');
    meta.innerHTML = `
      <div><dt>Population</dt><dd class="num">${int(d.n_customers)}</dd></div>
      <div><dt>Booster</dt><dd style="font-size:11.5px">${d.booster}</dd></div>
      <div><dt>Corr vs truth</dt><dd class="num pos">${gt.corr.toFixed(3)}</dd></div>
      <div><dt>MAE</dt><dd class="num">${gt.mae.toFixed(4)}</dd></div>
      <div><dt>RMSE</dt><dd class="num">${gt.rmse.toFixed(4)}</dd></div>
      <div><dt>Trained</dt><dd style="font-size:11.5px">${d.generated_at}</dd></div>`;

    const armBox = el('div');
    armBox.append(arm);
    armBox.appendChild(el('div', 'note', `<b>This is the randomisation check.</b> Arm sizes should match the design
      (25% control, 15% per offer) and the control conversion rate is the counterfactual every uplift number is measured
      against. If these drift apart, nothing downstream is trustworthy.`));

    const g2 = el('div', 'grid g-2');
    g2.append(panel('Experiment arms', 'observed conversion by treatment arm', armBox).panel,
              panel('Champion diagnostics', 'error against known ground truth', meta).panel);
    c.appendChild(g2);
  },
};

/* ------------------------------------------------------------- drawer */
async function openDrawer(id) {
  const dr = $('#drawer'), sc = $('#scrim');
  dr.classList.add('open'); sc.classList.add('open');
  dr.innerHTML = `<div class="drawer-head"><div><h3 style="font-size:16px">Customer ${id}</h3></div></div>
    <div class="drawer-body"><div class="skeleton" style="height:180px"></div><div class="skeleton" style="height:220px"></div></div>`;

  const d = await api('/api/customer/' + id);
  dr.innerHTML = '';

  const head = el('div', 'drawer-head');
  head.innerHTML = `<div>
      <h3 style="font-size:17px;letter-spacing:-.02em">Customer ${d.customer_id}</h3>
      <div style="display:flex;gap:7px;align-items:center;margin-top:6px;flex-wrap:wrap">
        ${qpill(d.quadrant, d.quadrant_label)}
        <span class="tag">${d.category}</span><span class="tag">${d.segment}</span>
      </div></div>`;
  const close = el('button', 'icon-btn', 'Close');
  close.onclick = closeDrawer;
  head.appendChild(close);

  const body = el('div', 'drawer-body');

  // recommendation
  const rec = el('div');
  const isContact = d.recommendation.action === 'contact';
  rec.style.cssText = `border:1px solid ${isContact ? 'var(--jade)' : 'var(--amber)'};background:${isContact ? 'var(--jade-soft)' : 'var(--amber-soft)'};border-radius:8px;padding:14px 16px`;
  rec.innerHTML = `
    <div class="sec-title" style="color:${isContact ? 'var(--jade)' : 'var(--amber)'};margin-bottom:6px">
      ${isContact ? 'Recommended action — contact' : 'Recommended action — suppress'}</div>
    <div style="font-family:Archivo;font-size:19px;font-weight:700;letter-spacing:-.02em">${d.recommendation.offer_label}</div>
    <div style="font-size:12.5px;color:var(--ink-2);margin-top:5px">
      Expected incremental profit <b style="color:var(--ink)">${money2(d.recommendation.expected_profit)}</b>
      at a cost of ${money2(d.recommendation.expected_cost)}</div>`;
  body.appendChild(rec);

  // counterfactual
  const cf = d.counterfactual;
  const cfBox = el('div');
  const mkBar = (label, v, color) => {
    const r = el('div');
    r.style.cssText = 'display:grid;grid-template-columns:150px 1fr 62px;gap:11px;align-items:center;margin-bottom:8px';
    const track = el('div');
    track.style.cssText = 'height:9px;background:var(--panel-3);border-radius:5px;overflow:hidden';
    const bar = el('div');
    bar.style.cssText = `height:100%;width:${Math.min(100, (v / Math.max(cf.p_treated_pct, cf.p_control_pct, 1)) * 100)}%;background:${color};border-radius:5px`;
    track.appendChild(bar);
    r.append(el('div', null, `<span style="font-size:12px;color:var(--ink-2)">${label}</span>`), track,
             el('div', 'num', `<span style="font-size:12.5px">${v.toFixed(2)}%</span>`));
    return r;
  };
  cfBox.append(
    el('div', 'sec-title', 'Counterfactual — the two worlds'),
    mkBar('If contacted', cf.p_treated_pct, 'var(--jade)'),
    mkBar('If left alone', cf.p_control_pct, 'var(--steel)'),
  );
  const delta = el('div');
  delta.style.cssText = 'font-size:12.5px;color:var(--ink-2);margin-top:4px';
  delta.innerHTML = `Difference: <b class="${cls(cf.uplift_pp)}">${pp(cf.uplift_pp)}</b> — only one of these two worlds is ever observed for a real customer.`;
  cfBox.appendChild(delta);
  body.appendChild(cfBox);

  // drivers
  const dv = el('div');
  dv.append(el('div', 'sec-title', 'What drives this uplift score'));
  dv.appendChild(hBars(d.drivers.map(x => ({ label: `${x.label} = ${x.value}`, value: x.contribution })),
    { signed: true, fmt: v => pp(v) }));
  body.appendChild(dv);

  // offers
  const ot = el('div');
  ot.append(el('div', 'sec-title', 'Every offer, scored'));
  ot.appendChild(table([
    { label: 'Offer', strong: true, render: r => r.label + (r.chosen ? ' <span class="tag" style="color:var(--jade);border-color:var(--jade)">chosen</span>' : '') },
    { label: 'Uplift', right: true, render: r => `<span class="${cls(r.uplift_pp)}">${pp(r.uplift_pp)}</span>` },
    { label: 'Cost', right: true, render: r => money2(r.cost) },
    { label: 'Exp. profit', right: true, render: r => `<span class="${cls(r.expected_profit)}">${money2(r.expected_profit)}</span>` },
  ], d.offers));
  body.appendChild(ot);

  // profile
  const pf = el('div');
  pf.append(el('div', 'sec-title', 'Profile'));
  const kv = el('div', 'kv');
  const P = d.profile;
  kv.innerHTML = `
    <div><dt>Recency</dt><dd class="num">${int(P.recency_days)}d</dd></div>
    <div><dt>Frequency 12m</dt><dd class="num">${P.frequency_12m}</dd></div>
    <div><dt>Spend 12m</dt><dd class="num">${money(P.monetary_12m)}</dd></div>
    <div><dt>Avg order</dt><dd class="num">${money2(P.avg_order_value)}</dd></div>
    <div><dt>Tenure</dt><dd class="num">${P.tenure_years}y</dd></div>
    <div><dt>Engagement</dt><dd class="num">${P.engagement}</dd></div>
    <div><dt>Discount affinity</dt><dd class="num">${P.discount_affinity}</dd></div>
    <div><dt>Price tier</dt><dd class="num">${P.price_tier}</dd></div>
    <div><dt>Membership</dt><dd>${P.is_registered ? 'Member' : 'No'}</dd></div>`;
  pf.appendChild(kv);
  body.appendChild(pf);

  dr.append(head, body);
}

function closeDrawer() {
  $('#drawer').classList.remove('open');
  $('#scrim').classList.remove('open');
}

/* ------------------------------------------------------------- routing */
async function go(view) {
  S.view = view;
  document.querySelectorAll('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.view === view));
  const v = VIEWS[view];
  $('#viewTitle').textContent = v.title;
  $('#viewSub').textContent = v.sub;
  const c = $('#content');
  c.innerHTML = '<div class="skeleton" style="height:96px"></div><div class="skeleton" style="height:300px"></div>';
  try {
    c.innerHTML = '';
    await v.render(c);
  } catch (e) {
    c.innerHTML = `<div class="panel"><div class="panel-body"><b style="color:var(--danger)">Failed to render.</b>
      <pre class="mono" style="font-size:11.5px;color:var(--ink-2);white-space:pre-wrap;margin-top:8px">${e.message}</pre></div></div>`;
    console.error(e);
  }
}

/* ---------------------------------------------------------------- boot */
(async function boot() {
  document.querySelectorAll('.nav-item').forEach(b => { b.onclick = () => go(b.dataset.view); });
  $('#scrim').onclick = closeDrawer;
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });

  const tb = $('#themeBtn');
  const saved = localStorage.getItem('nb-theme') || 'dark';
  document.documentElement.dataset.theme = saved;
  tb.textContent = saved === 'dark' ? 'Light' : 'Dark';
  tb.onclick = () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('nb-theme', next);
    tb.textContent = next === 'dark' ? 'Light' : 'Dark';
    go(S.view);
  };

  try {
    S.filters = await api('/api/filters');
    const ov = await api('/api/overview');
    $('#champBadge').innerHTML = `Champion<br><b>${ov.champion.name}</b><br>
      ${int(ov.n_customers)} customers · ${ov.booster.replace('sklearn ', '')}`;
  } catch (e) {
    $('#content').innerHTML = `<div class="panel"><div class="panel-body">
      <b style="color:var(--danger)">Could not load artifacts.</b>
      <p style="color:var(--ink-2);font-size:13px">Run <code class="mono">python -m nextbest.train</code> first, then restart the API.</p>
      <pre class="mono" style="font-size:11.5px;color:var(--ink-3)">${e.message}</pre></div></div>`;
    return;
  }
  go('command');
})();
