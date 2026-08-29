/* ==========================================================================
   NextBest — Retail Customer Intelligence
   Four tabs, plain language, no framework. Charts are hand-built SVG that read
   their colours from CSS variables, so a theme switch repaints them for free.
   ====================================================================== */

const S = { data: null, tab: 'models',
            cf: { action: 'all', category: 'all', channel: 'all', reward: 'all',
                  q: '', sort: 'value_of_contact', offset: 0, limit: 40 } };

/* ------------------------------------------------------- static mode ----
   GitHub Pages serves files, not Python. Every endpoint here is either a
   lookup or arithmetic, so when a data bundle is present the same UI answers
   its own requests locally. /api/whatif is the exception -- it needs the fitted
   models -- so that feature hides itself rather than faking a number. */
let NB = null;                       // the bundle, when running static

async function loadStatic() {
  try {
    const r = await fetch('data.json', { cache: 'force-cache' });
    if (!r.ok) return false;
    NB = await r.json();
    NB.index = new Map(NB.columns.map((c, i) => [c, i]));
    NB.byId = new Map(NB.rows.map(row => [row[NB.index.get('customer_id')], row]));
    return true;
  } catch (_) { return false; }
}

/** Read a cell, decoding the dictionary-encoded string columns. */
const col = (row, name) => {
  const v = row[NB.index.get(name)];
  const d = NB.dict && NB.dict[name];
  return d ? d[v] : v;
};

/** The same formula the server used: value a chosen list against the real
 *  held-back group, then price it at the caller's margin and contact cost. */
function priceList(pick, margin, cost) {
  const { uplift, t, y, rev_per_incremental_visit: rev } = NB.eval;
  let n = 0, nt = 0, nc = 0, yt = 0, yc = 0;
  for (let i = 0; i < uplift.length; i++) {
    if (!pick[i]) continue;
    n++;
    if (t[i]) { nt++; yt += y[i]; } else { nc++; yc += y[i]; }
  }
  if (!n || !nt || !nc) return { value: 0, extra: 0, n };
  const lift = yt / nt - yc / nc;
  const extra = lift * n;
  return { value: extra * rev * margin - n * cost, extra, n };
}

function staticRecompute({ margin, cost }) {
  const { uplift, rev_per_incremental_visit: rev, lift_all, n_base } = NB.eval;
  const N = uplift.length, scale = n_base / N;
  const mpv = rev * margin;
  const everyone = new Uint8Array(N).fill(1);
  const blanket = priceList(everyone, margin, cost);

  const pick = new Uint8Array(N);
  for (let i = 0; i < N; i++) pick[i] = (uplift[i] * mpv) > cost ? 1 : 0;
  const targeted = priceList(pick, margin, cost);

  // Mirrors the verdict rules in realbuild.py. A campaign that still loses money
  // is not a targeting win, and a list of 34 people is not a campaign.
  const MIN_SHARE = 0.005;
  const tgScaled = targeted.value * scale;
  let verdict, advice;
  if (!targeted.n) {
    verdict = 'stop'; advice = 'Do not run it. Nobody is worth this cost.';
  } else if (tgScaled <= 0) {
    verdict = 'stop';
    advice = 'Do not run it at this price. Even the best list still loses money.';
  } else if (targeted.n < N * MIN_SHARE) {
    verdict = 'stop';
    advice = `Only ${int(Math.round(targeted.n * scale))} customers clear this cost — too few to run as a campaign.`;
  } else if (targeted.value - blanket.value > Math.abs(blanket.value) * 0.02 && targeted.n < N * 0.95) {
    verdict = 'target'; advice = `Target ${(targeted.n / N * 100).toFixed(0)}% of the base.`;
  } else { verdict = 'blanket'; advice = 'Contact everyone. A model adds nothing here.'; }

  const curve = [];
  for (let k = 1; k <= 50; k++) {
    const c = +(k * 0.02).toFixed(2);
    const b = priceList(everyone, margin, c);
    const pk = new Uint8Array(N);
    for (let i = 0; i < N; i++) pk[i] = (uplift[i] * mpv) > c ? 1 : 0;
    const tg = priceList(pk, margin, c);
    curve.push({ cost: c, blanket: +(b.value * scale).toFixed(2),
                 targeted: +(tg.value * scale).toFixed(2), n: Math.round(tg.n * scale) });
  }

  const bu = NB.base.uplift, bp = NB.base.p_control;
  const sorted = [...bp].sort((a, b) => a - b);
  const demand = sorted[Math.floor(sorted.length / 2)];
  const acts = NB.metrics.actions;
  const count = {}; Object.keys(acts).forEach(k => count[k] = 0);
  for (let i = 0; i < bu.length; i++) {
    const v = bu[i] * mpv - cost;
    const k = bu[i] < -0.002 ? 'suppress' : v > 0 ? 'contact'
            : bp[i] >= demand ? 'no_offer' : 'not_worth';
    count[k]++;
  }

  return {
    margin, cost,
    breakeven: +(lift_all * mpv).toFixed(4),
    margin_per_visit: +mpv.toFixed(2),
    blanket: +(blanket.value * scale).toFixed(2),
    targeted: +(targeted.value * scale).toFixed(2),
    gain: +((targeted.value - blanket.value) * scale).toFixed(2),
    gain_is_avoided_loss: tgScaled <= 0,
    n_targeted: Math.round(targeted.n * scale),
    share_targeted: +(targeted.n / N * 100).toFixed(1),
    extra_visits: Math.round(targeted.extra * scale),
    verdict, advice, curve, n_base,
    groups: Object.keys(acts).map(k => ({
      key: k, label: acts[k].label, tone: acts[k].tone,
      n: count[k], share: +(count[k] / bu.length * 100).toFixed(1),
    })),
  };
}

function staticCustomers(qs) {
  const P = Object.fromEntries(new URLSearchParams(qs));
  let rows = NB.rows;
  for (const [k, c] of [['action', 'action'], ['category', 'category'],
                        ['channel', 'channel'], ['reward', 'reward']]) {
    if (P[k] && P[k] !== 'all') rows = rows.filter(r => col(r, c) === P[k]);
  }
  if (P.q) {
    const n = parseInt(P.q, 10);
    rows = Number.isNaN(n)
      ? rows.filter(r => String(col(r, 'spend_band')).toLowerCase().includes(P.q.toLowerCase()))
      : rows.filter(r => col(r, 'customer_id') === n);
  }
  const sort = NB.index.has(P.sort || '') ? P.sort : 'value_of_contact';
  const asc = sort === 'months_since_purchase';
  rows = [...rows].sort((a, b) => (asc ? 1 : -1) * (col(a, sort) - col(b, sort)));

  const limit = +(P.limit || 40), offset = +(P.offset || 0);
  const CAT = NB.metrics.categories.reduce((m, c) => (m[c.key] = c.label, m), {});
  const CH = NB.metrics.channels.reduce((m, c) => (m[c.key] = c.label, m), {});
  return {
    total: rows.length, offset, limit,
    priced_at: NB.metrics.priced_at.customers,
    rows: rows.slice(offset, offset + limit).map(r => ({
      customer_id: col(r, 'customer_id'),
      buys: CAT[col(r, 'category')] || col(r, 'category'),
      spend_band: col(r, 'spend_band'),
      spend_12m: col(r, 'spend_12m'),
      months_since_purchase: col(r, 'months_since_purchase'),
      new_customer: !!col(r, 'new_customer'),
      area: col(r, 'area'),
      channel: CH[col(r, 'channel')] || col(r, 'channel'),
      extra_sales_pp: col(r, 'extra_sales_pp'),
      value_of_contact: col(r, 'value_of_contact'),
      action: col(r, 'action'), reward: col(r, 'reward'),
    })),
  };
}

function staticCustomer(id) {
  const r = NB.byId.get(+id);
  if (!r) throw new Error('404 customer not found');
  const M = NB.metrics, act = M.actions[col(r, 'action')];
  const rew = M.rewards.find(x => x.key === col(r, 'reward'));
  const CAT = M.categories.reduce((m, c) => (m[c.key] = c.label, m), {});
  const CH = M.channels.reduce((m, c) => (m[c.key] = c.label, m), {});
  return {
    customer_id: col(r, 'customer_id'),
    known: {
      months_since_purchase: col(r, 'months_since_purchase'),
      spend_12m: col(r, 'spend_12m'), spend_band: col(r, 'spend_band'),
      buys: CAT[col(r, 'category')] || col(r, 'category'),
      new_customer: !!col(r, 'new_customer'),
      area: col(r, 'area'), channel: CH[col(r, 'channel')] || col(r, 'channel'),
    },
    happened: { arm: col(r, 'arm'), visited: !!col(r, 'visited'), spent: col(r, 'spent') },
    prediction: {
      buys_alone_pct: col(r, 'buys_alone_pct'),
      buys_if_contacted_pct: col(r, 'buys_if_contacted_pct'),
      extra_sales_pp: col(r, 'extra_sales_pp'),
    },
    decision: {
      action: col(r, 'action'), label: act.label, tone: act.tone, why: act.why,
      reward: rew ? rew.label : null, reward_why: rew ? rew.why : null,
      value: col(r, 'value_of_contact'), cost: col(r, 'cost_of_contact'),
    },
  };
}

/** Build the CSV in the browser and hand it to the download. */
function staticExport(action, cost, margin, filt) {
  const F = filt || {};
  const mpv = NB.eval.rev_per_incremental_visit * margin;
  const bu = NB.base.uplift, bp = NB.base.p_control;
  const sorted = [...bp].sort((a, b) => a - b);
  const demand = sorted[Math.floor(sorted.length / 2)];

  const head = ['customer_id', 'months_since_last_purchase', 'spend_past_year', 'spend_band',
                'buys_mens', 'buys_womens', 'new_customer', 'area', 'channel',
                'predicted_extra_visits_pp', 'expected_value', 'action', 'send_which_email'];
  const lines = [head.join(',')];
  NB.rows.forEach((r, i) => {
    const v = bu[i] * mpv - cost;
    const a = bu[i] < -0.002 ? 'suppress' : v > 0 ? 'contact'
            : bp[i] >= demand ? 'no_offer' : 'not_worth';
    if (action !== 'all' && a !== action) return;
    // the file has to be the list that was on screen
    if (F.category && F.category !== 'all' && col(r, 'category') !== F.category) return;
    if (F.channel && F.channel !== 'all' && col(r, 'channel') !== F.channel) return;
    if (F.reward && F.reward !== 'all' && col(r, 'reward') !== F.reward) return;
    if (F.q) {
      const qn = parseInt(F.q, 10);
      const hit = Number.isNaN(qn)
        ? String(col(r, 'spend_band')).toLowerCase().includes(F.q.toLowerCase())
        : col(r, 'customer_id') === qn;
      if (!hit) return;
    }
    const q = s => /[",\n]/.test(String(s)) ? `"${String(s).replace(/"/g, '""')}"` : s;
    lines.push([col(r, 'customer_id'), col(r, 'months_since_purchase'), col(r, 'spend_12m'),
                q(col(r, 'spend_band')), col(r, 'buys_mens'), col(r, 'buys_womens'),
                col(r, 'new_customer'), q(col(r, 'area')), q(col(r, 'channel')),
                col(r, 'extra_sales_pp'), v.toFixed(4), a, col(r, 'reward')].join(','));
  });

  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const tags = ['category', 'channel', 'reward']
    .map(k => F[k]).filter(x => x && x !== 'all').map(x => '_' + x).join('');
  a.download = `nextbest_${action}${tags}_${lines.length - 1}rows_cost${cost.toFixed(2)}_margin${Math.round(margin * 100)}pct.csv`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const $ = s => document.querySelector(s);
const el = (t, c, h) => { const n = document.createElement(t); if (c) n.className = c; if (h != null) n.innerHTML = h; return n; };
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

async function api(p, o) {
  if (NB) {                                   // static: answer it here
    if (p.startsWith('/api/overview')) return NB.metrics;
    if (p.startsWith('/api/customers')) return staticCustomers(p.split('?')[1] || '');
    if (p.startsWith('/api/customer/')) return staticCustomer(p.split('/').pop());
    if (p.startsWith('/api/options')) return { categories: NB.metrics.categories,
                                               channels: NB.metrics.channels };
  }
  const r = await fetch(p, o);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}
async function post(p, b) {
  if (NB && p.startsWith('/api/recompute')) return staticRecompute(b);
  return api(p, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify(b) });
}

const int = n => Math.round(n).toLocaleString('en-US');
const money = n => (n < 0 ? '−$' : '$') + Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const money0 = n => (n < 0 ? '−$' : '$') + int(Math.abs(n));
const pts = n => (n >= 0 ? '+' : '−') + Math.abs(n).toFixed(1);
const dir = n => (n > 0 ? 'up' : n < 0 ? 'dn' : '');

/** Profit against cost-per-contact: two lines, the break-even marked, and a
 *  dot showing where the current assumptions put you. */
function costChart(curve, cost, breakeven, h = 260) {
  const w = 680, m = { t: 16, r: 16, b: 34, l: 66 };
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const xs = curve.map(d => d.cost);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  // The blanket line falls to about -$53k at the top of the price range. Scaling
  // to both lines let that tail own the domain and squashed the targeted line --
  // the one the user is actually deciding on -- into a flat squiggle near zero.
  // Scale to the targeted line and let blanket run off-plot, where it is clipped.
  const ts = curve.map(d => d.targeted);
  let y0 = Math.min(0, ...ts), y1 = Math.max(0, ...ts);
  const pad = ((y1 - y0) || 1) * .12; y0 -= pad; y1 += pad;
  // Round the domain outward to a 1/2/5 step so every tick is a round number.
  const raw = (y1 - y0) / 4;
  const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(raw) || 1)));
  const nm = raw / mag;
  const step = (nm <= 1 ? 1 : nm <= 2 ? 2 : nm <= 5 ? 5 : 10) * mag;
  y0 = Math.floor(y0 / step) * step;
  y1 = Math.ceil(y1 / step) * step;
  const X = v => m.l + ((v - x0) / (x1 - x0 || 1)) * iw;
  const Y = v => m.t + ih - ((v - y0) / (y1 - y0 || 1)) * ih;
  const NS = 'http://www.w3.org/2000/svg';
  const sv = (t, a) => { const n = document.createElementNS(NS, t); for (const k in a) n.setAttribute(k, a[k]); return n; };
  const tx = (s2, a) => { const n = sv('text', a); n.textContent = s2; return n; };

  const g = sv('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%', height: h, role: 'img',
    'aria-label': 'Profit against cost per contact, for sending to everyone versus a targeted list' });

  const cid = 'clip' + (costChart._n = (costChart._n || 0) + 1);
  const cp = sv('clipPath', { id: cid });
  cp.appendChild(sv('rect', { x: m.l, y: m.t, width: iw, height: ih }));
  g.appendChild(cp);
  for (let v = y0; v <= y1 + step * 1e-6; v += step) {
    g.appendChild(sv('line', { x1: m.l, x2: w - m.r, y1: Y(v), y2: Y(v), stroke: 'var(--line)' }));
    g.appendChild(tx(money0(v), { x: m.l - 9, y: Y(v) + 3.4, 'text-anchor': 'end', class: 'ax' }));
  }
  if (y0 < 0 && y1 > 0)
    g.appendChild(sv('line', { x1: m.l, x2: w - m.r, y1: Y(0), y2: Y(0), stroke: 'var(--line2)' }));
  [0.08, 0.30, 0.45, 0.85].forEach(c => {
    if (c < x0 || c > x1) return;
    g.appendChild(tx('$' + c.toFixed(2), { x: X(c), y: h - 12, 'text-anchor': 'middle', class: 'ax' }));
  });

  const path = (key, col, dash) => {
    const d = curve.map((p2, i) => `${i ? 'L' : 'M'}${X(p2.cost).toFixed(1)} ${Y(p2[key]).toFixed(1)}`).join(' ');
    g.appendChild(sv('path', { d, fill: 'none', stroke: col, 'stroke-width': 2.2,
      'clip-path': `url(#${cid})`,
      'stroke-linecap': 'round', ...(dash ? { 'stroke-dasharray': dash } : {}) }));
  };
  path('blanket', 'var(--neg)', '5 4');
  path('targeted', 'var(--pos)');

  // break-even: above this price, blanket sending loses money
  if (breakeven >= x0 && breakeven <= x1) {
    g.appendChild(sv('line', { x1: X(breakeven), x2: X(breakeven), y1: m.t, y2: m.t + ih,
      stroke: 'var(--accent)', 'stroke-dasharray': '3 3' }));
    g.appendChild(tx('break-even', { x: X(breakeven) + 6, y: m.t + 11, class: 'ax', fill: 'var(--accent)' }));
  }
  // where the current assumptions put you
  const at = curve.reduce((a, b) => Math.abs(b.cost - cost) < Math.abs(a.cost - cost) ? b : a, curve[0]);
  [['blanket', 'var(--neg)'], ['targeted', 'var(--pos)']].forEach(([k, c]) =>
    g.appendChild(sv('circle', { cx: X(at.cost), cy: Y(at[k]), r: 4.5, fill: c })));
  g.appendChild(sv('line', { x1: X(at.cost), x2: X(at.cost), y1: m.t, y2: m.t + ih,
    stroke: 'var(--ink3)', 'stroke-width': 1 }));
  return g;
}

/* --------------------------------------------------------- scroll reveal */
let _io = null;

/** Fade each top-level block in as it scrolls into view, with a short stagger.
 *  Blocks already on screen animate immediately, so the first paint is not static. */
function revealAll(root) {
  if (_io) { _io.disconnect(); _io = null; }
  if (!('IntersectionObserver' in window) ||
      matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  _io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      e.target.classList.add('in');
      // cascade the children of this block
      const kids = e.target.querySelectorAll(
        '.groups>*, .ps2>*, .breakeven>*, .exp>*, .vs>*, .kpis>*, .deliv>*, tbody tr');
      kids.forEach((k, i) => k.style.setProperty('--cd', Math.min(i, 12) * 40 + 'ms'));
      _io.unobserve(e.target);
    });
  }, { rootMargin: '0px 0px -6% 0px', threshold: 0.04 });

  // The opening pass gets a wider, more deliberate cascade than a scroll does.
  const booting = document.body.classList.contains('boot');
  const gap = booting ? 95 : 60;
  const base = booting ? 260 : 0;      // let the masthead land first
  Array.from(root.children).forEach((child, i) => {
    child.classList.add('rv');
    child.style.setProperty('--d', base + Math.min(i, 5) * gap + 'ms');
    _io.observe(child);
  });
}

/* ------------------------------------------------------------------ icons */
/* Inline stroke icons. currentColor means each one takes the semantic colour of
   whatever it sits inside, so no per-icon colour rules are needed. */
const ICONS = {
  send: '<path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4Z"/>',
  pause: '<rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/>',
  skip: '<circle cx="12" cy="12" r="9"/><path d="M8 12h8"/>',
  ban: '<circle cx="12" cy="12" r="9"/><path d="M5.6 5.6l12.8 12.8"/>',
  up: '<path d="M3 17l6-6 4 4 8-8"/><path d="M17 7h4v4"/>',
  down: '<path d="M3 7l6 6 4-4 8 8"/><path d="M17 17h4v-4"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  download: '<path d="M12 3v12"/><path d="m7 11 5 5 5-5"/><path d="M4 20h16"/>',
  alert: '<path d="M12 3 2 20h20L12 3Z"/><path d="M12 10v4"/><path d="M12 17.5h.01"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1"/>',
  users: '<path d="M16 20v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="3.2"/><path d="M22 20v-2a4 4 0 0 0-3-3.8"/>',
  tag: '<path d="M20.6 13.4 12 22l-9-9V3h10l7.6 7.6a2 2 0 0 1 0 2.8Z"/><path d="M7.5 7.5h.01"/>',
};
const ico = (name, cls) =>
  `<svg class="ico ${cls || ''}" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ''}</svg>`;

/* An icon per decision, so the four actions are distinguishable without colour. */
const ACTION_ICON = { contact: 'send', no_offer: 'pause', not_worth: 'skip', suppress: 'ban' };

/* ------------------------------------------------------------------ bits */
function card(title, hint, body) {
  const c = el('div', 'card');
  if (title) c.appendChild(el('div', 'card-h', `<h3>${title}</h3><span class="hint">${hint || ''}</span>`));
  const b = el('div', 'card-b');
  if (body) b.appendChild(body);
  c.appendChild(b);
  return { card: c, body: b };
}

function barList(items, valueKey, fmt) {
  const wrap = el('div', 'bars');
  const mx = Math.max(...items.map(i => i[valueKey]), 1);
  items.forEach(i => wrap.appendChild(el('div', 'bar', `
    <div class="nm">${esc(i.label)}</div>
    <div class="tr"><i style="width:${(i[valueKey] / mx) * 100}%"></i></div>
    <div class="vv">${fmt(i)}</div>`)));
  return wrap;
}

function table(cols, rows, opts = {}) {
  if (!rows.length) return el('div', 'empty', opts.empty || 'Nothing to show');
  const w = el('div', 'tw');
  const t = el('table');
  t.appendChild(el('thead', null,
    '<tr>' + cols.map(c => `<th class="${c.right ? 'r' : ''}">${c.label}</th>`).join('') + '</tr>'));
  const tb = el('tbody');
  rows.forEach(r => {
    const tr = el('tr', opts.onRow ? 'click' : '');
    cols.forEach(c => tr.appendChild(el('td', (c.right ? 'r ' : '') + (c.strong ? 's' : ''), c.render(r))));
    if (opts.onRow) tr.onclick = () => opts.onRow(r);
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  w.appendChild(t);
  return w;
}

/** Ten signed bars: best-ranked customers on the left. */
function decileChart(values, h = 190) {
  const w = 600, m = { t: 16, r: 12, b: 28, l: 34 };
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const mx = Math.max(...values.map(Math.abs), 1) * 1.25;
  const Y = v => m.t + ih / 2 - (v / mx) * (ih / 2);
  const bw = (iw / values.length) * 0.58;
  const NS = 'http://www.w3.org/2000/svg';
  const sv = (t, a) => { const n = document.createElementNS(NS, t); for (const k in a) n.setAttribute(k, a[k]); return n; };
  const g = sv('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%', height: h, role: 'img',
                        'aria-label': 'Extra sales by customer group' });
  g.appendChild(sv('line', { x1: m.l, x2: w - m.r, y1: Y(0), y2: Y(0), stroke: 'var(--line2)' }));
  values.forEach((v, i) => {
    const cx = m.l + (iw / values.length) * (i + 0.5);
    const y = v >= 0 ? Y(v) : Y(0);
    g.appendChild(sv('rect', { x: cx - bw / 2, y, width: bw, height: Math.max(2, Math.abs(Y(v) - Y(0))),
                               rx: 2, fill: v >= 0 ? 'var(--pos)' : 'var(--neg)' }));
    const t = sv('text', { x: cx, y: h - 10, 'text-anchor': 'middle', class: 'ax' });
    t.textContent = i + 1; g.appendChild(t);
  });
  [mx, -mx].forEach(v => {
    const t = sv('text', { x: m.l - 7, y: Y(v) + 3.4, 'text-anchor': 'end', class: 'ax' });
    t.textContent = v.toFixed(0); g.appendChild(t);
  });
  return g;
}

/* ================================================================== TABS */
const TABS = {};

/** Turn a model's numbers into the sentence a marketing lead would say out loud.
 *  Generated from live figures, so it cannot drift from the table above it. */
function takeaway(m, d) {
  const X = m.decision, E = d.economics;
  const isBest = m.key === d.best.key;
  if (!m.is_uplift) {
    return { tone: 'bad2', short: 'Contacts almost everyone',
      line: `Contact ${int(X.contact_n)} of ${int(d.base.n_test)} — and lose ${money0(Math.abs(X.earns))}.`,
      body: `It puts nearly everyone on the list, because nearly everyone has some chance of visiting.
             At ${money(E.ranked_at)} a contact that is <b>${money0(Math.abs(X.vs_best))}</b> worse than
             the recommended list. This is the model most retail teams already run.` };
  }
  // At the comparison price every model on this dataset loses money. Saying
  // otherwise was hardcoded copy; read the number instead.
  const blanket = d.cost_curve[1].blanket;
  if (isBest) {
    const profitable = X.earns > 0;
    return { tone: profitable ? 'good2' : 'mid2',
      short: profitable ? 'The recommended list' : 'Best of five, still a loss',
      line: profitable
        ? `Contact ${int(X.contact_n)}, not ${int(d.base.n_test)}. Earns ${money0(X.earns)} where contacting everyone loses ${money0(Math.abs(blanket))}.`
        : `Contact ${int(X.contact_n)}, not ${int(d.base.n_test)}. Still loses ${money0(Math.abs(X.earns))} — but ${money0(Math.abs(blanket - X.earns))} less than contacting everyone.`,
      body: profitable
        ? `At ${money(E.ranked_at)} a contact, blanket sending loses money and this list does not.
           <b>Use this one.</b>`
        : `At ${money(E.ranked_at)} a contact nothing here makes money — this is the best of the five
           and it is still a loss. Targeting narrows the damage; it does not fix the price.
           <b>The honest answer at this cost is not to send it.</b>` };
  }
  return { tone: 'mid2', short: X.earns > 0 ? 'Works, but behind' : 'Behind the recommended list',
    line: `Contact ${int(X.contact_n)}. ${X.earns > 0 ? 'Earns' : 'Loses'} ${money0(Math.abs(X.earns))} — ${money0(Math.abs(X.vs_best))} behind the recommended list.`,
    body: `It finds real signal, but casts a wider net for less return at ${money(E.ranked_at)} a contact.` };
}

/* ============================== 1. THE DECISION — real data, real break-even */
TABS.models = function (v) {
  const d = S.data, P = d.problem, E = d.economics, best = d.models.find(m => m.key === d.best.key);

  // ---- the problem, on real numbers
  // ---- problem and answer, side by side and short
  const PR = d.project;
  // The first screen used to show no data at all. Lead with the answer, priced
  // from cost_curve so it can never drift from the model that produced it.
  const cheap = d.cost_curve[0], dear = d.cost_curve[1];
  const ab = el('div', 'answerbar');
  ab.innerHTML = `
    <div class="ab-k">The answer</div>
    <p class="ab-l">At <b>${money(cheap.cost)}</b> an email, contact all
      <b>${int(d.base.n)}</b> ${cheap.blanket > 0
        ? `\u2014 it pays <b class="up">${money0(cheap.blanket)}</b>.`
        : `\u2014 it still loses <b class="dn">${money0(Math.abs(cheap.blanket))}</b>.`}</p>
    <p class="ab-l">At <b>${money(dear.cost)}</b>, only <b>${int(dear.n_targeted)}</b> are worth it
      ${dear.targeted > 0
        ? `\u2014 worth <b class="up">${money0(dear.targeted)}</b>.`
        : `\u2014 and even that list loses <b class="dn">${money0(Math.abs(dear.targeted))}</b>.`}</p>
`;

  const intro = el('div');
  intro.style.marginBottom = '16px';
  intro.innerHTML = `
    <div class="mh">
      <div class="mh-left">
        <div class="mh-dots" title="Four models predict the change we cause. One does not.">
          <i></i><i></i><i></i><i></i><i></i>
        </div>
        <div>
          <div class="mh-k">AI project</div>
          <div class="mh-v">Five machine learning models
            <em>— four predict the change we cause, one does not</em></div>
        </div>
      </div>
      <div class="mh-right">
        <div class="mh-seal">${ico('check')}</div>
        <div>
          <div class="mh-k">Real customer data</div>
          <div class="mh-src">${esc(d.source.name)} &middot; ${int(d.base.n)} customers &middot;
            <a href="${esc(d.source.url)}" target="_blank" rel="noopener">source</a></div>
        </div>
      </div>
    </div>
    <div class="ps2">
      <div class="p2">
        <div class="h">The problem</div>
        <h3>${esc(PR.problem_title)}</h3>
        <p>${PR.problem_body}</p>
        <p class="tagline">${PR.problem_cost}</p>
      </div>
      <div class="s2">
        <div class="h">What this is</div>
        <h3>${esc(PR.solution_title)}</h3>
        <p>${PR.solution_body}</p>
        <div class="chips2">${PR.deliverables.map(([t, x], i) =>
          `<button data-go="${i}"><b>${esc(t)}</b> — ${esc(x)}</button>`).join('')}</div>
      </div>
    </div>`;
  v.appendChild(intro);

  // "Offer per customer" lives on tab 2; the other three are all decided in the
  // pricing block further down this tab.
  intro.querySelectorAll('.chips2 button').forEach(b => {
    b.onclick = () => {
      if (b.dataset.go === '2') { S.tab = 'products'; render(); return; }
      const t = document.getElementById('nb-decide');
      if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
  });

  v.appendChild(ab);

  const hero = el('div', 'hero');
  const top = el('div', 'hero-top', `
    <div class="lab">The experiment behind every number here</div>`);

  // The experiment itself, as three columns rather than a paragraph of numbers.
  top.appendChild(el('div', 'exp', `
    <div>
      <div class="who">Emailed</div>
      <div class="n">${int(E.n_emailed)} customers</div>
      <div class="r">${E.rate_emailed}%</div>
      <div class="cap">came back</div>
    </div>
    <div>
      <div class="who">Held back at random</div>
      <div class="n">${int(E.n_held_back)} customers</div>
      <div class="r">${E.rate_held_back}%</div>
      <div class="cap">came back</div>
    </div>
    <div class="diff">
      <div class="who">The campaign caused</div>
      <div class="n">the difference between them</div>
      <div class="r">${pts(E.lift_pp)} points</div>
      <div class="r2">${money(E.spend_lift)} per customer</div>
      <div class="cap">extra visits, and extra spend</div>
    </div>`));
  hero.appendChild(top);
  v.appendChild(hero);

  // ---- what the campaign was actually worth, per contact
  const be = el('div', 'breakeven');
  be.innerHTML = `
    <div>
      <div class="k">A visit is worth</div>
      <div class="v">${money(E.margin_per_visit)}</div>
      <div class="s">${money(E.rev_per_visit)} revenue × ${(E.margin_rate * 100).toFixed(0)}% margin</div>
    </div>
    <div class="key">
      <div class="k">So one contact is worth</div>
      <div class="v">${money(E.value_per_contact)}</div>
      <div class="s">this is your break-even</div>
    </div>
    <div>
      <div class="k">Base size</div>
      <div class="v">${int(d.base.n)}</div>
      <div class="s">real customers</div>
    </div>`;
  v.appendChild(be);

  v.appendChild(el('div', 'rulefoot', `
    <span class="k">The rule</span>
    <p>Any way of reaching a customer that costs more than
      <b>${money(E.value_per_contact)}</b> loses money if you send it to everyone.
      <span>That single number decides whether you need a model at all. Below it, contact everyone.
      Above it, targeting is the only thing that keeps the campaign profitable.</span></p>`));

  // ---- the assumptions ARE the interface: change them, everything re-prices
  const st = { margin: E.margin_rate, cost: d.priced_at.bakeoff };
  const METHODS = d.cost_curve.map(r => ({ label: r.label, cost: r.cost }));

  const ctlHost = el('div');
  const outHost = el('div');
  const cc = card('What should we do?',
                  'change either assumption and every number below re-prices', ctlHost);
  cc.card.id = 'nb-decide';
  cc.body.appendChild(outHost);
  cc.card.style.marginBottom = '16px';
  v.appendChild(cc.card);

  const ctrls = el('div', 'ctrls');
  const mk = (label, key, min, max, step, fmt, ends) => {
    const wrap = el('div', 'ctrl2', `<div class="top"><span class="lb">${label}</span>
      <span class="val">${fmt(st[key])}</span></div>`);
    const r = document.createElement('input');
    r.type = 'range'; r.min = min; r.max = max; r.step = step; r.value = st[key];
    r.setAttribute('aria-label', label);
    r.oninput = () => {
      st[key] = +r.value;
      wrap.querySelector('.val').textContent = fmt(+r.value);
      if (key === 'cost') markPreset();
      schedule();
    };
    wrap.appendChild(r);
    wrap.appendChild(el('div', 'ends', `<span>${ends[0]}</span><span>${ends[1]}</span>`));
    return wrap;
  };
  const costW = mk('Cost to reach one customer', 'cost', 0.02, 1.00, 0.01,
                   v2 => money(v2), ['$0.02', '$1.00']);
  const presets = el('div', 'presets2');
  METHODS.forEach(mth => {
    const b = el('button', '', `${esc(mth.label)} · ${money(mth.cost)}`);
    b.dataset.cost = mth.cost;
    b.onclick = () => {
      st.cost = mth.cost;
      costW.querySelector('input').value = mth.cost;
      costW.querySelector('.val').textContent = money(mth.cost);
      markPreset(); run();
    };
    presets.appendChild(b);
  });
  costW.appendChild(presets);
  const markPreset = () => presets.querySelectorAll('button').forEach(b =>
    b.classList.toggle('on', Math.abs(+b.dataset.cost - st.cost) < 0.005));
  markPreset();

  ctrls.append(costW, mk('Margin you keep on an order', 'margin', 0.05, 0.90, 0.01,
                         v2 => Math.round(v2 * 100) + '%', ['5%', '90%']));
  ctlHost.appendChild(ctrls);

  let timer = null, seq = 0;
  const schedule = () => { clearTimeout(timer); timer = setTimeout(run, 140); };

  async function run() {
    const mine = ++seq;
    let r;
    try { r = await post('/api/recompute', st); }
    catch (e) {
      if (mine !== seq) return;
      outHost.innerHTML = `<div class="warn"><p>Could not re-price. ${esc(e.message)}</p></div>`;
      return;
    }
    if (mine !== seq) return;
    outHost.innerHTML = '';

    outHost.appendChild(el('div', 'verdictbar ' + r.verdict, `
      <div>
        <div class="big3">${r.verdict === 'target' ? ico('target') + 'Target a list'
          : r.verdict === 'blanket' ? ico('users') + 'Contact everyone' : ico('ban') + 'Do not run it'}</div>
        <p>${esc(r.advice)}</p>
      </div>
      <div class="num3">
        <b>${money(r.breakeven)}</b>break-even per contact
      </div>
      <div class="num3">
        <b>${r.n_targeted ? int(r.n_targeted) : '—'}</b>of ${int(r.n_base)} customers
      </div>`));

    outHost.appendChild(costChart(r.curve, r.cost, r.breakeven));
    outHost.appendChild(el('div', 'legend', `
      <span><i style="background:var(--neg)"></i>Send to everyone</span>
      <span><i style="background:var(--pos)"></i>Send to a targeted list</span>
      <span><i style="background:var(--accent)"></i>Break-even</span>`));
    outHost.lastChild.style.margin = '12px 0 16px';

    outHost.appendChild(el('div', 'vs', `
      <div class="${r.blanket < 0 ? 'loss' : 'win'}">
        <div class="k">Send to everyone</div><div class="v">${money0(r.blanket)}</div>
        <div class="s">${r.blanket < 0 ? 'a loss at this price' : 'net profit'}</div>
      </div>
      <div class="${r.targeted < 0 ? 'loss' : 'win'}">
        <div class="k">Send to ${r.n_targeted ? int(r.n_targeted) : 'nobody'}</div>
        <div class="v">${money0(r.targeted)}</div>
        <div class="s">${r.share_targeted}% of the base${r.targeted < 0 ? ' — still a loss' : ''}</div>
      </div>
      <div class="${r.gain > 0 && !r.gain_is_avoided_loss ? 'gain' : ''}">
        <div class="k">${r.gain_is_avoided_loss ? 'Targeting avoids' : 'Targeting is worth'}</div>
        <div class="v">${r.gain > 0 ? '+' + money0(r.gain) : '—'}</div>
        <div class="s">${r.gain <= 0 ? 'nothing at this price'
          : r.gain_is_avoided_loss ? 'of the loss — it does not make this profitable'
          : 'versus sending to everyone'}</div>
      </div>`));

    const dl = el('div', 'dl');
    dl.style.marginTop = '16px';
    const mkBtn = (label, act, cls) => {
      const b = el('button', cls, ico('download') + label);
      b.onclick = () => {
        if (NB) return staticExport(act, st.cost, st.margin);
        window.location = `/api/export?action=${act}&cost=${st.cost}&margin=${st.margin}`;
      };
      return b;
    };
    dl.append(mkBtn('Download target list', 'contact', ''),
              mkBtn('Download suppression list', 'suppress', 'ghost2'),
              mkBtn('Download everyone', 'all', 'ghost2'),
              el('span', 'hintx', 'CSV, priced at the assumptions above — they are in the filename'));
    outHost.appendChild(dl);
  }
  run();

  // ---- what is being predicted
  v.appendChild(el('div', 'oneline', `
    <span class="k">We predict</span>
    <p>${d.predicts}
      <span>Not “will they come back” — that is already in your reports. This is “will they come back
      <i>because of us</i>”, which is the only part you can act on.</span></p>`));

  // ---- the five models, compared where the choice matters
  const cmp = table([
    { label: '#', render: r => String(d.models.indexOf(r) + 1) },
    { label: 'Model', strong: true, render: r => `${esc(r.name)}<span class="sumline">${esc(takeaway(r, d).short)}</span>` },
    { label: 'What it predicts', render: r => r.is_uplift ? 'the change we cause'
        : '<span class="dn">who will visit — wrong thing</span>' },
    { label: 'Qini', right: true, render: r => Math.round(r.qini) },
    { label: 'Contacts', right: true, render: r => int(r.decision.contact_n) },
    { label: `Net at ${money(d.economics.ranked_at)}`, right: true, render: r => `<span class="${dir(r.decision.earns)}">${money0(r.decision.earns)}</span>` },
    { label: 'Verdict', render: r => `<span class="badge ${r.verdict}">${esc(r.verdict_text)}</span>` },
    { label: '', render: () => '<span class="chev">▶</span>' },
  ], d.models);

  const wrap = el('div', 'tw');
  wrap.appendChild(cmp.querySelector('table') || cmp);
  const tbody = wrap.querySelector('tbody');
  Array.from(tbody.rows).forEach((row, i) => {
    const m = d.models[i], X = m.decision, T = takeaway(m, d);
    row.classList.add('click');
    const det = el('tr', 'detrow');
    det.hidden = true;
    const cell = el('td');
    cell.colSpan = 8;
    cell.innerHTML = `
      <div class="take ${T.tone}">
        <div class="k">What this means for marketing</div>
        <div class="line">${T.line}</div>
        <p>${T.body}</p>
      </div>
      <div class="forwho">For the data team</div>
      <div class="det-cols">
        <div><div class="k">How it works</div><p>${m.how}</p></div>
        <div class="gk"><div class="k">Good</div><p>${esc(m.good)}</p></div>
        <div class="bk"><div class="k">Careful</div><p>${esc(m.careful)}</p></div>
      </div>
      <div class="det-cols" style="grid-template-columns:1fr">
        <div><div class="k">Where it fits</div><p>${esc(m.business)}</p></div>
      </div>`;
    const box = el('div');
    box.style.cssText = 'background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px';
    const k = el('div', null, 'Extra visits, best customers to worst');
    k.style.cssText = 'font-family:"JetBrains Mono",monospace;font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink3);margin-bottom:8px';
    box.append(k, decileChart(m.deciles, 150));
    const cap = el('p', null, m.is_uplift
      ? 'Ten groups of real customers, best on the left. Measured against the held-back group, not predicted.'
      : '<span class="dn">Nearly flat.</span> Ranking by "who will visit" barely separates who was persuaded.');
    cap.style.cssText = 'color:var(--ink3);font-size:12.5px;margin:9px 0 0';
    box.appendChild(cap);
    cell.appendChild(box);
    det.appendChild(cell);
    row.tabIndex = 0;
    row.setAttribute('role', 'button');
    row.setAttribute('aria-expanded', 'false');
    row.setAttribute('aria-label', `${m.name} — show detail`);
    const toggle = () => {
      const open = !det.hidden;
      det.hidden = open;
      row.classList.toggle('open', !open);
      row.setAttribute('aria-expanded', String(!open));
    };
    row.onclick = toggle;
    row.onkeydown = (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
    };
    row.after(det);
  });

  const mc = card(`Five ways to predict it`,
                  `compared at ${money(E.ranked_at)} a contact, where the choice actually matters`, wrap);
  mc.body.style.padding = '0';
  mc.body.firstChild.style.border = 'none';
  mc.card.style.marginBottom = '16px';
  v.appendChild(mc.card);

  if (d.honesty && d.honesty.ci_overlap) {
    const hw = el('div', 'warn');
    hw.style.marginBottom = '16px';
    hw.innerHTML = `<p><b>Read this before quoting any number above.</b> ${d.honesty.verdict}</p>`;
    v.appendChild(hw);
  }

  v.appendChild(el('div', 'note', `<b>No accuracy column, and that is the point.</b> On real customers
    there is no per-person truth to score against — you see what someone did after being emailed, never
    what they would have done otherwise. So each model picks a list, and that list is priced against what
    those exact customers really did. ${esc(best.name)} wins on money, not on a statistic.`));

  // ---- folded detail
  const src = el('details', 'more');
  src.innerHTML = `<summary>Where this data comes from</summary>
    <div class="inner" style="color:var(--ink2);font-size:14px">
      <p><b style="color:var(--ink)">${esc(d.source.name)}</b> — ${esc(d.source.detail)}</p>
      <p>Because the split was random, the gap between the emailed and held-back groups is a real
      causal effect. Every figure on these three tabs comes from those customers. Nothing is simulated.</p>
      <p style="margin:0">Modelled outcome: a store or site visit in the two weeks after the send
      (${d.base.control_rate}% held back vs ${d.base.treated_rate}% emailed). Revenue per visit and
      order value are observed; gross margin (${(E.margin_rate * 100).toFixed(0)}%) and the cost of
      each contact method are the two assumptions, and both are shown on screen so you can change them.</p>
    </div>`;
  v.appendChild(src);

  const gl = el('details', 'more');
  gl.innerHTML = '<summary>What the words mean</summary>';
  const gi = el('div', 'inner');
  const dl = el('dl', 'gl');
  d.glossary.forEach(([t, def]) => dl.appendChild(el('div', null, `<dt>${esc(t)}</dt><dd>${esc(def)}</dd>`)));
  gi.appendChild(dl);
  gl.appendChild(gi);
  v.appendChild(gl);
};

/* ================================= 2. PRODUCTS & OFFERS — the business answer */
TABS.products = function (v) {
  const d = S.data, F = d.product_focus;
  const worth = d.answer.groups.find(g => g.key === 'contact');

  v.appendChild(el('div', 'head', `
    <h2>Which products sell, which do not, and what fixes them</h2>
    <p>The <b>${esc(d.best.name)}</b> scored every customer. Rolled up by product, it tells you three
    things a campaign report cannot: which lines are already selling without help, which are weak,
    and for the weak ones \u2014 what offer to use and which customers to send it to.</p>`));

  // Every tab states which model produced its numbers, so no figure on any page
  // is unattributed.
  v.appendChild(el('div', 'attrib', `
    <span><b>Extra visits</b> predicted by the <b>${esc(F.scored_by)}</b> — the model that earned most on tab 1</span>
    <span><b>Which email</b> chosen by ${esc(F.offer_scored_by)}, each trained against the same held-back group</span>
    <span>Priced at <b>${money(d.priced_at.customers)}</b> a send — the campaign as it ran</span>`));

  const figs = r => `
    <div class="figs">
      <div><div class="k">Sales 12m</div><div class="v">${money0(r.spend)}</div></div>
      <div><div class="k">Buyers</div><div class="v">${int(r.n)}</div></div>
      <div><div class="k">Extra visits</div><div class="v ${dir(r.extra_sales_pp)}">${pts(r.extra_sales_pp)}</div></div>
    </div>`;

  const pf = el('div', 'pf');
  pf.appendChild(el('div', 'pcard win', `
    <div class="lab">Best selling</div>
    <h3>${esc(F.best.product)}</h3>
    <div class="sub">${esc(F.best.category)} &middot; ${money0(F.best.avg_order)} average order</div>
    ${figs(F.best)}
    <div class="fix none">
      <div class="t">No offer needed</div>
      <p>It already sells. Only <b>${int(F.best.contact_n)}</b> of its ${int(F.best.n)} buyers are worth
      contacting \u2014 discounting the rest cuts margin on a product that needs no help.</p>
    </div>`));

  const W = F.worst;
  pf.appendChild(el('div', 'pcard weak', `
    <div class="lab">Least selling</div>
    <h3>${esc(W.product)}</h3>
    <div class="sub">${esc(W.category)} &middot; ${money0(W.avg_order)} average order</div>
    ${figs(W)}
    ${W.fixable
      ? `<div class="fix">
           <div class="t">Give them: ${esc(W.reward_label)}</div>
           <p>${esc(W.reward_why)}</p>
           <p style="margin-top:9px"><b style="color:var(--ink)">Send it to: ${esc(W.audience)}</b><br>
             <span style="font-size:12.5px">${esc(W.audience_detail)}</span></p>
           <span class="n">${int(W.contact_n)} of ${int(W.n)} buyers &nbsp;\u2192&nbsp;
             about ${int(W.extra_sales)} extra sales, ${money0(W.gain)} profit</span>
         </div>`
      : `<div class="fix none">
           <div class="t">No offer will fix this</div>
           <p>Its buyers barely move when contacted \u2014 a range or pricing problem, not a marketing one.</p>
         </div>`}`));
  v.appendChild(pf);

  v.appendChild(el('div', 'note', `<b>One caution on the ranking.</b> ${esc(F.note)}`));

  const weakT = table([
    { label: 'Least selling product', strong: true, render: r => esc(r.product) },
    { label: 'Sales 12m', right: true, render: r => money0(r.spend) },
    { label: 'Extra visits', right: true, render: r => `<span class="${dir(r.extra_sales_pp)}">${pts(r.extra_sales_pp)}</span>` },
    { label: 'Give them', render: r => r.fixable
        ? `<b style="color:var(--ink)">${esc(r.reward_label)}</b>`
        : '<span style="color:var(--ink3)">nothing \u2014 range problem</span>' },
    { label: 'Send it to these customers', render: r => esc(r.audience) },
    { label: 'How many', right: true, render: r => int(r.contact_n) },
    { label: 'Worth', right: true, render: r => r.fixable ? `<span class="${dir(r.gain)}">${money0(r.gain)}</span>` : '\u2014' },
  ], F.weak5);
  const wc = card('The five weakest products \u2014 offer, and who to send it to',
                  `${F.fixable_n} of 5 can be moved by an offer`, weakT);
  wc.body.style.padding = '0';
  wc.body.firstChild.style.border = 'none';
  wc.card.style.marginBottom = '16px';
  v.appendChild(wc.card);

  const bestT = table([
    { label: 'Best selling product', strong: true, render: r => esc(r.product) },
    { label: 'Category', render: r => esc(r.category) },
    { label: 'Sales 12m', right: true, render: r => money0(r.spend) },
    { label: 'Buyers', right: true, render: r => int(r.n) },
    { label: 'Extra visits', right: true, render: r => `<span class="${dir(r.extra_sales_pp)}">${pts(r.extra_sales_pp)}</span>` },
    { label: 'Worth contacting', right: true, render: r => `${int(r.contact_n)} of ${int(r.n)}` },
  ], F.top5);
  const bc = card('The five best selling products', 'most of these need no offer at all', bestT);
  bc.body.style.padding = '0';
  bc.body.firstChild.style.border = 'none';
  bc.card.style.marginBottom = '16px';
  v.appendChild(bc.card);

  const oppT = table([
    { label: 'Product', strong: true, render: r => esc(r.product) },
    { label: 'Give them', render: r => esc(r.reward_label) },
    { label: 'Send it to', render: r => esc(r.audience) },
    { label: 'How many', right: true, render: r => `${int(r.contact_n)} of ${int(r.n)}` },
    { label: 'Profit per 100 buyers', right: true, render: r => `<span class="${dir(r.gain_per_100)}">${money(r.gain_per_100)}</span>` },
    { label: 'Total worth', right: true, render: r => money0(r.gain) },
  ], F.opportunity);
  const oc = card('Where an offer pays back most', 'ranked by profit per 100 buyers, not by size', oppT);
  oc.body.style.padding = '0';
  oc.body.firstChild.style.border = 'none';
  oc.card.style.marginBottom = '16px';
  v.appendChild(oc.card);

  // ---- the three rewards across the whole base
  const rh = el('div', 'head', `
    <h2>The ${d.rewards.length === 2 ? 'two' : d.rewards.length} offers, and who gets each</h2>
    <p>Across the whole base, <b>${int(worth.n)}</b> customers are worth contacting. Pushing a discount
    at someone who does not need one only costs margin, so they do not all get the same thing.</p>`);
  rh.style.marginTop = '28px';
  v.appendChild(rh);

  const g = el('div', 'g2');
  d.rewards.forEach(r => g.appendChild(el('div', 'rw', `
    <div class="n">${int(r.n)}</div>
    <h3>${esc(r.label)}</h3>
    <p>${esc(r.why)}</p>
    <div class="foot">Expected extra profit <b>${money0(r.value)}</b>
      &nbsp;&middot;&nbsp; about <b>${int(r.extra_sales)}</b> extra sales</div>`)));
  v.appendChild(g);

  // The men's arm lifts more overall, so an unconstrained model hands the men's
  // email to most womenswear buyers. Show whether the split is defensible.
  const OM = d.offer_match;
  if (OM) {
    const lopsided = OM.share < 75;
    const note = el('div', lopsided ? 'warn' : 'oneline');
    note.style.marginTop = '14px';
    note.innerHTML = `
      <span class="k">${lopsided ? 'Check this split' : 'Offer matches merchandise'}</span>
      <p><b>${OM.share}%</b> of the ${int(OM.total)} customers on the list get the campaign for the
      merchandise they actually buy. ${esc(OM.note)}
      ${lopsided ? '<b>Below 75% the split needs a merchandising review before it ships.</b>' : ''}</p>`;
    v.appendChild(note);
  }
};

/* ============================================ 3. CUSTOMERS — the detail, last */
TABS.customers = function (v) {
  const d = S.data, A = d.answer, B = d.base;

  v.appendChild(el('div', 'head', `
    <h2>Your customers</h2>
    <p>The detail behind everything above: who they are, what they buy, and what to do about each one.
    Click any row for the full record.</p>`));

  v.appendChild(el('div', 'attrib', `
    <span><b>Extra visits</b> and <b>Worth</b> predicted by the <b>${esc(d.best.name)}</b></span>
    <span>Priced at <b>${money(d.priced_at.customers)}</b> a send</span>
    <span>Only fields this dataset genuinely records are shown</span>`));

  const kpis = el('dl', 'kpis');
  kpis.innerHTML = `
    <div><dt>Customers</dt><dd>${int(B.n)}</dd></div>
    <div><dt>Sales (12 months)</dt><dd>${money0(B.revenue)}</dd></div>
    <div><dt>Average order</dt><dd>${money0(B.avg_order)}</dd></div>
    <div><dt>Came back without contact</dt><dd>${B.control_rate}%</dd></div>
    <div><dt>Came back after contact</dt><dd class="up">${B.treated_rate}%</dd></div>
    <div><dt>Actually bought</dt><dd>${B.conversion_rate}%</dd></div>`;
  kpis.style.marginBottom = '16px';
  v.appendChild(kpis);

  const groups = el('div', 'groups');
  groups.style.marginBottom = '16px';
  A.groups.forEach(g => {
    const b = el('button', 'grp ' + g.tone, `
      <div class="ghead">${ico(ACTION_ICON[g.key])}<div class="n">${int(g.n)}</div></div>
      <div class="l">${esc(g.label)} <span style="color:var(--ink3);font-weight:400">${g.share}%</span></div>
      <div class="w">${esc(g.why)}</div>
      <span class="go">Filter to these \u2192</span>`);
    b.onclick = () => {
      // render() ends with scrollTo(top:0). Calling it here filtered correctly
      // but threw the user to the top of the page, away from the table that had
      // just changed -- so a working filter looked broken. Update in place.
      S.cf = { ...S.cf, action: g.key, offset: 0 };
      const first = f.querySelector('select');
      if (first) first.value = g.key;
      groups.querySelectorAll('.grp').forEach(x => x.classList.remove('picked'));
      b.classList.add('picked');
      b.setAttribute('aria-pressed', 'true');
      load();
      host.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
    groups.appendChild(b);
  });
  v.appendChild(groups);

  const cats = card('What they buy', 'share of your customer base',
    barList(d.categories, 'n', i => `${int(i.n)} \u00b7 ${money0(i.spend)}`));
  const chans = card('How they buy', 'where the order is placed',
    barList(d.channels, 'n', i => `${int(i.n)} \u00b7 ${i.share}%`));
  const g2 = el('div', 'g2');
  g2.style.marginBottom = '16px';
  g2.append(cats.card, chans.card);
  v.appendChild(g2);

  const f = el('div', 'filters');
  const sel = (label, key, opts) => {
    const w = el('div', 'f', `<label>${label}</label>`);
    const sl = document.createElement('select');
    opts.forEach(o => { const op = document.createElement('option'); op.value = o.v; op.textContent = o.l; sl.appendChild(op); });
    sl.value = S.cf[key];
    sl.onchange = () => { S.cf[key] = sl.value; S.cf.offset = 0; load(); };
    w.appendChild(sl);
    return w;
  };
  f.append(
    sel('What to do', 'action', [{ v: 'all', l: 'All customers' },
      ...A.groups.map(g => ({ v: g.key, l: g.label }))]),
    sel('Category', 'category', [{ v: 'all', l: 'All categories' },
      ...d.categories.map(c => ({ v: c.key, l: c.label }))]),
    sel('Channel', 'channel', [{ v: 'all', l: 'All channels' },
      ...d.channels.map(c => ({ v: c.key, l: c.label }))]),
    sel('Sort by', 'sort', [
      { v: 'value_of_contact', l: 'Value of contacting' },
      { v: 'extra_sales_pp', l: 'Extra visits' },
      { v: 'spend_12m', l: 'Spent in past year' },
      { v: 'months_since_purchase', l: 'Most recent first' },
    ]));
  const sw = el('div', 'f', '<label>Search</label>');
  const inp = document.createElement('input');
  inp.type = 'text'; inp.placeholder = 'Customer ID or spend band'; inp.value = S.cf.q;
  inp.oninput = () => { S.cf.q = inp.value.trim(); S.cf.offset = 0; clearTimeout(inp._t); inp._t = setTimeout(load, 250); };
  sw.appendChild(inp);
  f.appendChild(sw);
  const fc = card('Filters', '', f);
  fc.card.style.marginBottom = '16px';
  v.appendChild(fc.card);

  const host = el('div');
  v.appendChild(host);

  async function load() {
    const qs = new URLSearchParams({ sort: S.cf.sort, limit: S.cf.limit, offset: S.cf.offset });
    ['action', 'category', 'channel', 'reward'].forEach(k => { if (S.cf[k] !== 'all') qs.set(k, S.cf[k]); });
    if (S.cf.q) qs.set('q', S.cf.q);

    host.innerHTML = '<div class="sk" style="height:420px"></div>';
    let r;
    try { r = await api('/api/customers?' + qs); }
    catch (e) { host.innerHTML = `<div class="warn"><p>Could not load customers. ${esc(e.message)}</p></div>`; return; }

    host.innerHTML = '';
    const t = table([
      { label: 'Customer', strong: true, render: x => '#' + x.customer_id },
      { label: 'Buys', render: x => esc(x.buys) },
      { label: 'Spent (12m)', right: true, render: x => money(x.spend_12m) },
      { label: 'Band', render: x => esc(x.spend_band) },
      { label: 'Last bought', right: true, render: x => int(x.months_since_purchase) + ' mo' },
      { label: 'Area', render: x => esc(x.area) },
      { label: 'Channel', render: x => esc(x.channel) },
      { label: 'New?', render: x => x.new_customer ? 'yes' : '—' },
      { label: 'Extra visits', right: true, render: x => `<span class="${dir(x.extra_sales_pp)}">${pts(x.extra_sales_pp)}</span>` },
      { label: 'Worth', right: true, render: x => `<span class="${dir(x.value_of_contact)}">${money(x.value_of_contact)}</span>` },
      // Constant by definition once a group filter is on, so it is dropped there.
      ...(S.cf.action === 'all' ? [{ label: 'What to do',
        render: x => `<span class="pill ${x.action}">${ico(ACTION_ICON[x.action])}${esc(actionLabel(x.action))}</span>` }] : []),
    ], r.rows, { onRow: x => openCustomer(x.customer_id), empty: 'No customers match these filters' });

    const dl2 = el('div', 'dl');
    const exp = el('button', '', ico('download') + 'Download these as CSV');
    exp.onclick = () => {
      const act = S.cf.action === 'all' ? 'all' : S.cf.action;
      if (NB) return staticExport(act, S.data.priced_at.customers,
                                  S.data.economics.margin_rate, S.cf);
      const qs = new URLSearchParams({ action: act });
      ['category', 'channel', 'reward'].forEach(k => {
        if (S.cf[k] && S.cf[k] !== 'all') qs.set(k, S.cf[k]);
      });
      if (S.cf.q) qs.set('q', S.cf.q);
      window.location = `/api/export?${qs}`;
    };
    dl2.append(exp, el('span', 'hintx', `${int(r.total)} rows`));

    const c = card('Customer list', `${int(r.total)} match your filters`, t);
    c.body.style.padding = '0';
    if (r.rows.length) c.body.firstChild.style.border = 'none';

    const foot = el('div');
    foot.style.cssText = 'display:flex;justify-content:space-between;align-items:center;padding:13px 18px;border-top:1px solid var(--line);font-size:13px;color:var(--ink3)';
    const from = r.total ? r.offset + 1 : 0;
    foot.appendChild(el('span', 'num', `${int(from)}\u2013${int(Math.min(r.offset + r.limit, r.total))} of ${int(r.total)}`));
    const nav = el('div');
    nav.style.cssText = 'display:flex;gap:8px';
    const prev = el('button', 'btn', 'Previous'), next = el('button', 'btn', 'Next');
    prev.disabled = r.offset === 0;
    next.disabled = r.offset + r.limit >= r.total;
    prev.onclick = () => { S.cf.offset = Math.max(0, S.cf.offset - S.cf.limit); load(); };
    next.onclick = () => { S.cf.offset += S.cf.limit; load(); };
    nav.append(prev, next);
    foot.appendChild(nav);
    c.body.appendChild(foot);
    const dlWrap = el('div');
    dlWrap.style.cssText = 'padding:14px 18px;border-top:1px solid var(--line)';
    dlWrap.appendChild(dl2);
    c.body.appendChild(dlWrap);
    host.appendChild(c.card);
  }
  load();
};

const actionLabel = key => {
  const g = S.data.answer.groups.find(x => x.key === key);
  return g ? g.label : key;
};

/* ================================== 4. THE ML — formal reference, data team */
TABS.reference = function (v) {
  const d = S.data, R = d.reference, E2 = d.economics;

  v.appendChild(el('div', 'head', `
    <h2>How the models actually work</h2>
    <p>The first three tabs are written for whoever runs the campaign. This one is for whoever has
    to build, defend or maintain the models — definitions, notation, the assumptions everything
    rests on, and the papers each estimator comes from.</p>`));

  if (NB) {
    v.appendChild(el('div', 'note', `<b>This is the hosted, read-only build.</b> Everything on the
      first three tabs is live \u2014 the sliders re-price against the real held-back group and the
      CSV exports are generated in your browser. The single-customer scorer needs the fitted models,
      so it is not included here; run the project locally to use it.`));
  }

  v.appendChild(el('div', 'attrib', `
    <span>Recommended on this data: <b>${esc(d.best.name)}</b></span>
    <span>Compared at <b>${money(d.priced_at.bakeoff)}</b> a contact — see tab 1</span>`));

  // ---- the estimation problem
  const setup = el('div');
  setup.innerHTML = `<p style="color:var(--ink2);font-size:14.5px">${R.setup.body}</p>`;
  R.setup.eq.forEach(([nm, eq, note]) => setup.appendChild(el('div', 'eqrow',
    `<div class="nm">${esc(nm)}<small>${esc(note)}</small></div><div class="eq">${esc(eq)}</div>`)));
  const c1 = card(R.setup.title, 'potential outcomes framework', setup);
  c1.card.style.marginBottom = '16px';
  v.appendChild(c1.card);

  // ---- identifying assumptions
  const as = el('div', 'assume');
  R.assumptions.items.forEach(([t, f, p2]) => as.appendChild(el('div', null,
    `<div class="t">${esc(t)}</div><div class="f">${esc(f)}</div><p>${esc(p2)}</p>`)));
  const c2 = card(R.assumptions.title, 'all three hold here by design of the experiment', as);
  c2.card.style.marginBottom = '16px';
  v.appendChild(c2.card);

  // ---- each estimator, formally
  const box = el('div');
  d.models.forEach((m, i) => {
    const L = R.learners[m.key];
    // Same three states as tab 1, same colours: recommended, not-causal, or behind.
    const state = m.key === d.best.key ? 'v-best' : !m.is_uplift ? 'v-avoid' : 'v-mid';
    const w = el('div', 'lrn ' + state);
    w.innerHTML = `
      <div class="hd2">
        <div class="t"><span class="rank">${i + 1}</span>${esc(L.formal)}</div>
        ${L.cite ? `<div class="c">${esc(L.cite)}</div>` : '<div class="c">baseline, not causal</div>'}
      </div>
      <div class="lmetrics">
        <span><b>${m.qini.toFixed(1)}</b>Qini</span>
        <span><b>${pts(m.top)}pp</b>top decile</span>
        <span><b class="${dir(m.decision.earns)}">${money0(m.decision.earns)}</b>net at ${money(E2.ranked_at)}</span>
        <span class="vd">${esc(m.verdict_text)}</span>
      </div>
      <div class="bd2">
        <div class="eq">${esc(L.eq)}</div>
        <p>${esc(L.detail)}</p>
      </div>`;
    box.appendChild(w);
  });
  const c3 = card('The five estimators',
                  `ranked by what their decisions earned at ${money(E2.ranked_at)} a contact \u2014 the same order as tab 1`,
                  box);
  c3.card.style.marginBottom = '16px';
  v.appendChild(c3.card);

  // ---- evaluation
  const ev = el('div');
  ev.innerHTML = `<p style="color:var(--ink2);font-size:14.5px">${R.evaluation.body}</p>`;
  R.evaluation.eq.forEach(([nm, eq, note]) => ev.appendChild(el('div', 'eqrow',
    `<div class="nm">${esc(nm)}<small>${esc(note)}</small></div><div class="eq">${esc(eq)}</div>`)));
  const bad = el('div');
  bad.innerHTML = `<div class="lab" style="margin-top:18px">Not valid here</div>
    <div class="novalid">${R.evaluation.invalid.map(x => `<span>${esc(x)}</span>`).join('')}</div>
    <p style="margin:0;font-size:13.5px;color:var(--ink2)">${esc(R.evaluation.invalid_why)}</p>`;
  ev.appendChild(bad);
  const c4 = card(R.evaluation.title, '', ev);
  c4.card.style.marginBottom = '16px';
  v.appendChild(c4.card);

  // ---- implementation
  const st = table([
    { label: 'Component', strong: true, render: r => esc(r[0]) },
    { label: 'Choice', render: r => `<span class="mono" style="font-size:12.5px">${esc(r[1])}</span>` },
    { label: 'Why', render: r => `<span style="white-space:normal;display:block">${esc(r[2])}</span>` },
  ], R.stack);
  const c5 = card('Implementation', 'what is actually running', st);
  c5.body.style.padding = '0';
  c5.body.firstChild.style.border = 'none';
  v.appendChild(c5.card);
};

/* ---------------------------------------------------------------- drawer */
async function openCustomer(id) {
  const dr = $('#drawer'), sc = $('#scrim');
  dr.classList.add('open'); sc.classList.add('open');
  dr.innerHTML = `<div class="dh"><h3>Customer #${id}</h3></div>
    <div class="db"><div class="sk" style="height:170px"></div><div class="sk" style="height:230px"></div></div>`;

  let d;
  try { d = await api('/api/customer/' + id); }
  catch (e) { dr.innerHTML = `<div class="db"><div class="warn"><p>Could not load this customer. ${esc(e.message)}</p></div></div>`; return; }

  const K = d.known, H = d.happened, P = d.prediction, D = d.decision;
  dr.innerHTML = '';
  const head = el('div', 'dh', `<div>
      <h3 style="font-size:18px">Customer #${d.customer_id}</h3>
      <div style="display:flex;gap:6px;margin-top:7px;flex-wrap:wrap">
        <span class="pill ${D.action}">${ico(ACTION_ICON[D.action])}${esc(D.label)}</span>
        <span class="tag">${esc(K.buys)}</span><span class="tag">${esc(K.channel)}</span>
      </div></div>`);
  const x = el('button', 'iconbtn', 'Close');
  x.onclick = closeDrawer;
  head.appendChild(x);

  const body = el('div', 'db');

  // 1 -- only what this dataset genuinely records
  const known = el('div', null,
    '<div class="step"><i>1</i><div class="lab">What we know about them</div></div>');
  const kv = el('div', 'kv');
  kv.innerHTML = `
    <div><dt>Last bought</dt><dd>${K.months_since_purchase} months ago</dd></div>
    <div><dt>Spent (12m)</dt><dd>${money(K.spend_12m)}</dd></div>
    <div><dt>Spend band</dt><dd style="font-size:12px">${esc(K.spend_band)}</dd></div>
    <div><dt>Buys</dt><dd style="font-size:12px">${esc(K.buys)}</dd></div>
    <div><dt>Area</dt><dd style="font-size:12.5px">${esc(K.area)}</dd></div>
    <div><dt>Channel</dt><dd style="font-size:12.5px">${esc(K.channel)}</dd></div>
    <div><dt>New customer</dt><dd>${K.new_customer ? 'Yes' : 'No'}</dd></div>`;
  known.appendChild(kv);
  known.appendChild(el('div', 'note', `These are the only seven things this dataset records about a
    customer. Nothing here is invented or inferred.`));
  known.lastChild.style.marginTop = '11px';
  body.appendChild(known);

  // 2 -- the two worlds
  const cf = el('div', null,
    '<div class="step"><i>2</i><div class="lab">What happens either way</div></div>');
  const mx = Math.max(P.buys_if_contacted_pct, P.buys_alone_pct, 1);
  [['If we contact them', P.buys_if_contacted_pct, 'var(--act)'],
   ['If we leave them alone', P.buys_alone_pct, 'var(--zero)']].forEach(([t, val, col]) => {
    cf.appendChild(el('div', 'cf', `<div class="t">${t}</div>
      <div class="tr"><i style="width:${(val / mx) * 100}%;background:${col}"></i></div>
      <div class="v">${val.toFixed(1)}%</div>`));
  });
  const note = el('p', null, `Difference: <b class="${dir(P.extra_sales_pp)}">${pts(P.extra_sales_pp)}
    points</b>. This is an estimate for one person, and single-customer estimates are noisy —
    trust the ranking across thousands, not the decimal here.`);
  note.style.cssText = 'color:var(--ink2);font-size:13px;margin:10px 0 0';
  cf.appendChild(note);
  body.appendChild(cf);

  // 3 -- the decision
  const decide = el('div', null,
    '<div class="step final"><i>3</i><div class="lab">So what should we do</div></div>');
  decide.appendChild(el('div', 'verdict ' + D.tone, `
    <span class="then">Based on the two figures above</span>
    <div class="t">${esc(D.label)}</div>
    <p>${esc(D.why)}</p>
    ${D.reward ? `<p style="margin-top:11px"><b style="color:var(--ink)">Send: ${esc(D.reward)}</b><br>
      <span style="font-size:13px">${esc(D.reward_why)}</span></p>` : ''}
    <p style="margin-top:11px;font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--ink)">
      Worth ${money(D.value)} &middot; costs ${money(D.cost)} to reach</p>`));
  body.appendChild(decide);

  // 4 -- what actually happened, since this is a completed experiment
  const hp = el('div', null,
    '<div class="step"><i>4</i><div class="lab">What actually happened to them</div></div>');
  hp.innerHTML += `<div class="kv">
      <div><dt>They were</dt><dd style="font-size:12.5px">${esc(H.arm)}</dd></div>
      <div><dt>Came back?</dt><dd class="${H.visited ? 'up' : ''}">${H.visited ? 'Yes' : 'No'}</dd></div>
      <div><dt>Spent</dt><dd>${money(H.spent)}</dd></div>
    </div>`;
  hp.appendChild(el('div', 'note', `This campaign already ran, so the outcome is on record. The
    model never saw it — it is here so you can check the recommendation against reality.`));
  hp.lastChild.style.marginTop = '11px';
  body.appendChild(hp);

  dr.append(head, body);
}

const closeDrawer = () => {
  $('#drawer').classList.remove('open');
  $('#scrim').classList.remove('open');
};

/* ------------------------------------------------------------------ boot */
function render() {
  const v = $('#view');
  v.innerHTML = '';
  document.querySelectorAll('.tab').forEach(t => {
    const on = t.dataset.tab === S.tab;
    t.classList.toggle('on', on);
    t.setAttribute('aria-selected', on ? 'true' : 'false');
  });
  try { TABS[S.tab](v); }
  catch (e) {
    v.innerHTML = `<div class="card"><div class="card-b"><b class="dn">Something went wrong on this tab.</b>
      <pre class="mono" style="font-size:12px;color:var(--ink2);white-space:pre-wrap;margin-top:8px">${esc(e.message)}</pre></div></div>`;
    console.error(e);
  }
  window.scrollTo({ top: 0, behavior: 'instant' });
  revealAll(v);
}

(async function boot() {
  document.querySelectorAll('.tab').forEach(t => { t.onclick = () => { S.tab = t.dataset.tab; render(); }; });
  $('#scrim').onclick = closeDrawer;
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });

  const tb = $('#theme');
  const saved = localStorage.getItem('nb-theme') === 'dark' ? 'dark' : 'light';
  document.documentElement.dataset.theme = saved;
  tb.textContent = saved === 'dark' ? 'Light' : 'Dark';
  tb.onclick = () => {
    const n = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = n;
    try { localStorage.setItem('nb-theme', n); } catch (_) {}
    tb.textContent = n === 'dark' ? 'Light' : 'Dark';
    render();
  };

  $('#view').innerHTML = '<div class="sk" style="height:170px;margin-bottom:16px"></div><div class="sk" style="height:420px"></div>';
  // Only the built static bundle ships this marker. Without the guard the
  // server build fetched a data.json that does not exist and logged a 404.
  if (window.NB_STATIC) await loadStatic();
  try {
    S.data = await api('/api/overview');
    if (S.data.app) {
      $('#bname').textContent = S.data.app.name;
      $('#btag').textContent = S.data.app.tagline;
      $('#mark').setAttribute('aria-label', S.data.app.name);
      document.title = S.data.app.name;   // a tab truncates past ~20 chars
    }
    // Opening sequence: one pass, then the class comes off so tab switches and
    // re-renders are instant rather than re-animating the chrome every time.
    document.body.classList.add('boot');
    render();
    setTimeout(() => document.body.classList.remove('boot'), 1500);
  } catch (e) {
    $('#view').innerHTML = `<div class="card"><div class="card-b">
      <b class="dn">The data has not been built yet.</b>
      <p style="color:var(--ink2);margin-top:8px">Run <code class="mono">python simple/realbuild.py</code>, then reload.</p>
      <pre class="mono" style="font-size:12px;color:var(--ink3)">${esc(e.message)}</pre></div></div>`;
    console.error(e);
  }
})();
