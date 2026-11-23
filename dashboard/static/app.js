// Live quoting dashboard: polls the local engine API and draws real vs simulated quotes.
const COLORS = { hjb: "#2563eb", fixed: "#9333ea", skew: "#d97706", ppo: "#059669", v3: "#dc2626" };
const REFRESH_MS = 3000;
let selected = null;
try { selected = localStorage.getItem("dash.market"); } catch (e) {}
let chart = null;
let lastState = null;

const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);
const money = x => {
  if (x === null || x === undefined) return "–";
  if (Math.abs(x) < 0.005) return "$0.00";          // fee rounding dust, not a real gain or loss
  return `<span class="${x > 0 ? "pos" : "neg"}">${x > 0 ? "+" : "−"}$${Math.abs(x).toFixed(2)}</span>`;
};
const clock = ts => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
function dur(s) {
  if (s === null || s === undefined) return "–";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${sec}s` : `${sec}s`;
}
function closeText(m, st) {
  if (m.status !== "open") return m.status + (m.result ? " " + m.result.toUpperCase() : "");
  const left = m.close_ts - st.now;
  // Expected close is an estimate; games often run past it. Strategies then sit at the episode end (t = 1).
  return left > 0 ? `closes in ${dur(left)}` : `past expected close by ${dur(-left)}, still trading`;
}
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

function renderStatus(st) {
  const up = dur(st.now - st.started);
  document.getElementById("status").textContent =
    `updated ${clock(st.now)} · up ${up} · ${st.requests} API requests · ${st.feed_errors} retries · decisions every ${dur(st.decision_s)}`;
  document.getElementById("footer").textContent =
    `Fill model: queue ahead at placement, then ${Math.round(st.settings.participation * 100)}% of the remaining trade volume; ` +
    `${st.settings.latency_s}s latency; ${st.settings.quote_size} contracts per side. Data: Kalshi public production API.`;
}

function renderTotals(st) {
  document.getElementById("totals").innerHTML = Object.entries(st.totals).map(([name, t]) => `
    <div class="total">
      <div class="name"><span class="swatch" style="background:${COLORS[name] || "#888"}"></span>${esc(t.label)}</div>
      <div class="value">${t.available ? money(t.pnl) : '<span class="muted">unavailable</span>'}</div>
      <div class="sub">${t.fills} simulated fills · marked PnL, all markets</div>
    </div>`).join("");
}

function renderMarkets(st) {
  if (!selected || !st.markets.some(m => m.ticker === selected)) selected = st.markets[0]?.ticker || null;
  document.getElementById("markets").innerHTML = st.markets.map(m => `
    <div class="market ${m.ticker === selected ? "active" : ""}" data-ticker="${esc(m.ticker)}">
      <div class="t">${esc(m.title)}</div>
      <div class="s">${esc(m.subtitle)} · ${esc(m.ticker)}</div>
      <div class="q">${m.healthy ? `${fmt(m.bid)} / ${fmt(m.ask)}` : '<span class="muted">book unavailable</span>'}
        <span class="muted"> · ${closeText(m, st)}</span></div>
    </div>`).join("");
  document.querySelectorAll(".market").forEach(el => el.onclick = () => {
    selected = el.dataset.ticker;
    try { localStorage.setItem("dash.market", selected); } catch (e) {}
    refresh();
  });
}

function renderHead(m, st) {
  document.getElementById("market-head").innerHTML = `
    <div><div class="title">${esc(m.title)}</div>
      <div class="meta">${esc(m.subtitle)} · ${esc(m.ticker)} · ${esc(m.series)}</div></div>
    <div class="meta">best ${fmt(m.bid)} / ${fmt(m.ask)} · ${closeText(m, st)}
      ${m.status === "open" ? `· next decision in ${dur(m.next_decision_s)}` : ""} · ${m.trades_seen} trades seen</div>`;
}

function renderStrategyTable(m) {
  const rows = m.strategies.map(s => {
    if (!s.available) return `<tr><td><span class="swatch" style="background:${(COLORS[s.name] || "#888")}"></span> ${esc(s.label)}</td>
      <td colspan="7" class="muted" style="text-align:left">${esc(s.note)}</td></tr>`;
    const b = s.orders.bid, a = s.orders.ask;
    const side = o => o ? `${fmt(o.price)} <span class="muted">×${fmt(o.remaining, 0)}</span>` : '<span class="muted">–</span>';
    const q = o => o ? fmt(o.queue_ahead, 0) : "–";
    return `<tr>
      <td><span class="swatch" style="background:${COLORS[s.name]}"></span> ${esc(s.label)}${s.note ? ` <span class="muted">(${esc(s.note)})</span>` : ""}</td>
      <td>${side(b)}</td><td>${side(a)}</td><td>${q(b)} / ${q(a)}</td>
      <td>${fmt(s.inventory, 0)}</td><td>${s.fills}</td><td>${money(s.pnl)}</td>
      <td>${fmt(s.info.episode_fraction, 2)}</td></tr>`;
  }).join("");
  document.getElementById("strategy-table").innerHTML =
    `<tr><th>Strategy</th><th>Bid</th><th>Ask</th><th>Queue ahead (bid/ask)</th><th>Inventory</th><th>Fills</th><th>PnL</th><th>Episode t</th></tr>${rows}`;
}

function renderBook(detail, m) {
  const el = document.getElementById("book");
  if (!detail.levels) { el.innerHTML = '<tr><td class="muted">book unavailable</td></tr>'; return; }
  const ours = new Set();
  m.strategies.forEach(s => Object.values(s.orders).forEach(o => ours.add(o.price.toFixed(4))));
  const row = (cls, p, q) => `<tr class="${cls}"><td>${cls === "askrow" ? "ask" : "bid"}</td>
    <td class="${ours.has(Number(p).toFixed(4)) ? "ours" : ""}">${fmt(p)}${ours.has(Number(p).toFixed(4)) ? " ◀ ours" : ""}</td><td>${fmt(q, 0)}</td></tr>`;
  const asks = detail.levels.asks.slice(0, 6).reverse().map(([p, q]) => row("askrow", p, q)).join("");
  const bids = detail.levels.bids.slice(0, 6).map(([p, q]) => row("bidrow", p, q)).join("");
  el.innerHTML = `<tr><th>Side</th><th>Price</th><th>Size</th></tr>${asks}${bids}`;
}

function renderFills(detail, m) {
  const labels = Object.fromEntries(m.strategies.map(s => [s.name, s.label]));
  const all = Object.entries(detail.fills).flatMap(([n, fs]) => fs.map(f => ({ ...f, n })))
    .sort((a, b) => b.ts - a.ts).slice(0, 25);
  document.getElementById("fills").innerHTML = all.length
    ? `<tr><th>Time</th><th>Strategy</th><th>Side</th><th>Price</th><th>Qty</th><th>Trade (price × size)</th><th>Queue before → after</th><th>Inventory</th></tr>` +
      all.map(f => `<tr><td>${clock(f.ts)}</td><td>${esc(labels[f.n])}</td><td>${f.side === "bid" ? "bought" : "sold"}</td>
        <td>${fmt(f.price)}</td><td>${fmt(f.qty)}</td><td>${fmt(f.trade_price)} × ${fmt(f.trade_qty, 0)}</td>
        <td>${fmt(f.queue_before, 0)} → ${fmt(f.queue_after, 0)}</td><td>${fmt(f.inventory, 0)}</td></tr>`).join("")
    : '<tr><td class="muted">No simulated fills yet in this market. Real trades must reach our price and clear the queue ahead first.</td></tr>';
}

function stepSeries(points, idx, endTs) {
  // Quote history as a step line that runs to "now"; null breaks the line when not quoting.
  const out = points.map(p => ({ x: p[0], y: p[idx] }));
  if (points.length) out.push({ x: endTs, y: points[points.length - 1][idx] });
  return out;
}

function renderChart(detail, m, st) {
  const hist = detail.history;
  // Chart.js draws lower "order" on top: real market data (0) above simulated quotes (1).
  const datasets = [
    { label: "Market bid", data: hist.map(h => ({ x: h[0], y: h[1] })), borderColor: "#98a2b3", borderWidth: 2, pointRadius: 0, order: 0 },
    { label: "Market ask", data: hist.map(h => ({ x: h[0], y: h[2] })), borderColor: "#667085", borderWidth: 2, pointRadius: 0, order: 0 },
    { label: "Trades (YES taker)", type: "scatter", data: detail.trades.filter(t => t.side === "yes").map(t => ({ x: t.ts, y: t.price })),
      backgroundColor: "rgba(15,157,88,.7)", pointRadius: 3, order: 0 },
    { label: "Trades (NO taker)", type: "scatter", data: detail.trades.filter(t => t.side === "no").map(t => ({ x: t.ts, y: t.price })),
      backgroundColor: "rgba(217,48,37,.7)", pointRadius: 3, order: 0 },
  ];
  m.strategies.filter(s => s.available).forEach(s => {
    const q = detail.quotes[s.name] || [];
    datasets.push({ label: `${s.label} bid`, data: stepSeries(q, 1, st.now), stepped: "after", borderColor: COLORS[s.name],
      borderWidth: 2, pointRadius: 0, spanGaps: false, order: 1 });
    datasets.push({ label: `${s.label} ask`, data: stepSeries(q, 2, st.now), stepped: "after", borderColor: COLORS[s.name],
      borderDash: [6, 4], borderWidth: 2, pointRadius: 0, spanGaps: false, order: 1 });
  });
  if (!chart) {
    chart = new Chart(document.getElementById("chart"), {
      type: "line", data: { datasets },
      options: {
        animation: false, maintainAspectRatio: false, parsing: false, normalized: true,
        interaction: { mode: "nearest", intersect: false },
        scales: {
          x: { type: "linear", ticks: { callback: v => clock(v), maxTicksLimit: 8 } },
          y: { title: { display: true, text: "YES price" }, ticks: { callback: v => v.toFixed(2) } },
        },
        plugins: { legend: { labels: { boxWidth: 12, font: { size: 11 } } },
                   tooltip: { callbacks: { title: items => clock(items[0].parsed.x) } } },
      },
    });
  } else {
    const hidden = new Set(chart.data.datasets.filter((d, i) => !chart.isDatasetVisible(i)).map(d => d.label));
    chart.data.datasets = datasets;
    chart.data.datasets.forEach((d, i) => d.hidden = hidden.has(d.label));
    chart.update("none");
  }
}

async function refresh() {
  try {
    const st = await getJSON("/api/state");
    lastState = st;
    renderStatus(st); renderTotals(st); renderMarkets(st);
    const m = st.markets.find(x => x.ticker === selected);
    if (!m) return;
    const detail = await getJSON(`/api/market/${encodeURIComponent(m.ticker)}`);
    renderHead(m, st); renderStrategyTable(m); renderBook(detail, m); renderFills(detail, m); renderChart(detail, m, st);
  } catch (e) {
    document.getElementById("status").textContent = `connection problem: ${e.message}`;
  }
}

refresh();
setInterval(refresh, REFRESH_MS);
