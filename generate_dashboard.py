import os
import json
import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

STRATEGIES    = ["daily", "weekly", "monthly"]
STOCK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "URTH", "EEM", "IEV", "EWJ", "SCZ",
    "TSM", "ASML", "NVO", "MHVYF", "SIEGY",
    "SBGSY", "BN", "SHEL", "BRK-B",
]
BENCHMARK     = "SPY"
LOOKBACK      = 90
TOP_N         = 3


def fetch_stocks(tickers):
    data = yf.download(tickers, period="1y", auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    data.index = pd.to_datetime(data.index).normalize()
    return data


def fetch_btc():
    try:
        resp = requests.get(
            "https://api.kraken.com/0/public/OHLC",
            params={"pair": "XBTUSD", "interval": 1440},
            timeout=10,
        )
        result = resp.json()
        key = next(k for k in result["result"] if k != "last")
        rows = result["result"][key]
        df = pd.DataFrame(rows, columns=["time","open","high","low","close","vwap","volume","count"])
        df["date"] = pd.to_datetime(df["time"], unit="s").dt.normalize()
        df = df.set_index("date")[["close"]].rename(columns={"close": "BTC"})
        df["BTC"] = df["BTC"].astype(float)
        return df
    except Exception as e:
        print(f"BTC fetch failed: {e}")
        return None


def sma(s, w):
    return s.rolling(w).mean()


def rsi(s, period=14):
    d = s.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


def load_performance(strategy):
    f = f"performance_log_{strategy}.csv"
    if not os.path.exists(f):
        return pd.DataFrame(columns=["date", "portfolio_value", "benchmark_value"])
    return pd.read_csv(f, parse_dates=["date"])


def load_portfolio(strategy):
    f = f"paper_portfolio_{strategy}.csv"
    if not os.path.exists(f):
        return pd.DataFrame(columns=["ticker", "shares"])
    return pd.read_csv(f)


def load_trades(strategy):
    f = f"trade_log_{strategy}.csv"
    if not os.path.exists(f):
        return pd.DataFrame()
    return pd.read_csv(f, parse_dates=["date"])


def momentum_history(prices):
    mom = prices.pct_change(LOOKBACK) * 100
    dates = mom.index.strftime("%Y-%m-%d").tolist()
    series = {}
    for col in mom.columns:
        if col == BENCHMARK:
            continue
        series[col] = [round(v, 2) if not np.isnan(v) else None for v in mom[col].tolist()]
    return {"dates": dates, "series": series}


def selection_history(prices):
    mom = prices.drop(columns=[BENCHMARK], errors="ignore").pct_change(LOOKBACK)
    assets = mom.columns.tolist()
    dates = []
    matrix = {t: [] for t in assets}
    for date, row in mom.iterrows():
        valid = row.dropna()
        if len(valid) < TOP_N:
            continue
        top = set(valid.sort_values(ascending=False).head(TOP_N).index)
        dates.append(date.strftime("%Y-%m-%d"))
        for t in assets:
            matrix[t].append(1 if t in top else 0)
    return {"dates": dates, "assets": assets, "matrix": matrix}


def correlation(prices):
    rets = prices.drop(columns=[BENCHMARK], errors="ignore").pct_change().dropna()
    corr = rets.corr().round(3)
    return {"tickers": corr.columns.tolist(), "matrix": corr.values.tolist()}


def strategy_analysis(prices):
    assets = [c for c in prices.columns if c != BENCHMARK]
    ret90 = prices[assets].pct_change(LOOKBACK).iloc[-1].dropna().sort_values(ascending=False)
    ret30 = prices[assets].pct_change(21).iloc[-1]
    ret7  = prices[assets].pct_change(5).iloc[-1]
    ret1  = prices[assets].pct_change(1).iloc[-1]

    rows = []
    for rank, (t, v) in enumerate(ret90.items(), 1):
        rows.append({
            "rank":   rank,
            "ticker": t,
            "mom90":  round(float(v) * 100, 2),
            "mom30":  round(float(ret30[t]) * 100, 2) if t in ret30 else None,
            "mom7":   round(float(ret7[t]) * 100, 2)  if t in ret7  else None,
            "mom1":   round(float(ret1[t]) * 100, 2)  if t in ret1  else None,
            "in_top": rank <= TOP_N,
        })

    per_strategy = {}
    top_set = set(ret90.head(TOP_N).index)
    for s in STRATEGIES:
        port = load_portfolio(s)
        held = {r["ticker"] for _, r in port.iterrows() if r["ticker"] != "CASH"}
        per_strategy[s] = {
            "held":        list(held),
            "should_buy":  [t for t in top_set if t not in held],
            "should_sell": [t for t in held if t not in top_set],
            "holding_ok":  [t for t in held if t in top_set],
        }

    return {"ranking": rows, "strategy_status": per_strategy}


def build_data():
    prices = fetch_stocks(STOCK_TICKERS + [BENCHMARK])
    btc = fetch_btc()
    if btc is not None:
        prices = prices.join(btc, how="left")
        prices["BTC"] = prices["BTC"].ffill()

    out = {}

    equity = {}
    for s in STRATEGIES:
        df = load_performance(s)
        if df.empty:
            continue
        equity[s] = {
            "dates":     df["date"].dt.strftime("%Y-%m-%d").tolist(),
            "portfolio": df["portfolio_value"].tolist(),
            "benchmark": df["benchmark_value"].tolist() if "benchmark_value" in df.columns else [],
        }
    out["equity"] = equity

    all_tickers = [t for t in STOCK_TICKERS + ["BTC"] if t in prices.columns]
    technicals = {}
    for s in STRATEGIES:
        port = load_portfolio(s)
        held = {r["ticker"] for _, r in port.iterrows() if r["ticker"] != "CASH"}
        ticker_data = {}
        for t in all_tickers:
            series = prices[t].dropna()
            dates  = series.index.strftime("%Y-%m-%d").tolist()
            ticker_data[t] = {
                "dates":  dates,
                "close":  series.tolist(),
                "sma20":  sma(series, 20).tolist(),
                "sma50":  sma(series, 50).tolist(),
                "sma200": sma(series, 200).tolist(),
                "rsi14":  rsi(series, 14).tolist(),
                "held":   t in held,
            }
        technicals[s] = ticker_data
    out["technicals"] = technicals

    trades = {}
    for s in STRATEGIES:
        df = load_trades(s)
        trades[s] = [] if df.empty else df.sort_values("date", ascending=False).head(20).to_dict(orient="records")
    out["trades"] = trades

    ret = prices.pct_change(LOOKBACK).iloc[-1].dropna().sort_values(ascending=False)
    out["momentum"] = {
        "tickers": ret.index.tolist(),
        "returns": [round(v * 100, 2) for v in ret.tolist()],
    }

    out["momentum_history"]  = momentum_history(prices)
    out["selection_history"] = selection_history(prices)
    out["correlation"]       = correlation(prices)
    out["strategy_analysis"] = strategy_analysis(prices)
    out["updated"] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return out


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Paper Trader</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e2e8f0; --muted: #8892a4; --accent: #6366f1;
    --green: #22c55e; --red: #ef4444;
    --daily: #6366f1; --weekly: #22c55e; --monthly: #f59e0b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; padding: 24px; max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: 4px; }
  .meta { color: var(--muted); font-size: .8rem; margin-bottom: 28px; }
  h2 { font-size: 1.1rem; font-weight: 600; margin-bottom: 14px; }
  h3 { font-size: .8rem; font-weight: 600; margin-bottom: 8px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
  .note { font-size: .82rem; color: var(--muted); margin-bottom: 14px; }
  .tabs { display: flex; gap: 8px; margin-bottom: 16px; }
  .tab { padding: 6px 16px; border-radius: 6px; border: 1px solid var(--border); background: transparent; color: var(--muted); cursor: pointer; font-size: .85rem; }
  .tab.active { background: var(--accent); border-color: var(--accent); color: #fff; }
  .tab:hover:not(.active) { border-color: var(--accent); color: var(--text); }
  .plot      { width: 100%; height: 320px; }
  .plot-tall { width: 100%; height: 420px; }
  .plot-sm   { width: 100%; height: 220px; }
  table { width: 100%; border-collapse: collapse; font-size: .85rem; }
  th { text-align: left; padding: 8px 10px; color: var(--muted); border-bottom: 1px solid var(--border); font-weight: 500; }
  td { padding: 8px 10px; border-bottom: 1px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  .buy  { color: var(--green); font-weight: 600; }
  .sell { color: var(--red);   font-weight: 600; }
  .hidden { display: none; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .legend { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: .82rem; color: var(--muted); }
  @media(max-width:700px) { .grid2 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>Paper Trader</h1>
<p class="meta">Updated: <span id="updated"></span></p>

<div class="card">
  <h2>Performance vs SPY</h2>
  <div class="legend">
    <div class="legend-item"><div class="dot" style="background:var(--daily)"></div>Daily</div>
    <div class="legend-item"><div class="dot" style="background:var(--weekly)"></div>Weekly</div>
    <div class="legend-item"><div class="dot" style="background:var(--monthly)"></div>Monthly</div>
    <div class="legend-item"><div class="dot" style="background:#475569"></div>SPY</div>
  </div>
  <div id="equity-plot" class="plot-tall"></div>
</div>

<div class="card">
  <h2>How assets are selected</h2>
  <p class="note">All assets are ranked by 90-day momentum. The top 3 are held. Rebalancing happens daily / every Monday / end of month depending on the strategy.</p>

  <h3>Momentum score history</h3>
  <p class="note">When lines cross, the ranking changes and a rebalance will swap positions on the next scheduled date.</p>
  <div id="momentum-history-plot" class="plot-tall"></div>

  <h3 style="margin-top:24px">Selection history — which assets were in the top 3</h3>
  <div id="selection-heatmap" class="plot"></div>

  <div class="grid2" style="margin-top:24px">
    <div>
      <h3>Current ranking (90d return)</h3>
      <div id="momentum-bar" class="plot-sm"></div>
    </div>
    <div>
      <h3>Return correlations (1y daily)</h3>
      <div id="correlation-plot" class="plot-sm"></div>
    </div>
  </div>

  <h3 style="margin-top:24px">Ranking table</h3>
  <div id="ranking-table" style="margin-bottom:20px"></div>

  <h3 style="margin-top:20px">Current position status</h3>
  <div id="position-status"></div>
</div>

<div class="card">
  <h2>Technicals — all assets</h2>
  <div class="tabs" id="tech-tabs">
    <button class="tab active" onclick="showTab('tech','daily',this)">Daily</button>
    <button class="tab" onclick="showTab('tech','weekly',this)">Weekly</button>
    <button class="tab" onclick="showTab('tech','monthly',this)">Monthly</button>
  </div>
  <div id="tech-daily"></div>
  <div id="tech-weekly" class="hidden"></div>
  <div id="tech-monthly" class="hidden"></div>
</div>

<div class="card">
  <h2>Recent trades</h2>
  <div class="tabs" id="trades-tabs">
    <button class="tab active" onclick="showTab('trades','daily',this)">Daily</button>
    <button class="tab" onclick="showTab('trades','weekly',this)">Weekly</button>
    <button class="tab" onclick="showTab('trades','monthly',this)">Monthly</button>
  </div>
  <div id="trades-daily"></div>
  <div id="trades-weekly" class="hidden"></div>
  <div id="trades-monthly" class="hidden"></div>
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;

const ASSET_COLORS    = ['#6366f1','#22c55e','#f59e0b','#ef4444','#38bdf8','#a78bfa','#fb923c'];
const STRATEGY_COLORS = { daily: '#6366f1', weekly: '#22c55e', monthly: '#f59e0b' };
const L = {
  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
  font: { color: '#e2e8f0', size: 11 },
  margin: { t: 10, b: 50, l: 55, r: 10 },
  xaxis: { gridcolor: '#2a2d3a', linecolor: '#2a2d3a' },
  yaxis: { gridcolor: '#2a2d3a', linecolor: '#2a2d3a' },
  legend: { bgcolor: 'transparent' },
  hovermode: 'x unified',
};
const CFG = { displayModeBar: false, responsive: true };

document.getElementById('updated').textContent = DATA.updated;

// equity curves
(function() {
  const traces = [];
  let benchAdded = false;
  for (const [s, d] of Object.entries(DATA.equity)) {
    traces.push({ x: d.dates, y: d.portfolio, name: s.charAt(0).toUpperCase()+s.slice(1),
                  line: { color: STRATEGY_COLORS[s], width: 2 }, type: 'scatter' });
    if (!benchAdded && d.benchmark?.length) {
      traces.push({ x: d.dates, y: d.benchmark, name: 'SPY',
                    line: { color: '#475569', width: 1.5, dash: 'dash' }, type: 'scatter' });
      benchAdded = true;
    }
  }
  Plotly.newPlot('equity-plot', traces, { ...L, yaxis: { ...L.yaxis, tickprefix: '$' } }, CFG);
})();

// momentum history
(function() {
  const mh = DATA.momentum_history;
  const traces = Object.entries(mh.series).map(([t, vals], i) => ({
    x: mh.dates, y: vals, name: t,
    line: { color: ASSET_COLORS[i % ASSET_COLORS.length], width: 1.5 },
    type: 'scatter',
    hovertemplate: `${t}: %{y:.1f}%<extra></extra>`,
  }));
  const shapes = [{ type: 'line', x0: mh.dates[0], x1: mh.dates[mh.dates.length-1],
                    y0: 0, y1: 0, line: { color: '#475569', width: 1, dash: 'dot' } }];
  Plotly.newPlot('momentum-history-plot', traces, { ...L, yaxis: { ...L.yaxis, ticksuffix: '%' }, shapes }, CFG);
})();

// selection heatmap
(function() {
  const sh = DATA.selection_history;
  if (!sh.dates.length) return;
  Plotly.newPlot('selection-heatmap', [{
    type: 'heatmap', x: sh.dates, y: sh.assets,
    z: sh.assets.map(a => sh.matrix[a]),
    colorscale: [[0, '#1a1d27'], [1, '#6366f1']],
    showscale: false,
    hovertemplate: '%{y} on %{x}: %{z}<extra></extra>',
  }], { ...L, margin: { t: 5, b: 50, l: 60, r: 10 }, yaxis: { ...L.yaxis, autorange: 'reversed' } }, CFG);
})();

// momentum bar
(function() {
  const m = DATA.momentum;
  const top3 = new Set(m.tickers.slice(0, 3));
  const colors = m.tickers.map((t, i) => top3.has(t)
    ? (m.returns[i] >= 0 ? '#22c55e' : '#ef4444')
    : (m.returns[i] >= 0 ? '#2d4a38' : '#4a2d2d'));
  Plotly.newPlot('momentum-bar', [{
    x: m.tickers, y: m.returns, type: 'bar',
    marker: { color: colors },
    text: m.returns.map(v => v.toFixed(1)+'%'), textposition: 'outside',
    hovertemplate: '%{x}: %{y:.1f}%<extra></extra>',
  }], { ...L, margin: { t: 20, b: 40, l: 45, r: 10 }, yaxis: { ...L.yaxis, ticksuffix: '%' } }, CFG);
})();

// correlation
(function() {
  const c = DATA.correlation;
  Plotly.newPlot('correlation-plot', [{
    type: 'heatmap', x: c.tickers, y: c.tickers, z: c.matrix,
    colorscale: 'RdBu', reversescale: true, zmin: -1, zmax: 1,
    text: c.matrix.map(row => row.map(v => v.toFixed(2))),
    texttemplate: '%{text}', showscale: false,
    hovertemplate: '%{y} vs %{x}: %{z:.2f}<extra></extra>',
  }], { ...L, margin: { t: 5, b: 60, l: 60, r: 10 },
        xaxis: { ...L.xaxis, tickangle: -35 },
        yaxis: { ...L.yaxis, autorange: 'reversed' } }, CFG);
})();

// ranking table + position status
(function() {
  const sa = DATA.strategy_analysis;
  const fmt = v => v == null || isNaN(v) ? '—' : (v >= 0 ? '+' : '') + v.toFixed(1) + '%';
  const col  = v => v == null || isNaN(v) ? '' : v >= 0 ? 'style="color:var(--green)"' : 'style="color:var(--red)"';

  document.getElementById('ranking-table').innerHTML = `<table>
    <thead><tr><th>Rank</th><th>Ticker</th><th>90d</th><th>30d</th><th>7d</th><th>1d</th><th></th></tr></thead>
    <tbody>${sa.ranking.map(r => `<tr style="${r.in_top ? 'background:rgba(99,102,241,0.08)' : ''}">
      <td>${r.rank}</td>
      <td><strong>${r.ticker}</strong></td>
      <td ${col(r.mom90)}><strong>${fmt(r.mom90)}</strong></td>
      <td ${col(r.mom30)}>${fmt(r.mom30)}</td>
      <td ${col(r.mom7)}>${fmt(r.mom7)}</td>
      <td ${col(r.mom1)}>${fmt(r.mom1)}</td>
      <td>${r.in_top ? '<span style="background:var(--accent);color:#fff;font-size:.72rem;padding:2px 8px;border-radius:4px">SELECTED</span>' : ''}</td>
    </tr>`).join('')}</tbody>
  </table>`;

  document.getElementById('position-status').innerHTML =
    `<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
    ${['daily','weekly','monthly'].map(s => {
      const st = sa.strategy_status[s];
      const lines = [];
      if (st.holding_ok.length)  lines.push(`<div style="margin-bottom:6px"><span style="color:var(--green);font-weight:600">Holding</span><br><span style="font-size:.85rem">${st.holding_ok.join(', ')}</span></div>`);
      if (st.should_buy.length)  lines.push(`<div style="margin-bottom:6px"><span style="color:var(--accent);font-weight:600">Will buy on rebalance</span><br><span style="font-size:.85rem">${st.should_buy.join(', ')}</span></div>`);
      if (st.should_sell.length) lines.push(`<div style="margin-bottom:6px"><span style="color:var(--red);font-weight:600">Will sell on rebalance</span><br><span style="font-size:.85rem">${st.should_sell.join(', ')}</span></div>`);
      if (!st.held.length)       lines.push(`<span style="color:var(--muted);font-size:.85rem">No positions yet</span>`);
      return `<div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:14px">
        <div style="font-weight:700;margin-bottom:10px;text-transform:capitalize">${s}</div>
        ${lines.join('')}
      </div>`;
    }).join('')}
  </div>`;
})();

// technicals
function buildTechnicals(strategy) {
  const container = document.getElementById(`tech-${strategy}`);
  const tickers = DATA.technicals[strategy];
  if (!tickers || !Object.keys(tickers).length) {
    container.innerHTML = '<p style="color:var(--muted);font-size:.85rem;padding:8px 0">No data yet.</p>';
    return;
  }
  container.innerHTML = '';
  for (const [t, d] of Object.entries(tickers)) {
    const wrap = document.createElement('div');
    wrap.style.marginBottom = '28px';

    const h = document.createElement('h3');
    const badge = d.held ? ' <span style="background:var(--accent);color:#fff;font-size:.7rem;padding:2px 7px;border-radius:4px;font-weight:600">HELD</span>' : '';
    h.innerHTML = `${t} — Price & SMAs${badge}`;
    wrap.appendChild(h);

    const priceEl = document.createElement('div');
    priceEl.id = `price-${strategy}-${t}`;
    priceEl.className = 'plot';
    wrap.appendChild(priceEl);

    const rsiH = document.createElement('h3');
    rsiH.textContent = `${t} — RSI 14`;
    rsiH.style.marginTop = '14px';
    wrap.appendChild(rsiH);

    const rsiEl = document.createElement('div');
    rsiEl.id = `rsi-${strategy}-${t}`;
    rsiEl.style.height = '160px';
    wrap.appendChild(rsiEl);

    container.appendChild(wrap);

    Plotly.newPlot(priceEl.id, [
      { x: d.dates, y: d.close,  name: 'Price',   line: { color: '#e2e8f0', width: 1.5 }, type: 'scatter' },
      { x: d.dates, y: d.sma20,  name: 'SMA 20',  line: { color: '#6366f1', width: 1.2, dash: 'dot' }, type: 'scatter' },
      { x: d.dates, y: d.sma50,  name: 'SMA 50',  line: { color: '#22c55e', width: 1.2, dash: 'dot' }, type: 'scatter' },
      { x: d.dates, y: d.sma200, name: 'SMA 200', line: { color: '#f59e0b', width: 1.2, dash: 'dot' }, type: 'scatter' },
    ], { ...L, yaxis: { ...L.yaxis, tickprefix: '$' } }, CFG);

    const hline = (y0, color) => ({ type: 'line', x0: d.dates[0], x1: d.dates[d.dates.length-1],
                                    y0, y1: y0, line: { color, width: 1, dash: 'dash' } });
    Plotly.newPlot(rsiEl.id, [{
      x: d.dates, y: d.rsi14, name: 'RSI 14',
      line: { color: '#a78bfa', width: 1.5 }, type: 'scatter',
      fill: 'tozeroy', fillcolor: 'rgba(167,139,250,0.08)',
    }], {
      ...L,
      shapes: [hline(70, '#ef4444'), hline(30, '#22c55e')],
      yaxis: { ...L.yaxis, range: [0, 100], tickvals: [0,30,50,70,100] },
      margin: { t: 5, b: 40, l: 45, r: 10 },
    }, CFG);
  }
}
['daily','weekly','monthly'].forEach(buildTechnicals);

// trade log
function buildTrades(strategy) {
  const container = document.getElementById(`trades-${strategy}`);
  const rows = DATA.trades[strategy];
  if (!rows?.length) {
    container.innerHTML = '<p style="color:var(--muted);font-size:.85rem;padding:8px 0">No trades yet.</p>';
    return;
  }
  container.innerHTML = `<table>
    <thead><tr><th>Date</th><th>Ticker</th><th>Action</th><th>Shares</th><th>Price</th><th>Value</th><th>Reason</th></tr></thead>
    <tbody>${rows.map(r => `<tr>
      <td>${String(r.date).slice(0,10)}</td>
      <td><strong>${r.ticker}</strong></td>
      <td class="${String(r.action).toLowerCase()}">${r.action}</td>
      <td>${Number(r.shares).toFixed(4)}</td>
      <td>$${Number(r.price).toFixed(2)}</td>
      <td>$${Number(r.value).toFixed(2)}</td>
      <td style="color:var(--muted);font-size:.8rem">${r.reason || '—'}</td>
    </tr>`).join('')}</tbody>
  </table>`;
}
['daily','weekly','monthly'].forEach(buildTrades);

function showTab(group, strategy, btn) {
  ['daily','weekly','monthly'].forEach(s => {
    document.getElementById(`${group}-${s}`).classList.toggle('hidden', s !== strategy);
  });
  document.querySelectorAll(`#${group}-tabs .tab`).forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
}
</script>
</body>
</html>"""


def generate():
    data = build_data()
    html = HTML.replace("__DATA_PLACEHOLDER__", json.dumps(data, default=str))
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w") as f:
        f.write(html)
    print(f"Dashboard written ({len(html)//1024}KB)")


if __name__ == "__main__":
    generate()
