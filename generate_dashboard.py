import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
import json
import datetime

STRATEGIES = ["daily", "weekly", "monthly"]
STOCK_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
BENCHMARK = "SPY"
LOOKBACK = 90
TOP_N = 3

# ── Price helpers ─────────────────────────────────────────────────────────────

def get_stock_prices(tickers):
    data = yf.download(tickers, period="1y", auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    data.index = pd.to_datetime(data.index).normalize()
    return data

def get_btc_prices():
    url = "https://api.kraken.com/0/public/OHLC"
    try:
        resp = requests.get(url, params={"pair": "XBTUSD", "interval": 1440}, timeout=10)
        data = resp.json()
        key = [k for k in data["result"] if k != "last"][0]
        rows = data["result"][key]
        df = pd.DataFrame(rows, columns=["time","open","high","low","close","vwap","volume","count"])
        df["date"] = pd.to_datetime(df["time"], unit="s").dt.normalize()
        df = df.set_index("date")[["close"]].rename(columns={"close": "BTC"})
        df["BTC"] = df["BTC"].astype(float)
        return df
    except Exception as e:
        print(f"BTC fetch failed: {e}")
        return None

# ── Indicators ────────────────────────────────────────────────────────────────

def sma(series, window):
    return series.rolling(window).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ── Load data ─────────────────────────────────────────────────────────────────

def load_performance(strategy):
    f = f"performance_log_{strategy}.csv"
    if not os.path.exists(f):
        return pd.DataFrame(columns=["date","portfolio_value","benchmark_value"])
    return pd.read_csv(f, parse_dates=["date"])

def load_portfolio(strategy):
    f = f"paper_portfolio_{strategy}.csv"
    if not os.path.exists(f):
        return pd.DataFrame(columns=["ticker","shares"])
    return pd.read_csv(f)

def load_trades(strategy):
    f = f"trade_log_{strategy}.csv"
    if not os.path.exists(f):
        return pd.DataFrame()
    return pd.read_csv(f, parse_dates=["date"])

# ── Strategy analysis data ────────────────────────────────────────────────────

def build_momentum_history(prices):
    """Rolling 90-day momentum score for every asset, every day."""
    momentum = prices.pct_change(LOOKBACK) * 100
    dates = momentum.index.strftime("%Y-%m-%d").tolist()
    series = {}
    for col in momentum.columns:
        if col == BENCHMARK:
            continue
        vals = momentum[col].tolist()
        series[col] = [round(v, 2) if not np.isnan(v) else None for v in vals]
    return {"dates": dates, "series": series}

def build_selection_history(prices):
    """
    For each day, compute which TOP_N assets would have been selected.
    Returns a matrix: rows = assets, cols = dates, value = 1 if selected else 0.
    Only computed from day LOOKBACK onwards.
    """
    momentum = prices.drop(columns=[BENCHMARK], errors="ignore").pct_change(LOOKBACK)
    asset_cols = momentum.columns.tolist()
    dates = []
    # dict: ticker -> list of 0/1
    selections = {t: [] for t in asset_cols}

    for i, (date, row) in enumerate(momentum.iterrows()):
        valid = row.dropna()
        if len(valid) < TOP_N:
            continue
        top = set(valid.sort_values(ascending=False).head(TOP_N).index)
        dates.append(date.strftime("%Y-%m-%d"))
        for t in asset_cols:
            selections[t].append(1 if t in top else 0)

    return {"dates": dates, "assets": asset_cols, "matrix": selections}

def build_correlation(prices):
    """Correlation matrix of daily returns over the past year."""
    rets = prices.drop(columns=[BENCHMARK], errors="ignore").pct_change().dropna()
    corr = rets.corr().round(3)
    tickers = corr.columns.tolist()
    matrix = corr.values.tolist()
    return {"tickers": tickers, "matrix": matrix}

# ── Master build ──────────────────────────────────────────────────────────────

def build_data():
    all_prices = get_stock_prices(STOCK_TICKERS + [BENCHMARK])
    btc = get_btc_prices()
    if btc is not None:
        all_prices = all_prices.join(btc, how="left")
        all_prices["BTC"] = all_prices["BTC"].ffill()

    out = {}

    # 1. Equity curves
    equity = {}
    for s in STRATEGIES:
        df = load_performance(s)
        if df.empty:
            continue
        equity[s] = {
            "dates": df["date"].dt.strftime("%Y-%m-%d").tolist(),
            "portfolio": df["portfolio_value"].tolist(),
            "benchmark": df["benchmark_value"].tolist() if "benchmark_value" in df.columns else [],
        }
    out["equity"] = equity

    # 2. Technicals for current holdings
    technicals = {}
    for s in STRATEGIES:
        port = load_portfolio(s)
        held = [r["ticker"] for _, r in port.iterrows() if r["ticker"] != "CASH"]
        ticker_data = {}
        for ticker in held:
            if ticker not in all_prices.columns:
                continue
            series = all_prices[ticker].dropna()
            dates = series.index.strftime("%Y-%m-%d").tolist()
            ticker_data[ticker] = {
                "dates": dates,
                "close": series.tolist(),
                "sma20":  sma(series, 20).tolist(),
                "sma50":  sma(series, 50).tolist(),
                "sma200": sma(series, 200).tolist(),
                "rsi14":  rsi(series, 14).tolist(),
            }
        technicals[s] = ticker_data
    out["technicals"] = technicals

    # 3. Trades
    trades = {}
    for s in STRATEGIES:
        df = load_trades(s)
        if df.empty:
            trades[s] = []
        else:
            df = df.sort_values("date", ascending=False).head(20)
            trades[s] = df.to_dict(orient="records")
    out["trades"] = trades

    # 4. Current momentum snapshot
    returns = all_prices.pct_change(LOOKBACK).iloc[-1].dropna().sort_values(ascending=False)
    out["momentum"] = {
        "tickers": returns.index.tolist(),
        "returns": [round(v * 100, 2) for v in returns.tolist()],
    }

    # 5. NEW: Momentum score history (rolling 90d return per asset over time)
    out["momentum_history"] = build_momentum_history(all_prices)

    # 6. NEW: Selection history (which assets were in top N each day)
    out["selection_history"] = build_selection_history(all_prices)

    # 7. NEW: Correlation matrix
    out["correlation"] = build_correlation(all_prices)

    out["updated"] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return out

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Paper Trader Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e2e8f0; --muted: #8892a4; --accent: #6366f1;
    --green: #22c55e; --red: #ef4444; --yellow: #f59e0b;
    --daily: #6366f1; --weekly: #22c55e; --monthly: #f59e0b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; padding: 24px; max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: 4px; }
  .updated { color: var(--muted); font-size: .8rem; margin-bottom: 28px; }
  h2 { font-size: 1.1rem; font-weight: 600; margin-bottom: 14px; color: var(--text); }
  h3 { font-size: .85rem; font-weight: 600; margin-bottom: 8px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
  .subtitle { font-size: .82rem; color: var(--muted); margin-bottom: 16px; margin-top: -8px; }
  .tabs { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
  .tab { padding: 6px 16px; border-radius: 6px; border: 1px solid var(--border); background: transparent;
         color: var(--muted); cursor: pointer; font-size: .85rem; transition: all .15s; }
  .tab.active { background: var(--accent); border-color: var(--accent); color: #fff; }
  .tab:hover:not(.active) { border-color: var(--accent); color: var(--text); }
  .plot { width: 100%; height: 320px; }
  .plot-tall { width: 100%; height: 420px; }
  .plot-short { width: 100%; height: 220px; }
  table { width: 100%; border-collapse: collapse; font-size: .85rem; }
  th { text-align: left; padding: 8px 10px; color: var(--muted); border-bottom: 1px solid var(--border); font-weight: 500; }
  td { padding: 8px 10px; border-bottom: 1px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  .buy { color: var(--green); font-weight: 600; } .sell { color: var(--red); font-weight: 600; }
  .legend { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: .82rem; color: var(--muted); }
  .dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .hidden { display: none; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media(max-width:700px) { .grid2 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>📈 Paper Trader Dashboard</h1>
<div class="updated">Last updated: <span id="updated"></span></div>

<!-- EQUITY CURVES -->
<div class="card">
  <h2>Strategy Performance vs SPY Benchmark</h2>
  <div class="legend">
    <div class="legend-item"><div class="dot" style="background:var(--daily)"></div>Daily</div>
    <div class="legend-item"><div class="dot" style="background:var(--weekly)"></div>Weekly</div>
    <div class="legend-item"><div class="dot" style="background:var(--monthly)"></div>Monthly</div>
    <div class="legend-item"><div class="dot" style="background:#475569"></div>SPY Benchmark</div>
  </div>
  <div id="equity-plot" class="plot-tall"></div>
</div>

<!-- STRATEGY ANALYSIS SECTION -->
<div class="card">
  <h2>Strategy Analysis — How Stocks Are Selected</h2>
  <p class="subtitle">The strategy ranks all assets by their 90-day momentum score and picks the top 3. These charts show how that works over time.</p>

  <!-- Momentum history -->
  <h3>Momentum Score History (Rolling 90-day return per asset)</h3>
  <p class="subtitle" style="margin-top:-4px">Lines crossing = ranking changes. The top 3 at any point are what gets bought.</p>
  <div id="momentum-history-plot" class="plot-tall"></div>

  <!-- Selection heatmap -->
  <h3 style="margin-top:24px">Selection History — Which Assets Were in the Top 3</h3>
  <p class="subtitle" style="margin-top:-4px">Dark = selected (holding), light = not selected. Shows how often each asset earns a slot.</p>
  <div id="selection-heatmap" class="plot"></div>

  <!-- Correlation + current snapshot side by side -->
  <div class="grid2" style="margin-top:24px">
    <div>
      <h3>Current Momentum Snapshot</h3>
      <p class="subtitle" style="margin-top:-4px">Today's 90-day return. Top 3 are highlighted.</p>
      <div id="momentum-bar" class="plot-short"></div>
    </div>
    <div>
      <h3>Asset Correlation Matrix</h3>
      <p class="subtitle" style="margin-top:-4px">Daily return correlations over the past year. High correlation = assets move together.</p>
      <div id="correlation-plot" class="plot-short"></div>
    </div>
  </div>
</div>

<!-- TECHNICALS PER STRATEGY -->
<div class="card">
  <h2>Technical Analysis — Current Holdings</h2>
  <div class="tabs" id="tech-tabs">
    <button class="tab active" onclick="showTab('tech','daily',this)">Daily</button>
    <button class="tab" onclick="showTab('tech','weekly',this)">Weekly</button>
    <button class="tab" onclick="showTab('tech','monthly',this)">Monthly</button>
  </div>
  <div id="tech-daily"></div>
  <div id="tech-weekly" class="hidden"></div>
  <div id="tech-monthly" class="hidden"></div>
</div>

<!-- TRADE LOG -->
<div class="card">
  <h2>Recent Trades</h2>
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

const ASSET_COLORS = ['#6366f1','#22c55e','#f59e0b','#ef4444','#38bdf8','#a78bfa','#fb923c'];
const STRATEGY_COLORS = { daily: '#6366f1', weekly: '#22c55e', monthly: '#f59e0b' };
const BASE_LAYOUT = {
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

// ── Equity curves ──────────────────────────────────────────────────────────
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
  Plotly.newPlot('equity-plot', traces,
    { ...BASE_LAYOUT, yaxis: { ...BASE_LAYOUT.yaxis, tickprefix: '$' } }, CFG);
})();

// ── Momentum score history ─────────────────────────────────────────────────
(function() {
  const mh = DATA.momentum_history;
  const traces = Object.entries(mh.series).map(([ticker, vals], i) => ({
    x: mh.dates, y: vals, name: ticker,
    line: { color: ASSET_COLORS[i % ASSET_COLORS.length], width: 1.5 },
    type: 'scatter',
    hovertemplate: `${ticker}: %{y:.1f}%<extra></extra>`,
  }));
  // Zero line
  const shapes = [{ type: 'line', x0: mh.dates[0], x1: mh.dates[mh.dates.length-1],
                    y0: 0, y1: 0, line: { color: '#475569', width: 1, dash: 'dot' } }];
  Plotly.newPlot('momentum-history-plot', traces,
    { ...BASE_LAYOUT, yaxis: { ...BASE_LAYOUT.yaxis, ticksuffix: '%' }, shapes }, CFG);
})();

// ── Selection heatmap ──────────────────────────────────────────────────────
(function() {
  const sh = DATA.selection_history;
  if (!sh.dates.length) return;
  const z = sh.assets.map(a => sh.matrix[a]);
  Plotly.newPlot('selection-heatmap', [{
    type: 'heatmap',
    x: sh.dates, y: sh.assets, z: z,
    colorscale: [[0, '#1a1d27'], [1, '#6366f1']],
    showscale: false,
    hovertemplate: '%{y} on %{x}: %{z}<extra></extra>',
  }], {
    ...BASE_LAYOUT,
    margin: { t: 5, b: 50, l: 60, r: 10 },
    yaxis: { ...BASE_LAYOUT.yaxis, autorange: 'reversed' },
  }, CFG);
})();

// ── Current momentum bar ───────────────────────────────────────────────────
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
  }], {
    ...BASE_LAYOUT,
    margin: { t: 20, b: 40, l: 45, r: 10 },
    yaxis: { ...BASE_LAYOUT.yaxis, ticksuffix: '%' },
    annotations: [{ text: '▲ Top 3 selected', x: 0.5, y: 1.05, xref: 'paper', yref: 'paper',
                    showarrow: false, font: { color: '#8892a4', size: 10 } }],
  }, CFG);
})();

// ── Correlation heatmap ────────────────────────────────────────────────────
(function() {
  const c = DATA.correlation;
  Plotly.newPlot('correlation-plot', [{
    type: 'heatmap',
    x: c.tickers, y: c.tickers, z: c.matrix,
    colorscale: 'RdBu', reversescale: true,
    zmin: -1, zmax: 1,
    text: c.matrix.map(row => row.map(v => v.toFixed(2))),
    texttemplate: '%{text}',
    showscale: false,
    hovertemplate: '%{y} vs %{x}: %{z:.2f}<extra></extra>',
  }], {
    ...BASE_LAYOUT,
    margin: { t: 5, b: 60, l: 60, r: 10 },
    xaxis: { ...BASE_LAYOUT.xaxis, tickangle: -35 },
    yaxis: { ...BASE_LAYOUT.yaxis, autorange: 'reversed' },
  }, CFG);
})();

// ── Technicals ─────────────────────────────────────────────────────────────
function buildTechnicals(strategy) {
  const container = document.getElementById(`tech-${strategy}`);
  const tickers = DATA.technicals[strategy];
  if (!tickers || !Object.keys(tickers).length) {
    container.innerHTML = '<p style="color:var(--muted);font-size:.85rem;padding:8px 0">No holdings yet.</p>';
    return;
  }
  container.innerHTML = '';
  for (const [ticker, d] of Object.entries(tickers)) {
    const wrap = document.createElement('div');
    wrap.style.marginBottom = '28px';

    const t1 = document.createElement('h3');
    t1.textContent = `${ticker} — Price & SMAs`;
    wrap.appendChild(t1);

    const priceDiv = document.createElement('div');
    priceDiv.id = `price-${strategy}-${ticker}`;
    priceDiv.className = 'plot';
    wrap.appendChild(priceDiv);

    const t2 = document.createElement('h3');
    t2.textContent = `${ticker} — RSI (14)`;
    t2.style.marginTop = '14px';
    wrap.appendChild(t2);

    const rsiDiv = document.createElement('div');
    rsiDiv.id = `rsi-${strategy}-${ticker}`;
    rsiDiv.style.height = '160px';
    wrap.appendChild(rsiDiv);

    container.appendChild(wrap);

    Plotly.newPlot(priceDiv.id, [
      { x: d.dates, y: d.close,  name: 'Price',   line: { color: '#e2e8f0', width: 1.5 }, type: 'scatter' },
      { x: d.dates, y: d.sma20,  name: 'SMA 20',  line: { color: '#6366f1', width: 1.2, dash:'dot' }, type: 'scatter' },
      { x: d.dates, y: d.sma50,  name: 'SMA 50',  line: { color: '#22c55e', width: 1.2, dash:'dot' }, type: 'scatter' },
      { x: d.dates, y: d.sma200, name: 'SMA 200', line: { color: '#f59e0b', width: 1.2, dash:'dot' }, type: 'scatter' },
    ], { ...BASE_LAYOUT, yaxis: { ...BASE_LAYOUT.yaxis, tickprefix: '$' } }, CFG);

    const mkLine = (y0, color) => ({
      type: 'line', x0: d.dates[0], x1: d.dates[d.dates.length-1],
      y0, y1: y0, line: { color, width: 1, dash: 'dash' }
    });
    Plotly.newPlot(rsiDiv.id, [{
      x: d.dates, y: d.rsi14, name: 'RSI 14',
      line: { color: '#a78bfa', width: 1.5 }, type: 'scatter',
      fill: 'tozeroy', fillcolor: 'rgba(167,139,250,0.08)',
    }], {
      ...BASE_LAYOUT,
      shapes: [mkLine(70, '#ef4444'), mkLine(30, '#22c55e')],
      yaxis: { ...BASE_LAYOUT.yaxis, range: [0, 100], tickvals: [0,30,50,70,100] },
      margin: { t: 5, b: 40, l: 45, r: 10 },
    }, CFG);
  }
}
['daily','weekly','monthly'].forEach(buildTechnicals);

// ── Trade log ──────────────────────────────────────────────────────────────
function buildTrades(strategy) {
  const container = document.getElementById(`trades-${strategy}`);
  const rows = DATA.trades[strategy];
  if (!rows?.length) {
    container.innerHTML = '<p style="color:var(--muted);font-size:.85rem;padding:8px 0">No trades yet.</p>';
    return;
  }
  container.innerHTML = `<table>
    <thead><tr><th>Date</th><th>Ticker</th><th>Action</th><th>Shares</th><th>Price</th><th>Value</th></tr></thead>
    <tbody>${rows.map(r => `<tr>
      <td>${String(r.date).slice(0,10)}</td>
      <td><strong>${r.ticker}</strong></td>
      <td class="${String(r.action).toLowerCase()}">${r.action}</td>
      <td>${Number(r.shares).toFixed(4)}</td>
      <td>$${Number(r.price).toFixed(2)}</td>
      <td>$${Number(r.value).toFixed(2)}</td>
    </tr>`).join('')}</tbody>
  </table>`;
}
['daily','weekly','monthly'].forEach(buildTrades);

// ── Tab switching ──────────────────────────────────────────────────────────
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
    json_str = json.dumps(data, default=str)
    html = HTML.replace("__DATA_PLACEHOLDER__", json_str)
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w") as f:
        f.write(html)
    print(f"Dashboard written to docs/index.html ({len(html)//1024}KB)")

if __name__ == "__main__":
    generate()
