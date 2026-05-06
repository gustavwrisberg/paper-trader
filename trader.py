import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import os
import smtplib
import matplotlib.pyplot as plt
import requests
from email.mime.text import MIMEText

# ======================
# CONFIG
# ======================
STOCK_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
BENCHMARK = "SPY"
LOOKBACK = 90
TOP_N = 3

# Strategy is set via env var: "daily", "weekly", or "monthly"
STRATEGY = os.getenv("STRATEGY", "monthly").lower()

PORTFOLIO_FILE   = f"paper_portfolio_{STRATEGY}.csv"
PERFORMANCE_FILE = f"performance_log_{STRATEGY}.csv"
TRADE_LOG_FILE   = f"trade_log_{STRATEGY}.csv"

# ======================
# EMAIL
# ======================
EMAIL_HOST     = os.getenv("EMAIL_HOST")
EMAIL_PORT     = int(os.getenv("EMAIL_PORT") or 587)
EMAIL_USER     = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_FROM     = os.getenv("EMAIL_FROM")
EMAIL_TO       = os.getenv("EMAIL_TO")

# ======================
# REBALANCE SCHEDULE
# ======================
def is_rebalance_day():
    today = datetime.datetime.utcnow().date()
    if STRATEGY == "daily":
        return True
    elif STRATEGY == "weekly":
        return today.weekday() == 0  # Monday
    elif STRATEGY == "monthly":
        return today.day >= 28
    return False

# ======================
# PRICE FETCHING
# ======================
def get_stock_prices(tickers):
    data = yf.download(tickers, period="1y", auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data

def get_btc_price_series():
    """Fetch BTC/USD OHLC from Kraken public API (no key needed)."""
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": "XBTUSD", "interval": 1440}  # daily candles
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise ValueError(f"Kraken error: {data['error']}")
        # Result key is dynamic, grab first non-'last' key
        pair_key = [k for k in data["result"] if k != "last"][0]
        rows = data["result"][pair_key]
        # columns: time, open, high, low, close, vwap, volume, count
        df = pd.DataFrame(rows, columns=["time","open","high","low","close","vwap","volume","count"])
        df["date"] = pd.to_datetime(df["time"], unit="s").dt.normalize()
        df = df.set_index("date")[["close"]].rename(columns={"close": "BTC"})
        df["BTC"] = df["BTC"].astype(float)
        return df
    except Exception as e:
        print(f"Warning: Could not fetch BTC price from Kraken: {e}")
        return None

def get_all_prices():
    stock_prices = get_stock_prices(STOCK_TICKERS)
    stock_prices.index = pd.to_datetime(stock_prices.index).normalize()

    btc_series = get_btc_price_series()
    if btc_series is not None:
        prices = stock_prices.join(btc_series, how="left")
        prices["BTC"] = prices["BTC"].ffill()
    else:
        prices = stock_prices

    return prices

def get_benchmark_value(initial_cash, prices):
    series = prices[BENCHMARK].dropna()
    if len(series) < 2:
        return initial_cash
    return initial_cash * (series.iloc[-1] / series.iloc[0])

# ======================
# PORTFOLIO HELPERS
# ======================
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        return pd.read_csv(PORTFOLIO_FILE)
    df = pd.DataFrame([{"ticker": "CASH", "shares": 10000.0}])
    df.to_csv(PORTFOLIO_FILE, index=False)
    return df

def save_portfolio(df):
    df.to_csv(PORTFOLIO_FILE, index=False)

def get_portfolio_value(portfolio, prices):
    total = 0.0
    latest = prices.iloc[-1]
    for _, row in portfolio.iterrows():
        ticker = row["ticker"]
        if ticker == "CASH":
            total += float(row["shares"])
        elif ticker in latest.index and not pd.isna(latest[ticker]):
            total += float(row["shares"]) * float(latest[ticker])
        else:
            print(f"Warning: {ticker} not found in price data, skipping.")
    return total

# ======================
# STRATEGY
# ======================
def get_top_assets(prices):
    returns = prices.pct_change(LOOKBACK).iloc[-1]
    valid = returns.dropna().sort_values(ascending=False)
    return list(valid.head(TOP_N).index), {t: (round(v*100,1), i+1) for i, (t,v) in enumerate(valid.items())}

# ======================
# PERFORMANCE TRACKING
# ======================
def update_performance(portfolio_value, benchmark_value):
    today = datetime.datetime.utcnow().date()

    if os.path.exists(PERFORMANCE_FILE):
        df = pd.read_csv(PERFORMANCE_FILE)
        if "benchmark_value" not in df.columns:
            df["benchmark_value"] = np.nan
    else:
        df = pd.DataFrame(columns=["date", "portfolio_value", "benchmark_value"])

    new_row = pd.DataFrame([{
        "date": today,
        "portfolio_value": round(portfolio_value, 2),
        "benchmark_value": round(benchmark_value, 2),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(PERFORMANCE_FILE, index=False)
    return df

def plot_performance(df):
    df["date"] = pd.to_datetime(df["date"])

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["portfolio_value"], label=f"Portfolio ({STRATEGY})")
    if "benchmark_value" in df.columns and df["benchmark_value"].notna().any():
        plt.plot(df["date"], df["benchmark_value"], label=f"Benchmark ({BENCHMARK})", linestyle="--")
    plt.title(f"Portfolio Value — {STRATEGY.capitalize()} Strategy")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/equity_curve_{STRATEGY}.png")
    plt.close()

def log_trade(ticker, action, shares, price, reason=""):
    today = datetime.datetime.utcnow().date()
    row = pd.DataFrame([{
        "date": today,
        "ticker": ticker,
        "action": action,
        "shares": round(shares, 6),
        "price": round(price, 4),
        "value": round(shares * price, 2),
        "reason": reason,
    }])
    if os.path.exists(TRADE_LOG_FILE):
        existing = pd.read_csv(TRADE_LOG_FILE)
        if "reason" not in existing.columns:
            existing["reason"] = ""
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(TRADE_LOG_FILE, index=False)

# ======================
# MAIN
# ======================
def main():
    print(f"Running strategy: {STRATEGY}")

    prices = get_all_prices()
    benchmark_prices = get_stock_prices([BENCHMARK])
    benchmark_prices.index = pd.to_datetime(benchmark_prices.index).normalize()

    portfolio = load_portfolio()
    top_assets, momentum_ranks = get_top_assets(prices)
    report = [f"Strategy: {STRATEGY}", f"Top assets: {top_assets}"]

    total_value = get_portfolio_value(portfolio, prices)
    report.append(f"Portfolio value: ${round(total_value, 2)}")

    benchmark_value = get_benchmark_value(10000, benchmark_prices)
    report.append(f"Benchmark ({BENCHMARK}) value: ${round(benchmark_value, 2)}")

    perf_df = update_performance(total_value, benchmark_value)
    plot_performance(perf_df)

    if not is_rebalance_day():
        report.append("No rebalance today.")
        send_email(f"[{STRATEGY}] Daily Update", "\n".join(report))
        print("\n".join(report))
        return

    # REBALANCE
    target_weight = 1 / len(top_assets)
    new_rows = []
    cash = total_value
    latest = prices.iloc[-1]

    old_holdings = {r["ticker"]: r["shares"] for _, r in portfolio.iterrows()}

    for ticker in top_assets:
        price = float(latest[ticker])
        target_value = total_value * target_weight
        shares = target_value / price
        mom_pct, rank = momentum_ranks[ticker]
        mom_str = f"{'+' if mom_pct >= 0 else ''}{mom_pct}% {LOOKBACK}d momentum, ranked #{rank}"

        # Log sells for positions being exited
        if ticker in old_holdings:
            old_shares = old_holdings[ticker]
            if abs(old_shares - shares) > 0.0001:
                if old_shares > shares:
                    log_trade(ticker, "SELL", old_shares - shares, price, f"Rebalance — trim to equal weight ({mom_str})")
                else:
                    log_trade(ticker, "BUY", shares - old_shares, price, f"Rebalance — top {rank} momentum ({mom_str})")
        else:
            log_trade(ticker, "BUY", shares, price, f"Entered — top {rank} momentum ({mom_str})")

        new_rows.append({"ticker": ticker, "shares": round(shares, 6)})
        cash -= shares * price

    # Log full exits
    for ticker, old_shares in old_holdings.items():
        if ticker not in top_assets and ticker != "CASH":
            price = float(latest[ticker]) if ticker in latest.index else 0
            if ticker in momentum_ranks:
                mom_pct, rank = momentum_ranks[ticker]
                reason = f"Exited — fell out of top {TOP_N} (ranked #{rank}, {'+' if mom_pct >= 0 else ''}{mom_pct}% {LOOKBACK}d momentum)"
            else:
                reason = f"Exited — no momentum data"
            log_trade(ticker, "SELL", old_shares, price, reason)

    new_rows.append({"ticker": "CASH", "shares": round(cash, 2)})
    new_portfolio = pd.DataFrame(new_rows)
    save_portfolio(new_portfolio)

    report.append("Rebalanced portfolio:")
    report.append(str(new_portfolio))

    send_email(f"[{STRATEGY}] Rebalance Executed", "\n".join(report))
    print("\n".join(report))


def send_email(subject, body):
    if not EMAIL_HOST:
        print("No EMAIL_HOST set, skipping email.")
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)


if __name__ == "__main__":
    main()
