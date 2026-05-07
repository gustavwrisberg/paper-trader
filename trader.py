import os
import datetime
import smtplib
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt

STOCK_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
BENCHMARK = "SPY"
LOOKBACK = 90
TOP_N = 3

STRATEGY = os.getenv("STRATEGY", "monthly").lower()
PORTFOLIO_FILE   = f"paper_portfolio_{STRATEGY}.csv"
PERFORMANCE_FILE = f"performance_log_{STRATEGY}.csv"
TRADE_LOG_FILE   = f"trade_log_{STRATEGY}.csv"

EMAIL_HOST     = os.getenv("EMAIL_HOST")
EMAIL_PORT     = int(os.getenv("EMAIL_PORT") or 587)
EMAIL_USER     = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_FROM     = os.getenv("EMAIL_FROM")
EMAIL_TO       = os.getenv("EMAIL_TO")


def rebalance_today():
    day = datetime.datetime.utcnow().date()
    if STRATEGY == "daily":   return True
    if STRATEGY == "weekly":  return day.weekday() == 0
    if STRATEGY == "monthly": return day.day >= 28
    return False


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
        resp.raise_for_status()
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


def fetch_all():
    prices = fetch_stocks(STOCK_TICKERS)
    btc = fetch_btc()
    if btc is not None:
        prices = prices.join(btc, how="left")
        prices["BTC"] = prices["BTC"].ffill()
    return prices


def rank_assets(prices):
    ret = prices.pct_change(LOOKBACK).iloc[-1].dropna().sort_values(ascending=False)
    ranks = {t: (round(v * 100, 1), i + 1) for i, (t, v) in enumerate(ret.items())}
    return list(ret.head(TOP_N).index), ranks


def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        return pd.read_csv(PORTFOLIO_FILE)
    df = pd.DataFrame([{"ticker": "CASH", "shares": 10000.0}])
    df.to_csv(PORTFOLIO_FILE, index=False)
    return df


def portfolio_value(portfolio, prices):
    latest = prices.iloc[-1]
    total = 0.0
    for _, row in portfolio.iterrows():
        t = row["ticker"]
        if t == "CASH":
            total += float(row["shares"])
        elif t in latest.index and not pd.isna(latest[t]):
            total += float(row["shares"]) * float(latest[t])
        else:
            print(f"Warning: {t} missing from price data")
    return total


def log_trade(ticker, action, shares, price, reason=""):
    row = pd.DataFrame([{
        "date":   datetime.datetime.utcnow().date(),
        "ticker": ticker,
        "action": action,
        "shares": round(shares, 6),
        "price":  round(price, 4),
        "value":  round(shares * price, 2),
        "reason": reason,
    }])
    if os.path.exists(TRADE_LOG_FILE):
        existing = pd.read_csv(TRADE_LOG_FILE)
        if "reason" not in existing.columns:
            existing["reason"] = ""
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(TRADE_LOG_FILE, index=False)


def update_performance(port_val, bench_val):
    today = datetime.datetime.utcnow().date()
    if os.path.exists(PERFORMANCE_FILE):
        df = pd.read_csv(PERFORMANCE_FILE)
        if "benchmark_value" not in df.columns:
            df["benchmark_value"] = np.nan
    else:
        df = pd.DataFrame(columns=["date", "portfolio_value", "benchmark_value"])

    df = pd.concat([df, pd.DataFrame([{
        "date": today,
        "portfolio_value": round(port_val, 2),
        "benchmark_value": round(bench_val, 2),
    }])], ignore_index=True)
    df.to_csv(PERFORMANCE_FILE, index=False)
    return df


def plot_equity(df):
    df["date"] = pd.to_datetime(df["date"])
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["portfolio_value"], label=f"Portfolio ({STRATEGY})")
    if df["benchmark_value"].notna().any():
        plt.plot(df["date"], df["benchmark_value"], label=f"{BENCHMARK}", linestyle="--")
    plt.title(f"{STRATEGY.capitalize()} strategy")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/equity_curve_{STRATEGY}.png")
    plt.close()


def send_email(subject, body):
    if not EMAIL_HOST:
        print("No EMAIL_HOST, skipping email.")
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as s:
        s.starttls()
        s.login(EMAIL_USER, EMAIL_PASSWORD)
        s.send_message(msg)


def main():
    print(f"Strategy: {STRATEGY}")

    prices = fetch_all()
    bench_prices = fetch_stocks([BENCHMARK])

    portfolio = load_portfolio()
    top, ranks = rank_assets(prices)

    val = portfolio_value(portfolio, prices)
    bench_start = bench_prices.iloc[0][BENCHMARK]
    bench_end   = bench_prices.iloc[-1][BENCHMARK]
    bench_val   = 10000 * (bench_end / bench_start)

    perf_df = update_performance(val, bench_val)
    plot_equity(perf_df)

    lines = [
        f"Strategy: {STRATEGY}",
        f"Top assets: {top}",
        f"Portfolio: ${val:.2f}",
        f"Benchmark ({BENCHMARK}): ${bench_val:.2f}",
    ]

    if not rebalance_today():
        lines.append("No rebalance.")
        send_email(f"[{STRATEGY}] update", "\n".join(lines))
        print("\n".join(lines))
        return

    weight = 1 / len(top)
    latest = prices.iloc[-1]
    old = {r["ticker"]: r["shares"] for _, r in portfolio.iterrows()}
    new_rows = []
    cash = val

    for t in top:
        price = float(latest[t])
        shares = (val * weight) / price
        pct, rank = ranks[t]
        sign = "+" if pct >= 0 else ""
        mom = f"{sign}{pct}% {LOOKBACK}d momentum, rank #{rank}"

        if t in old:
            diff = old[t] - shares
            if abs(diff) > 0.0001:
                if diff > 0:
                    log_trade(t, "SELL", diff, price, f"Trim to equal weight ({mom})")
                else:
                    log_trade(t, "BUY", -diff, price, f"Add to position — rank #{rank} ({mom})")
        else:
            log_trade(t, "BUY", shares, price, f"New entry — rank #{rank} ({mom})")

        new_rows.append({"ticker": t, "shares": round(shares, 6)})
        cash -= shares * price

    for t, s in old.items():
        if t not in top and t != "CASH":
            price = float(latest[t]) if t in latest.index else 0
            pct, rank = ranks.get(t, (0, "?"))
            sign = "+" if pct >= 0 else ""
            log_trade(t, "SELL", s, price, f"Dropped from top {TOP_N} (rank #{rank}, {sign}{pct}%)")

    new_rows.append({"ticker": "CASH", "shares": round(cash, 2)})
    pd.DataFrame(new_rows).to_csv(PORTFOLIO_FILE, index=False)

    lines.append("Rebalanced:\n" + str(pd.DataFrame(new_rows)))
    send_email(f"[{STRATEGY}] rebalance", "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
