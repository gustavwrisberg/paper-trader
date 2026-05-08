import os
import datetime
import smtplib
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt

# Stocks and ETFs to rank and trade
STOCK_TICKERS = [
    # Original US
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    # Global ETFs (all USD-listed)
    "URTH",  # MSCI World
    "EEM",   # MSCI Emerging Markets
    "IEV",   # STOXX Europe 600 (replaces EXSA.DE)
    "EWJ",   # Japan / TOPIX proxy (replaces 1306.T)
    "SCZ",   # MSCI EAFE Small Cap (replaces ISWC)
    # International stocks (USD-listed)
    "TSM",   # TSMC
    "ASML",  # ASML
    "NVO",   # Novo Nordisk
    "MHVYF", # Mitsubishi Heavy Industries (USD OTC)
    "SIEGY", # Siemens ADR (replaces SIE.DE)
    "SBGSY", # Schneider Electric ADR (replaces SU.PA)
    "BN",    # Brookfield Corporation
    "SHEL",  # Shell
    "BRK-B", # Berkshire Hathaway
]

BENCHMARK = "SPY"
LOOKBACK  = 90
TOP_N     = 3

# Alert thresholds (applied to current holdings only)
ALERT_DROP_1D  = -0.03   # -3% in one day
ALERT_RSI_LOW  = 35      # RSI below this = oversold
ALERT_SMA_BREAK = True   # alert when price crosses below 50d SMA

STRATEGY       = os.getenv("STRATEGY", "monthly").lower()
PORTFOLIO_FILE = f"paper_portfolio_{STRATEGY}.csv"
PERF_FILE      = f"performance_log_{STRATEGY}.csv"
TRADE_FILE     = f"trade_log_{STRATEGY}.csv"

EMAIL_HOST     = os.getenv("EMAIL_HOST")
EMAIL_PORT     = int(os.getenv("EMAIL_PORT") or 587)
EMAIL_USER     = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_FROM     = os.getenv("EMAIL_FROM")
EMAIL_TO       = os.getenv("EMAIL_TO")

TWILIO_SID   = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM  = os.getenv("TWILIO_FROM")  # your Twilio number e.g. +15551234567
TWILIO_TO    = os.getenv("TWILIO_TO")    # your number e.g. +4512345678


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


def rsi(series, period=14):
    d = series.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


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


def check_alerts(portfolio, prices):
    """
    Check held positions for crash signals.
    Returns a list of alert strings, empty if nothing triggered.
    """
    held = [r["ticker"] for _, r in portfolio.iterrows() if r["ticker"] != "CASH"]
    alerts = []

    for t in held:
        if t not in prices.columns:
            continue
        series = prices[t].dropna()
        if len(series) < 55:
            continue

        latest  = series.iloc[-1]
        prev    = series.iloc[-2]
        sma50   = series.rolling(50).mean().iloc[-1]
        rsi_val = rsi(series).iloc[-1]

        reasons = []

        change_1d = (latest - prev) / prev
        if change_1d <= ALERT_DROP_1D:
            reasons.append(f"down {change_1d*100:.1f}% today")

        if not np.isnan(rsi_val) and rsi_val < ALERT_RSI_LOW:
            reasons.append(f"RSI {rsi_val:.0f} (oversold)")

        if ALERT_SMA_BREAK and not np.isnan(sma50):
            prev_sma50 = series.rolling(50).mean().iloc[-2]
            if prev >= prev_sma50 and latest < sma50:
                reasons.append(f"broke below 50d SMA (${sma50:.2f})")

        if reasons:
            alerts.append(f"{t}: {', '.join(reasons)}")

    return alerts


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
    if os.path.exists(TRADE_FILE):
        existing = pd.read_csv(TRADE_FILE)
        if "reason" not in existing.columns:
            existing["reason"] = ""
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(TRADE_FILE, index=False)


def update_performance(port_val, bench_val):
    today = datetime.datetime.utcnow().date()
    if os.path.exists(PERF_FILE):
        df = pd.read_csv(PERF_FILE)
        if "benchmark_value" not in df.columns:
            df["benchmark_value"] = np.nan
    else:
        df = pd.DataFrame(columns=["date", "portfolio_value", "benchmark_value"])

    df = pd.concat([df, pd.DataFrame([{
        "date":            today,
        "portfolio_value": round(port_val, 2),
        "benchmark_value": round(bench_val, 2),
    }])], ignore_index=True)
    df.to_csv(PERF_FILE, index=False)
    return df


def plot_equity(df):
    df["date"] = pd.to_datetime(df["date"])
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["portfolio_value"], label=f"Portfolio ({STRATEGY})")
    if df["benchmark_value"].notna().any():
        plt.plot(df["date"], df["benchmark_value"], label=BENCHMARK, linestyle="--")
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


def send_sms(body):
    if not TWILIO_SID:
        print("No TWILIO_SID, skipping SMS.")
        return
    try:
        resp = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
            auth=(TWILIO_SID, TWILIO_TOKEN),
            data={"From": TWILIO_FROM, "To": TWILIO_TO, "Body": body},
            timeout=10,
        )
        resp.raise_for_status()
        print(f"SMS sent: {resp.json().get('sid')}")
    except Exception as e:
        print(f"SMS failed: {e}")


def main():
    print(f"Strategy: {STRATEGY}")

    prices      = fetch_all()
    bench_prices = fetch_stocks([BENCHMARK])

    portfolio   = load_portfolio()
    top, ranks  = rank_assets(prices)

    val         = portfolio_value(portfolio, prices)
    bench_start = bench_prices.iloc[0][BENCHMARK]
    bench_end   = bench_prices.iloc[-1][BENCHMARK]
    bench_val   = 10000 * (bench_end / bench_start)

    perf_df = update_performance(val, bench_val)
    plot_equity(perf_df)

    # ── Crash alerts ──────────────────────────────────────────────────────────
    alerts = check_alerts(portfolio, prices)
    if alerts:
        alert_text = f"[{STRATEGY.upper()}] ALERT — {datetime.datetime.utcnow().strftime('%H:%M UTC')}\n\n"
        alert_text += "\n".join(f"⚠ {a}" for a in alerts)
        print(alert_text)
        send_email(f"[{STRATEGY}] ⚠ crash alert", alert_text)
        send_sms(alert_text[:1600])  # Twilio cap

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

    # ── Rebalance ─────────────────────────────────────────────────────────────
    weight  = 1 / len(top)
    latest  = prices.iloc[-1]
    old     = {r["ticker"]: r["shares"] for _, r in portfolio.iterrows()}
    new_rows = []
    cash    = val

    for t in top:
        price  = float(latest[t])
        shares = (val * weight) / price
        pct, rank = ranks[t]
        sign = "+" if pct >= 0 else ""
        mom  = f"{sign}{pct}% {LOOKBACK}d momentum, rank #{rank}"

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
