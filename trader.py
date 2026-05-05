import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import os
import smtplib
import matplotlib.pyplot as plt
from email.mime.text import MIMEText

# ======================
# CONFIG
# ======================
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
LOOKBACK = 90
TOP_N = 3
REBALANCE_FREQ = "M"
FEE = 0.001
BENCHMARK = "SPY"

# ======================
# EMAIL
# ======================
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT") or 587)
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO = os.getenv("EMAIL_TO")

# ======================
# HELPERS
# ======================
def send_email(subject, body):
    if not EMAIL_HOST:
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)

def load_csv(file, columns):
    try:
        return pd.read_csv(file)
    except:
        return pd.DataFrame(columns=columns)

def save_csv(df, file):
    df.to_csv(file, index=False)

def is_rebalance_day():
    today = datetime.datetime.utcnow().date()
    return today.day == 28  # simple but safe

# ======================
# DATA
# ======================
def get_prices(tickers):
    data = yf.download(tickers, period="1y")["Adj Close"]
    return data

# ======================
# STRATEGY
# ======================
def get_top_assets(prices):
    returns = prices.pct_change(LOOKBACK).iloc[-1]
    return list(returns.sort_values(ascending=False).head(TOP_N).index)

# ======================
# PORTFOLIO
# ======================
def portfolio_value(portfolio, prices):
    total = 0
    for _, row in portfolio.iterrows():
        if row["ticker"] == "CASH":
            total += row["shares"]
        else:
            total += row["shares"] * prices[row["ticker"]].iloc[-1]
    return total

# ======================
# TRADE EXECUTION
# ======================
def execute_trades(portfolio, target_assets, prices, total_value):
    trades = []
    portfolio_dict = {row["ticker"]: row["shares"] for _, row in portfolio.iterrows()}

    cash = portfolio_dict.get("CASH", 0)

    target_weight = 1 / len(target_assets)

    # SELL everything not in target
    for ticker, shares in portfolio_dict.items():
        if ticker != "CASH" and ticker not in target_assets:
            price = prices[ticker].iloc[-1]
            value = shares * price
            cash += value * (1 - FEE)

            trades.append((ticker, "SELL", shares, price, value))

            portfolio_dict[ticker] = 0

    # BUY target assets
    for ticker in target_assets:
        price = prices[ticker].iloc[-1]
        target_value = total_value * target_weight

        current_shares = portfolio_dict.get(ticker, 0)
        current_value = current_shares * price

        diff_value = target_value - current_value

        if abs(diff_value) < 10:
            continue

        shares = diff_value / price

        if shares > 0:
            cost = shares * price * (1 + FEE)
            if cost > cash:
                continue
            cash -= cost
            action = "BUY"
        else:
            cash += abs(shares * price) * (1 - FEE)
            action = "SELL"

        portfolio_dict[ticker] = current_shares + shares
        trades.append((ticker, action, shares, price, abs(shares * price)))

    portfolio_dict["CASH"] = cash

    new_portfolio = pd.DataFrame([
        {"ticker": k, "shares": v} for k, v in portfolio_dict.items() if v > 0
    ])

    return new_portfolio, trades

# ======================
# LOGGING
# ======================
def log_trades(trades):
    df = load_csv("trade_log.csv", ["date","ticker","action","shares","price","value"])
    now = datetime.datetime.utcnow()

    for t in trades:
        new = pd.DataFrame([{
            "date": now,
            "ticker": t[0],
            "action": t[1],
            "shares": t[2],
            "price": t[3],
            "value": t[4]
        }])
        df = pd.concat([df, new])

    save_csv(df, "trade_log.csv")

def log_performance(value, benchmark_price):
    df = load_csv("performance_log.csv", ["date","portfolio_value","benchmark_value"])
    now = datetime.datetime.utcnow()

    new = pd.DataFrame([{
        "date": now,
        "portfolio_value": value,
        "benchmark_value": benchmark_price
    }])

    df = pd.concat([df, new])
    save_csv(df, "performance_log.csv")

# ======================
# PLOTTING
# ======================
def plot_performance():
    df = pd.read_csv("performance_log.csv")
    df["date"] = pd.to_datetime(df["date"])

    plt.figure()
    plt.plot(df["date"], df["portfolio_value"], label="Strategy")
    plt.plot(df["date"], df["benchmark_value"], label="Benchmark")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/equity_curve.png")
    plt.close()

# ======================
# MAIN
# ======================
def main():
    prices = get_prices(TICKERS + [BENCHMARK])
    portfolio = load_csv("paper_portfolio.csv", ["ticker","shares"])

    top_assets = get_top_assets(prices[TICKERS])

    total_value = portfolio_value(portfolio, prices)
    benchmark_price = prices[BENCHMARK].iloc[-1]

    report = []
    report.append(f"Top assets: {top_assets}")
    report.append(f"Portfolio value: {round(total_value,2)}")

    log_performance(total_value, benchmark_price)
    plot_performance()

    if not is_rebalance_day():
        report.append("No rebalance today.")
        send_email("Daily Update", "\n".join(report))
        return

    new_portfolio, trades = execute_trades(portfolio, top_assets, prices, total_value)

    save_csv(new_portfolio, "paper_portfolio.csv")
    log_trades(trades)

    report.append("Trades executed:")
    for t in trades:
        report.append(str(t))

    send_email("Rebalance Executed", "\n".join(report))

if __name__ == "__main__":
    main()
