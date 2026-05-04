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

# ======================
# EMAIL
# ======================
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO = os.getenv("EMAIL_TO")

# ======================
# HELPERS
# ======================
def is_rebalance_day():
    today = datetime.datetime.utcnow().date()
    return today.day >= 28

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

def load_portfolio():
    return pd.read_csv("paper_portfolio.csv")

def save_portfolio(df):
    df.to_csv("paper_portfolio.csv", index=False)

def get_prices(tickers):
    return yf.download(tickers, period="1y")["Adj Close"]

# ======================
# STRATEGY
# ======================
def get_top_assets(prices):
    returns = prices.pct_change(LOOKBACK).iloc[-1]
    return list(returns.sort_values(ascending=False).head(TOP_N).index)

def get_portfolio_value(portfolio, prices):
    total = 0
    for _, row in portfolio.iterrows():
        if row["ticker"] == "CASH":
            total += row["shares"]
        else:
            total += row["shares"] * prices[row["ticker"]].iloc[-1]
    return total

# ======================
# PERFORMANCE TRACKING
# ======================
def update_performance(value):
    today = datetime.datetime.utcnow().date()

    try:
        df = pd.read_csv("performance_log.csv")
    except:
        df = pd.DataFrame(columns=["date", "portfolio_value"])

    df = pd.concat([
        df,
        pd.DataFrame([{
            "date": today,
            "portfolio_value": value
        }])
    ], ignore_index=True)

    df.to_csv("performance_log.csv", index=False)
    return df

def plot_performance(df):
    df["date"] = pd.to_datetime(df["date"])

    plt.figure()
    plt.plot(df["date"], df["portfolio_value"])
    plt.title("Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/equity_curve.png")
    plt.close()

# ======================
# MAIN
# ======================
def main():
    prices = get_prices(TICKERS)
    portfolio = load_portfolio()

    top_assets = get_top_assets(prices)
    report = []
    report.append(f"Top assets: {top_assets}")

    total_value = get_portfolio_value(portfolio, prices)
    report.append(f"Portfolio value: {round(total_value,2)}")

    # update performance log
    perf_df = update_performance(total_value)
    plot_performance(perf_df)

    if not is_rebalance_day():
        report.append("No rebalance today.")
        send_email("Daily Update", "\n".join(report))
        return

    # REBALANCE
    target_weight = 1 / len(top_assets)
    new_rows = []
    cash = total_value

    for ticker in top_assets:
        price = prices[ticker].iloc[-1]
        target_value = total_value * target_weight
        shares = target_value / price

        new_rows.append({"ticker": ticker, "shares": shares})
        cash -= shares * price

    new_rows.append({"ticker": "CASH", "shares": cash})
    new_portfolio = pd.DataFrame(new_rows)

    save_portfolio(new_portfolio)

    report.append("Rebalanced portfolio:")
    report.append(str(new_portfolio))

    send_email("Rebalance Executed", "\n".join(report))


if __name__ == "__main__":
    main()
