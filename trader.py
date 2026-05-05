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
BENCHMARK = "SPY"
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

def load_portfolio():
    return pd.read_csv("paper_portfolio.csv")

def save_portfolio(df):
    df.to_csv("paper_portfolio.csv", index=False)

def get_prices(tickers):
    # yfinance 0.2.x+ uses "Close" instead of "Adj Close"
    data = yf.download(tickers, period="1y", auto_adjust=True)["Close"]
    # If only one ticker, yfinance returns a Series — normalize to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data

def get_benchmark_value(initial_cash, prices_series):
    """Return current value of a buy-and-hold benchmark starting with initial_cash."""
    first_valid = prices_series.dropna().iloc[0]
    last_valid = prices_series.dropna().iloc[-1]
    return initial_cash * (last_valid / first_valid)

# ======================
# STRATEGY
# ======================
def get_top_assets(prices):
    returns = prices.pct_change(LOOKBACK).iloc[-1]
    valid = returns.dropna()
    return list(valid.sort_values(ascending=False).head(TOP_N).index)

def get_portfolio_value(portfolio, prices):
    total = 0
    for _, row in portfolio.iterrows():
        if row["ticker"] == "CASH":
            total += float(row["shares"])
        elif row["ticker"] in prices.columns:
            price = prices[row["ticker"]].dropna().iloc[-1]
            total += float(row["shares"]) * float(price)
        else:
            print(f"Warning: {row['ticker']} not found in price data, skipping.")
    return total

# ======================
# PERFORMANCE TRACKING
# ======================
def update_performance(portfolio_value, benchmark_value):
    today = datetime.datetime.utcnow().date()

    try:
        df = pd.read_csv("performance_log.csv")
        # Ensure benchmark column exists (handles old logs without it)
        if "benchmark_value" not in df.columns:
            df["benchmark_value"] = np.nan
    except Exception:
        df = pd.DataFrame(columns=["date", "portfolio_value", "benchmark_value"])

    new_row = pd.DataFrame([{
        "date": today,
        "portfolio_value": round(portfolio_value, 2),
        "benchmark_value": round(benchmark_value, 2),
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv("performance_log.csv", index=False)
    return df

def plot_performance(df):
    df["date"] = pd.to_datetime(df["date"])

    plt.figure()
    plt.plot(df["date"], df["portfolio_value"], label="Portfolio")
    if "benchmark_value" in df.columns and df["benchmark_value"].notna().any():
        plt.plot(df["date"], df["benchmark_value"], label=f"Benchmark ({BENCHMARK})", linestyle="--")
    plt.title("Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
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
    prices = get_prices(TICKERS)
    benchmark_prices = get_prices([BENCHMARK])

    portfolio = load_portfolio()
    top_assets = get_top_assets(prices)
    report = []
    report.append(f"Top assets: {top_assets}")

    total_value = get_portfolio_value(portfolio, prices)
    report.append(f"Portfolio value: ${round(total_value, 2)}")

    # Benchmark: treat initial cash (10000) as buy-and-hold SPY
    initial_cash = 10000
    benchmark_value = get_benchmark_value(initial_cash, benchmark_prices[BENCHMARK])
    report.append(f"Benchmark ({BENCHMARK}) value: ${round(benchmark_value, 2)}")

    # Update performance log and chart
    perf_df = update_performance(total_value, benchmark_value)
    plot_performance(perf_df)

    if not is_rebalance_day():
        report.append("No rebalance today.")
        send_email("Daily Update", "\n".join(report))
        print("\n".join(report))
        return

    # REBALANCE
    target_weight = 1 / len(top_assets)
    new_rows = []
    cash = total_value

    for ticker in top_assets:
        price = float(prices[ticker].dropna().iloc[-1])
        target_value = total_value * target_weight
        shares = target_value / price

        new_rows.append({"ticker": ticker, "shares": round(shares, 6)})
        cash -= shares * price

    new_rows.append({"ticker": "CASH", "shares": round(cash, 2)})
    new_portfolio = pd.DataFrame(new_rows)

    save_portfolio(new_portfolio)

    report.append("Rebalanced portfolio:")
    report.append(str(new_portfolio))

    send_email("Rebalance Executed", "\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
