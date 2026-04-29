#!/usr/bin/env python3
"""
Daily portfolio allocator + Alpaca PAPER trader for the momentum + trend strategy.

What it does:
1. Downloads latest adjusted prices from Yahoo Finance.
2. Calculates target weights using the same backtest-style logic:
   - market trend filter
   - stock trend filter
   - volatility-adjusted momentum ranking
   - top-N equal-weight allocation with max weight cap
   - leftover stays in CASH
3. Reads holdings from current_holdings.csv.
4. Builds exact rebalance trades to match the target weights.
5. Only allows orders on scheduled rebalance days.
6. Can submit Alpaca PAPER market orders.
7. Can email the report when a rebalance is triggered.

Install:
    pip install yfinance pandas numpy matplotlib alpaca-py

Run monitor-only:
    python daily_portfolio_allocator_alpaca_paper.py

Run paper trading only on scheduled rebalance day:
    python daily_portfolio_allocator_alpaca_paper.py --paper-trade

Run with email alert:
    python daily_portfolio_allocator_alpaca_paper.py --email

Holdings file format: current_holdings.csv
    ticker,shares
    NVDA,5
    MSFT,3
    CASH,2500
"""

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


@dataclass
class Config:
    tickers: List[str] = None
    market_benchmark: str = "^GSPC"
    period: str = "10y"
    interval: str = "1d"
    top_n: int = 3
    max_weight_per_stock: float = 0.40
    market_sma: int = 200
    stock_sma: int = 100
    momentum_lookback: int = 126
    momentum_skip_days: int = 21
    vol_lookback: int = 63
    use_vol_adjusted_momentum: bool = True
    require_positive_momentum: bool = True
    rebalance_only_on: str = "M"  # D=daily, W=weekly, M=monthly
    min_trade_value: float = 50.0
    default_portfolio_value: float = 10_000.0
    allow_fractional_shares: bool = True
    plot_lookback_days: int = 504
    save_charts: bool = True
    alpaca_fractional: bool = True
    email_on_monitor_days: bool = False

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = ["NVDA", "AMD", "ASML", "AVGO", "MSFT", "AAPL", "AMZN", "GOOGL"]


CONFIG = Config()
HOLDINGS_FILE = Path("current_holdings.csv")
REPORT_DIR = Path("reports")


def download_prices(symbols: List[str], period: str, interval: str) -> pd.DataFrame:
    symbols = list(dict.fromkeys(symbols))
    raw = yf.download(symbols, period=period, interval=interval, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("No price data downloaded. Check tickers or internet connection.")
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": symbols[0]})
    prices = prices.dropna(how="all").ffill()
    prices.index = pd.to_datetime(prices.index)
    return prices


def calculate_scores(prices: pd.DataFrame, cfg: Config):
    stock_prices = prices[cfg.tickers].copy()
    market = prices[cfg.market_benchmark].copy()
    market_sma_line = market.rolling(cfg.market_sma).mean()
    market_ok = market > market_sma_line
    stock_sma_line = stock_prices.rolling(cfg.stock_sma).mean()
    stock_trend_ok = stock_prices > stock_sma_line
    past_price = stock_prices.shift(cfg.momentum_lookback)
    recent_price = stock_prices.shift(cfg.momentum_skip_days)
    momentum = recent_price / past_price - 1
    if cfg.use_vol_adjusted_momentum:
        volatility = stock_prices.pct_change().rolling(cfg.vol_lookback).std() * np.sqrt(252)
        score = momentum / volatility
    else:
        score = momentum
    score = score.where(stock_trend_ok)
    if cfg.require_positive_momentum:
        score = score.where(momentum > 0)
    score = score.where(market_ok, np.nan)
    return score, market_ok, momentum, stock_sma_line


def latest_target_weights(scores: pd.DataFrame, cfg: Config) -> pd.Series:
    valid_scores = scores.dropna(how="all")
    latest_date = valid_scores.index.max() if not valid_scores.empty else scores.index.max()
    latest_scores = scores.loc[latest_date].dropna().sort_values(ascending=False)
    target = pd.Series(0.0, index=cfg.tickers, dtype=float)
    if latest_scores.empty:
        return target
    selected = latest_scores.head(cfg.top_n).index.tolist()
    equal_weight = min(1.0 / len(selected), cfg.max_weight_per_stock)
    target.loc[selected] = equal_weight
    return target


def is_rebalance_day(today: pd.Timestamp, price_index: pd.DatetimeIndex, mode: str) -> bool:
    mode = mode.upper()
    if mode == "D":
        return True
    today = pd.Timestamp(today).normalize()
    idx = pd.DatetimeIndex(price_index).normalize()
    if mode == "W":
        current = idx[idx.to_period("W") == today.to_period("W")]
        return len(current) > 0 and today == current.max()
    if mode == "M":
        current = idx[idx.to_period("M") == today.to_period("M")]
        return len(current) > 0 and today == current.max()
    raise ValueError("rebalance_only_on must be D, W, or M")


def load_current_holdings(path: Path, latest_prices: pd.Series, cfg: Config) -> pd.DataFrame:
    rows = []
    if not path.exists():
        rows.append({"ticker": "CASH", "shares": cfg.default_portfolio_value, "price": 1.0, "current_value": cfg.default_portfolio_value})
        return pd.DataFrame(rows)
    holdings = pd.read_csv(path)
    if not {"ticker", "shares"}.issubset(holdings.columns):
        raise ValueError("current_holdings.csv must have columns: ticker,shares")
    for _, row in holdings.iterrows():
        ticker = str(row["ticker"]).upper().strip()
        shares = float(row["shares"])
        if ticker == "CASH":
            price = 1.0
            current_value = shares
        elif ticker in latest_prices.index and not pd.isna(latest_prices[ticker]):
            price = float(latest_prices[ticker])
            current_value = shares * price
        else:
            print(f"Warning: no latest price for {ticker}; value set to 0.")
            price = np.nan
            current_value = 0.0
        rows.append({"ticker": ticker, "shares": shares, "price": price, "current_value": current_value})
    if not rows:
        rows.append({"ticker": "CASH", "shares": cfg.default_portfolio_value, "price": 1.0, "current_value": cfg.default_portfolio_value})
    return pd.DataFrame(rows).groupby("ticker", as_index=False).agg({"shares": "sum", "price": "last", "current_value": "sum"})


def build_rebalance_plan(target_weights: pd.Series, holdings: pd.DataFrame, latest_prices: pd.Series, cfg: Config, execute_rebalance_today: bool) -> pd.DataFrame:
    portfolio_value = float(holdings["current_value"].sum())
    if portfolio_value <= 0:
        portfolio_value = cfg.default_portfolio_value
        holdings = pd.DataFrame([{"ticker": "CASH", "shares": portfolio_value, "price": 1.0, "current_value": portfolio_value}])
    tickers = list(dict.fromkeys(cfg.tickers + ["CASH"] + holdings["ticker"].tolist()))
    target_weights_full = pd.Series(0.0, index=tickers, dtype=float)
    target_weights_full.loc[cfg.tickers] = target_weights.reindex(cfg.tickers).fillna(0.0)
    target_weights_full.loc["CASH"] = max(0.0, 1.0 - float(target_weights_full.loc[cfg.tickers].sum()))
    current_shares = holdings.set_index("ticker")["shares"].reindex(tickers).fillna(0.0)
    current_values = holdings.set_index("ticker")["current_value"].reindex(tickers).fillna(0.0)
    current_weights = current_values / portfolio_value
    rows = []
    for ticker in tickers:
        price = 1.0 if ticker == "CASH" else float(latest_prices.get(ticker, np.nan))
        target_weight = float(target_weights_full.get(ticker, 0.0))
        target_value = target_weight * portfolio_value
        current_value = float(current_values.get(ticker, 0.0))
        raw_trade_value = target_value - current_value
        target_shares = target_value if ticker == "CASH" else (target_value / price if price and not pd.isna(price) else np.nan)
        if ticker == "CASH":
            action, trade_value, trade_shares = "CASH", raw_trade_value, raw_trade_value
        elif not execute_rebalance_today or abs(raw_trade_value) < cfg.min_trade_value:
            action, trade_value, trade_shares = "HOLD", 0.0, 0.0
        elif raw_trade_value > 0:
            action, trade_value, trade_shares = "BUY", raw_trade_value, raw_trade_value / price
        else:
            action, trade_value, trade_shares = "SELL", raw_trade_value, raw_trade_value / price
        if not cfg.allow_fractional_shares and ticker != "CASH" and not pd.isna(trade_shares):
            trade_shares = np.floor(abs(trade_shares)) * np.sign(trade_shares)
            trade_value = trade_shares * price
        rows.append({
            "ticker": ticker, "action": action, "latest_price": price,
            "current_shares": float(current_shares.get(ticker, 0.0)),
            "target_shares": target_shares, "trade_shares": trade_shares,
            "current_value": current_value, "target_value": target_value,
            "trade_value": trade_value,
            "current_weight": float(current_weights.get(ticker, 0.0)),
            "target_weight": target_weight,
            "raw_trade_value_before_threshold": raw_trade_value,
        })
    plan = pd.DataFrame(rows)
    order = {"SELL": 0, "BUY": 1, "HOLD": 2, "CASH": 3}
    plan["_order"] = plan["action"].map(order).fillna(9)
    return plan.sort_values(["_order", "ticker"]).drop(columns="_order").reset_index(drop=True)


def plot_stock_interpretation(ticker, prices, scores, momentum, stock_sma_line, market_ok, plan, cfg, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    close = prices[ticker].dropna()
    plot_index = close.index[-cfg.plot_lookback_days:]
    close = close.loc[plot_index]
    sma = stock_sma_line[ticker].loc[plot_index]
    score = scores[ticker].loc[plot_index]
    mom = momentum[ticker].loc[plot_index]
    market_regime = market_ok.reindex(plot_index).ffill().fillna(False)
    row = plan.loc[plan["ticker"] == ticker]
    action = str(row.iloc[0]["action"]) if not row.empty else "HOLD"
    target_weight = float(row.iloc[0]["target_weight"]) if not row.empty else 0.0
    current_weight = float(row.iloc[0]["current_weight"]) if not row.empty else 0.0
    fig, (ax_price, ax_score) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    start = plot_index[0]
    current_state = bool(market_regime.iloc[0])
    for date, state in market_regime.iloc[1:].items():
        state = bool(state)
        if state != current_state:
            ax_price.axvspan(start, date, alpha=0.08 if current_state else 0.12, color="green" if current_state else "red")
            start, current_state = date, state
    ax_price.axvspan(start, plot_index[-1], alpha=0.08 if current_state else 0.12, color="green" if current_state else "red")
    ax_price.plot(close.index, close.values, label=f"{ticker} adjusted close", linewidth=1.8)
    ax_price.plot(sma.index, sma.values, label=f"{cfg.stock_sma}-day SMA", linewidth=1.4)
    ax_price.scatter(close.index[-1], close.iloc[-1], s=70, label=f"Latest action: {action}")
    ax_price.set_title(f"{ticker} | {action} | current {current_weight:.1%} → target {target_weight:.1%}")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="best")
    ax_score.axhline(0, linewidth=1, alpha=0.7)
    ax_score.plot(score.index, score.values, label="vol-adjusted momentum score", linewidth=1.4)
    ax_score.plot(mom.index, mom.values, label="raw momentum", linewidth=1.0, alpha=0.7)
    ax_score.set_ylabel("Score")
    ax_score.set_xlabel("Date")
    ax_score.grid(True, alpha=0.25)
    ax_score.legend(loc="best")
    fig.tight_layout()
    path = out_dir / f"{ticker}_allocator_chart_{pd.Timestamp(prices.index.max()).strftime('%Y-%m-%d')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_charts(prices, scores, momentum, stock_sma_line, market_ok, plan, cfg, report_dir):
    paths = []
    for ticker in cfg.tickers:
        try:
            paths.append(plot_stock_interpretation(ticker, prices, scores, momentum, stock_sma_line, market_ok, plan, cfg, report_dir / "charts"))
        except Exception as exc:
            print(f"Warning: could not create chart for {ticker}: {exc}")
    return paths


def make_summary(prices, scores, market_ok, target_weights, plan, cfg, can_rebalance_today):
    latest_date = pd.Timestamp(prices.index.max())
    market_price = prices[cfg.market_benchmark].iloc[-1]
    market_sma_value = prices[cfg.market_benchmark].rolling(cfg.market_sma).mean().iloc[-1]
    portfolio_value = plan["current_value"].sum()
    selected = target_weights[target_weights > 0].sort_values(ascending=False)
    active = plan[plan["action"].isin(["BUY", "SELL"])]
    lines = ["=" * 78, "BACKTEST-MATCHING ALPACA PAPER PORTFOLIO ALLOCATOR", "=" * 78]
    lines.append(f"Latest data date: {latest_date.date()}")
    lines.append(f"Portfolio value used for sizing: ${portfolio_value:,.2f}")
    lines.append(f"Benchmark: {cfg.market_benchmark} close {market_price:,.2f} vs {cfg.market_sma}SMA {market_sma_value:,.2f}")
    lines.append(f"Market regime: {'RISK ON' if bool(market_ok.iloc[-1]) else 'RISK OFF'}")
    lines.append(f"Rebalance setting: {cfg.rebalance_only_on} | Rebalance today: {'YES' if can_rebalance_today else 'NO'}")
    lines.append("")
    if selected.empty:
        lines.append("TARGET PORTFOLIO: 100% CASH")
    else:
        lines.append("TARGET PORTFOLIO:")
        for ticker, weight in selected.items():
            lines.append(f"  {ticker:6s} {weight:6.1%} | score {scores[ticker].iloc[-1]: .3f}")
        cash_weight = max(0.0, 1.0 - selected.sum())
        if cash_weight > 0.001:
            lines.append(f"  CASH   {cash_weight:6.1%}")
    lines.append("")
    if not can_rebalance_today:
        lines.append("ACTIONS: HOLD — not a scheduled rebalance day. No Alpaca orders will be sent.")
    elif active.empty:
        lines.append("ACTIONS: HOLD — no active trades above threshold.")
    else:
        lines.append("ACTIONS TO MATCH BACKTEST TARGET WEIGHTS:")
        for _, row in active.iterrows():
            lines.append(f"  {row['action']:4s} {row['ticker']:6s} ${abs(row['trade_value']):10,.2f} | ~{abs(row['trade_shares']):,.4f} shares | current {row['current_weight']:6.1%} → target {row['target_weight']:6.1%}")
    lines.append("")
    lines.append("Paper trading only. Not financial advice.")
    return "\n".join(lines)


def get_alpaca_client():
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing Alpaca credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY.")
    try:
        from alpaca.trading.client import TradingClient
    except ImportError as exc:
        raise RuntimeError("alpaca-py is not installed. Run: pip install alpaca-py") from exc
    return TradingClient(key, secret, paper=True)


def submit_alpaca_paper_orders(plan: pd.DataFrame, cfg: Config):
    active = plan[plan["action"].isin(["SELL", "BUY"])].copy()
    active = active[active["ticker"] != "CASH"]
    if active.empty:
        return []
    client = get_alpaca_client()
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    active["_order"] = active["action"].map({"SELL": 0, "BUY": 1})
    active = active.sort_values(["_order", "ticker"])
    submitted = []
    for _, row in active.iterrows():
        ticker = str(row["ticker"])
        action = str(row["action"])
        trade_value = abs(float(row["trade_value"]))
        trade_shares = abs(float(row["trade_shares"]))
        if trade_value < cfg.min_trade_value:
            continue
        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
        if cfg.alpaca_fractional:
            # For buys, shave a little to reduce cash rejection risk from price movement.
            notional = round(trade_value * 0.995, 2) if action == "BUY" else round(trade_value, 2)
            if notional < 1:
                continue
            order = MarketOrderRequest(symbol=ticker, notional=notional, side=side, time_in_force=TimeInForce.DAY)
        else:
            qty = int(np.floor(trade_shares))
            if qty < 1:
                continue
            order = MarketOrderRequest(symbol=ticker, qty=qty, side=side, time_in_force=TimeInForce.DAY)
        resp = client.submit_order(order_data=order)
        submitted.append({"ticker": ticker, "action": action, "trade_value": trade_value, "trade_shares": trade_shares, "alpaca_order_id": getattr(resp, "id", None), "status": getattr(resp, "status", None)})
    return submitted


def send_email_alert(subject: str, body: str, attachments=None):
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    email_from = os.getenv("ALERT_EMAIL_FROM", smtp_user or "")
    email_to = os.getenv("ALERT_EMAIL_TO")
    missing = [k for k, v in {"SMTP_HOST": smtp_host, "SMTP_USER": smtp_user, "SMTP_PASSWORD": smtp_password, "ALERT_EMAIL_TO": email_to}.items() if not v]
    if missing:
        raise RuntimeError("Missing email environment variables: " + ", ".join(missing))
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to
    msg.set_content(body)
    for path in attachments or []:
        path = Path(path)
        if path.exists():
            msg.add_attachment(path.read_bytes(), maintype="application", subtype="octet-stream", filename=path.name)
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls(context=context)
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def main():
    parser = argparse.ArgumentParser(description="Scheduled rebalance allocator with optional Alpaca PAPER trading and email alerts.")
    parser.add_argument("--holdings", default=str(HOLDINGS_FILE), help="CSV with columns ticker,shares")
    parser.add_argument("--rebalance", choices=["D", "W", "M"], default=CONFIG.rebalance_only_on)
    parser.add_argument("--portfolio-value", type=float, default=None, help="Used only if holdings file is missing")
    parser.add_argument("--min-trade-value", type=float, default=CONFIG.min_trade_value)
    parser.add_argument("--whole-shares", action="store_true", help="Round trade shares toward zero")
    parser.add_argument("--no-charts", action="store_true", help="Do not save charts")
    parser.add_argument("--paper-trade", action="store_true", help="Submit Alpaca PAPER orders, only on scheduled rebalance days")
    parser.add_argument("--email", action="store_true", help="Send email alert on rebalance days / submitted paper orders")
    parser.add_argument("--email-monitor-days", action="store_true", help="Also send email on non-rebalance monitor days")
    args = parser.parse_args()

    cfg = CONFIG
    cfg.rebalance_only_on = args.rebalance
    cfg.min_trade_value = args.min_trade_value
    if args.portfolio_value is not None:
        cfg.default_portfolio_value = args.portfolio_value
    if args.whole_shares:
        cfg.allow_fractional_shares = False
        cfg.alpaca_fractional = False
    if args.no_charts:
        cfg.save_charts = False
    if args.email_monitor_days:
        cfg.email_on_monitor_days = True

    symbols = cfg.tickers + [cfg.market_benchmark]
    prices = download_prices(symbols, cfg.period, cfg.interval)
    missing = [s for s in symbols if s not in prices.columns]
    if missing:
        raise RuntimeError(f"Missing price columns: {missing}")
    scores, market_ok, momentum, stock_sma_line = calculate_scores(prices, cfg)
    target_weights = latest_target_weights(scores, cfg)
    latest_prices = prices.iloc[-1]
    holdings = load_current_holdings(Path(args.holdings), latest_prices, cfg)
    latest_date = pd.Timestamp(prices.index.max())
    can_rebalance_today = is_rebalance_day(latest_date, prices.index, cfg.rebalance_only_on)
    plan = build_rebalance_plan(target_weights, holdings, latest_prices, cfg, can_rebalance_today)
    summary = make_summary(prices, scores, market_ok, target_weights, plan, cfg, can_rebalance_today)
    print(summary)

    REPORT_DIR.mkdir(exist_ok=True)
    date_str = latest_date.strftime("%Y-%m-%d")
    report_path = REPORT_DIR / f"rebalance_plan_{date_str}.csv"
    plan.to_csv(report_path, index=False)
    print(f"\nSaved rebalance plan: {report_path}")

    if cfg.save_charts:
        chart_paths = save_charts(prices, scores, momentum, stock_sma_line, market_ok, plan, cfg, REPORT_DIR)
        print(f"Saved {len(chart_paths)} charts in: {REPORT_DIR / 'charts'}")

    submitted_orders = []
    if args.paper_trade:
        if not can_rebalance_today:
            print("\nPaper trading skipped: today is not a scheduled rebalance day.")
        elif plan[plan["action"].isin(["BUY", "SELL"])].empty:
            print("\nPaper trading skipped: no active trades above threshold.")
        else:
            print("\nSubmitting Alpaca PAPER orders...")
            submitted_orders = submit_alpaca_paper_orders(plan, cfg)
            orders_path = REPORT_DIR / f"alpaca_paper_orders_{date_str}.csv"
            pd.DataFrame(submitted_orders).to_csv(orders_path, index=False)
            print(f"Submitted {len(submitted_orders)} Alpaca PAPER orders. Saved: {orders_path}")

    should_email = args.email and (can_rebalance_today or cfg.email_on_monitor_days or bool(submitted_orders))
    if should_email:
        subject_prefix = "REBALANCE" if can_rebalance_today else "MONITOR"
        subject = f"{subject_prefix}: Momentum portfolio report {date_str}"
        body = summary
        if submitted_orders:
            body += "\n\nALPACA PAPER ORDERS SUBMITTED:\n"
            for order in submitted_orders:
                body += f"{order['action']} {order['ticker']} ${order['trade_value']:,.2f} | order_id={order['alpaca_order_id']} | status={order['status']}\n"
        try:
            send_email_alert(subject, body, attachments=[report_path])
            print("Email alert sent.")
        except Exception as exc:
            print(f"Warning: email alert failed: {exc}")


if __name__ == "__main__":
    main()
