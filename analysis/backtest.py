"""Clean replay of the paper-trader algorithm on complete daily closes.

Rules (as the README intends):
  - rank all watchlist assets by 90-trading-day return on the rebalance day's close
  - hold the top 3 equal-weighted, trade at that same close
  - daily = every trading day, weekly = first trading day of each week,
    monthly = first trading day on/after the 28th (once per month)
"""
import sys
import pandas as pd

LOOKBACK, TOP_N = 90, 3

px = pd.read_csv(sys.argv[1], dtype={"date": str})
px["date"] = pd.to_datetime("20" + px["date"], format="%Y%m%d")
px = px.set_index("date").rename(columns={"BTC-USD": "BTC"})
spy = px.pop("SPY")
mom = px.pct_change(LOOKBACK)


def rebalance_days(idx, freq):
    s = pd.Series(idx, index=idx)
    if freq == "daily":
        return set(idx)
    if freq == "weekly":
        return set(s.groupby(idx.to_period("W")).first())
    if freq == "monthly":
        late = s[s.dt.day >= 28]
        return set(late.groupby(late.dt.to_period("M")).first())


def run(freq, start, end=None, cost=0.0, lag=0):
    """lag=1 executes at the next day's close (no look-ahead at all)."""
    idx = px.loc[start:end].index
    rb = rebalance_days(idx, freq) | {idx[0]}
    hold, cash, vals, trades, picks = {}, 10000.0, [], 0, []
    pending = None
    for i, d in enumerate(idx):
        p = px.loc[d]
        if pending is not None:
            top = pending
            val = cash + sum(s * p[t] for t, s in hold.items())
            new = {t: val / len(top) / p[t] for t in top}
            turnover = sum(abs(new.get(t, 0) - hold.get(t, 0)) * p[t] for t in set(new) | set(hold))
            trades += sum(1 for t in set(new) | set(hold) if abs(new.get(t, 0) - hold.get(t, 0)) * p[t] > 1)
            val -= turnover * cost
            hold, cash = {t: val / len(top) / p[t] for t in top}, 0.0
            picks.append((d.date(), top))
            pending = None
        if d in rb:
            top = list(mom.loc[d].dropna().sort_values(ascending=False).head(TOP_N).index)
            if lag == 0:
                pending = top
                # execute immediately (same close)
                val = cash + sum(s * p[t] for t, s in hold.items())
                new = {t: val / len(top) / p[t] for t in top}
                turnover = sum(abs(new.get(t, 0) - hold.get(t, 0)) * p[t] for t in set(new) | set(hold))
                trades += sum(1 for t in set(new) | set(hold) if abs(new.get(t, 0) - hold.get(t, 0)) * p[t] > 1)
                val -= turnover * cost
                hold, cash = {t: val / len(top) / p[t] for t in top}, 0.0
                picks.append((d.date(), top))
                pending = None
            else:
                pending = top
        vals.append(cash + sum(s * p[t] for t, s in hold.items()))
    return pd.Series(vals, index=idx), trades, picks


def spy_bh(start, end=None):
    s = spy.loc[start:end]
    return 10000 * s / s.iloc[0]


def ew_bh(start, end=None):
    p = px.loc[start:end]
    return 10000 * (p / p.iloc[0]).mean(axis=1)


if __name__ == "__main__":
    logged = {"daily": 7486.48, "weekly": 8338.45, "monthly": 6610.66}
    starts = {"daily": "2026-05-08", "weekly": "2026-06-08", "monthly": "2026-06-08"}
    rows = []
    for f in ["daily", "weekly", "monthly"]:
        st = starts[f]
        v0, n0, picks = run(f, st)
        v1, _, _ = run(f, st, cost=0.001)
        v2, _, _ = run(f, st, lag=1)
        dd = (v0 / v0.cummax() - 1).min()
        rows.append(dict(strategy=f, start=st, bot_logged=logged[f], clean=round(v0.iloc[-1]),
                         clean_10bp=round(v1.iloc[-1]), clean_next_day=round(v2.iloc[-1]),
                         max_dd=f"{dd:.0%}", rebalances_with_trades=n0,
                         spy=round(spy_bh(st).iloc[-1]), equal_wt_watchlist=round(ew_bh(st).iloc[-1])))
        v0.to_csv(f"clean_{f}.csv")
        if f != "daily":
            print(f, "picks:")
            for d, t in picks:
                print("  ", d, t)
    print(pd.DataFrame(rows).to_string(index=False))

    # Longest window the data allows (first date with 90 days of history)
    st = mom.dropna(how="all").index[0]
    print("\nLongest window from", st.date())
    for f in ["daily", "weekly", "monthly"]:
        v, _, _ = run(f, st)
        print(f"  {f:8s} {v.iloc[-1]:8.0f}")
    print(f"  SPY      {spy_bh(st).iloc[-1]:8.0f}")
    print(f"  EW list  {ew_bh(st).iloc[-1]:8.0f}")
