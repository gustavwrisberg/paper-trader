# paper-trader: a failed experiment

> **Result: the strategy did not work.** Over May–November 2026, every variant lost 16–25%, while the S&P 500 (SPY) was roughly flat. Bugs in the bot made the logged numbers worse, but a clean re-run of the same algorithm on correct data also lost money and trailed SPY. This repo is kept as a record of what was tried and why it failed.

## The hypothesis

Pure price momentum: what has gone up the most recently will keep going up.

- **Universe:** 20 assets: US mega-caps (AAPL, MSFT, NVDA, GOOGL, AMZN), global ETFs (URTH, EEM, IEV, EWJ, SCZ), international stocks and ADRs (TSM, ASML, NVO, MHVYF, SIEGY, SBGSY, BN, SHEL, BRK-B), plus BTC.
- **Signal:** the 90-trading-day return of each asset.
- **Portfolio:** the top 3 assets, equal-weighted. No fundamentals, no forecasting, no risk filter.
- **Three variants:** rebalance **daily**, **weekly** (Mondays) or **monthly** (on or after the 28th).
- **Execution:** a paper portfolio starting with $10,000 per variant, run by GitHub Actions every 10 minutes during US market hours, with email and SMS alerts.

## Results

$10,000 starting capital, values on 23 Nov 2026:

| Variant | Start | Bot's own log | Clean re-run | SPY buy & hold | Hold all 20 assets equally |
|---|---|---|---|---|---|
| Daily | 8 May | $7,486 | **$7,869 (−21%)** | $9,842 (−2%) | $9,189 |
| Weekly | 8 Jun | $8,338 | **$8,363 (−16%)** | $9,821 (−2%) | $9,489 |
| Monthly | 8 Jun | $6,611 | **$7,517 (−25%)** | $9,821 (−2%) | $9,489 |

- The **clean re-run** ([`analysis/backtest.py`](analysis/backtest.py)) replays the algorithm exactly as intended: one rebalance per period, trading at the daily close and using complete prices. It has no transaction costs, and adding 0.1% per trade makes it 1–4% worse.
- Simply holding every asset on the watchlist beat picking the top 3 in every variant.
- Over the longest period the data allows (7 Apr to 23 Nov 2026), all three variants lost 13–15%, while SPY gained about 10%.
- Of 18 combinations of lookback (20, 60 or 90 days) and portfolio size (top 1, 3 or 5), 16 trailed SPY. The two that beat it look like luck, not a real edge.

## Why it failed

### 1. The strategy bought recent winners just before they reversed

Every purchase came after a 15–45% run-up over 90 days. In the sideways, choppy market of mid-to-late 2026, those runs reversed again and again. In the monthly variant, 17 of 18 positions that were fully sold were sold at a loss. Most of the monthly loss came in two months:

| Holding period | Holdings | Portfolio | SPY |
|---|---|---|---|
| 29 Jun → 28 Jul | ASML −16%, TSM −14%, AMZN −4% | −11.2% | 0.0% |
| 28 Oct → 23 Nov | BTC −25%, SHEL −11%, MSFT −7% | −14.2% | −6.5% |

Holding only 3 assets made every one of these reversals hurt more.

### 2. The bot itself was broken (about 0–9 points of extra loss)

- **"Weekly" and "monthly" were not weekly or monthly.** `rebalance_today()` returned true on every 10-minute run of a rebalance day, so the bot rebalanced about 40 times a day. Whenever assets ranked 3 and 4 were close, they swapped places during the day, and each swap lost a little. Monthly traded 11.5× its capital and daily traded 68×.
- **Partial price rows.** Ranking and valuation used `prices.iloc[-1]`. On weekend runs and during the trading day, that row was often missing most tickers. When that happened, only a few assets were ranked (BTC was once picked as #1 with −4.2% momentum), 98 trades were for zero shares, positions without a price were kept by accident, and the equity curve dropped to $0 or a third of its value on some days.
- **Runs overwrote each other.** The workflow had no `concurrency:` group and the daily job never ran `git pull`, so overlapping runs overwrote each other's files. The daily trade log does not match the saved portfolio: it implies a short position of −17.9 AAPL and four holdings that disappeared without ever being sold.

### 3. The benchmark line was wrong

The bot computed SPY as `10000 × SPY_today / SPY_one_year_ago`, which is SPY's return over the past year, not its return since the bot started. The SPY line on the dashboard therefore never showed how the bot did against the market.

## Lessons

1. **Backtest before running it live.** Testing on clean history would have shown in minutes what took six months of paper trading to learn.
2. **A strategy with no risk filter, on 20 assets, holding the top 3, is mostly noise.** A real momentum strategy needs a larger universe, skipping the most recent month when measuring momentum, a trend filter (for example, go to cash when SPY is below its 200-day average), and a buffer before selling a holding.
3. **Check the bookkeeping.** Replay the trade log and confirm it matches the saved portfolio before trusting any performance number.
4. **The benchmark must start on the same day as the strategy.**
5. **Seven months is a short test.** These results show the strategy did not work in this period. They do not prove that momentum can never work.

## Reproduce the analysis

```bash
cd analysis
python backtest.py prices.csv
```

`prices.csv` holds dividend-adjusted daily closes from Yahoo Finance (Nov 2025 – Nov 2026) for every asset in the watchlist and SPY.

## Repo contents

| Path | What it is |
|---|---|
| `trader.py` | The original bot, unchanged and buggy as described above |
| `generate_dashboard.py`, `docs/` | Dashboard ([live page](https://gustavwrisberg.github.io/paper-trader/)); its SPY line is wrong, see above |
| `*_daily.csv`, `*_weekly.csv`, `*_monthly.csv` | The bot's raw portfolio, trade and performance logs, including the corrupted data |
| `analysis/` | The clean re-run and the price data it uses |
| `.github/workflows/run_trader.yml` | The GitHub Actions schedule |
