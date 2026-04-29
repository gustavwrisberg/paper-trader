# GitHub Actions setup for Alpaca paper trading

This package runs your portfolio allocator in GitHub Actions, so your PC does not need to stay on.

## 1. Create a GitHub repository

Create a private repo, then upload these files/folders:

```text
.github/workflows/daily_alpaca_paper.yml
daily_portfolio_allocator_alpaca_paper.py
requirements.txt
current_holdings.csv
```

## 2. Edit your holdings

Edit `current_holdings.csv` in the repo.

Example:

```csv
ticker,shares
NVDA,5
MSFT,3
CASH,2500
```

`CASH` means uninvested cash in dollars.

## 3. Add GitHub Secrets

Go to:

```text
Repo → Settings → Secrets and variables → Actions → New repository secret
```

Add these required Alpaca paper-trading secrets:

```text
APCA_API_KEY_ID
APCA_API_SECRET_KEY
```

Add these email secrets if you want email alerts:

```text
SMTP_HOST
SMTP_PORT
SMTP_USER
SMTP_PASSWORD
ALERT_EMAIL_FROM
ALERT_EMAIL_TO
```

For Gmail, you usually need an app password, not your normal Gmail password.

## 4. Schedule

The workflow currently runs:

```text
Monday-Friday at 22:30 UTC
```

That is usually after the US market close in Denmark.

The script runs daily, but only submits Alpaca paper orders on rebalance days. The default rebalance frequency is monthly.

## 5. Manual test run

In GitHub:

```text
Actions → Daily Alpaca Paper Portfolio Check → Run workflow
```

Choose:

```text
paper_trade: false
email: false
rebalance: M
```

Do this first to confirm the script works before enabling paper orders.

## 6. Paper-trading run

After the dry run works, manually run with:

```text
paper_trade: true
email: true
rebalance: M
```

Or let the scheduled workflow run automatically.

## Important safety notes

This uses Alpaca paper trading only. It is for testing execution and workflow reliability, not real-money trading.

Check the generated reports under the workflow artifact named `portfolio-reports`.
