# Refit Runbook -- Kalshi BTC Signal Bot
Written 2026-05-24, before a 2-4 week unattended data-collection period.

## TL;DR for the return
1. Confirm collection ran: `ls -lt /var/lib/btc-signal/predictions/ | head` (today raw, older .gz).
2. Set the clean-window start: `export BTC_REFIT_SINCE=2026-05-25` (see Clean-data cutoff).
3. *** Do NOT trust refit CV numbers until window-level CV is implemented *** (see KNOWN ISSUE).
4. Grouping analysis: `BTC_PREDICTIONS_DIR=/var/lib/btc-signal/predictions python3 time_analysis.py`

## What is deployed
- mark_price = median of each asset's EXACT accessible CFB RTI constituents:
  BTC/ETH: coinbase,kraken,bitstamp,gemini,bullish,crypto_com | SOL: cb,kr,gemini,bitstamp,crypto_com
  XRP: cb,kr,bitstamp,crypto_com | DOGE: cb,gemini,kr | BNB: cb,kr | HYPE: cb,kr (Bitstamp has no public HYPE)
- Binance US is LOGGED as a directional/lead guide only; NOT in mark_price.
- Bullish/Crypto.com via cached collectors (20s/10s TTL) to avoid 429 rate limits.
- 15-feature schema (analyze.py WEIGHTS); the 4 forward-looking features are at weight 0.0 pending refit:
  established: momentum_short, momentum_medium, trend_slope, exchange_alignment, distance_from_strike,
              funding_rate, pre_window_slope, slot_30, slot_45
  market:     kalshi_log_odds, kalshi_log_odds_late
  forward:    vol_scaled_distance, accel, vol_regime, lead_lag
- Live archive: /var/lib/btc-signal/predictions (env BTC_PREDICTIONS_DIR). Settled: data/settled.jsonl (live).
- Disk: gzip-rotate cron 00:30 UTC (scripts/btc-signal-gzip-rotate.sh) compresses yesterday+older.
  All loaders are gz-aware. Steady-state ~2-3 GB.

## Clean-data cutoff (BTC_REFIT_SINCE)
Clean 6-source prices AND the full 15-feature schema both went live 2026-05-24 (~04:30 UTC).
First FULL clean day = 2026-05-25. Use BTC_REFIT_SINCE=2026-05-25.
build_dataset also auto-drops any record missing any of the 15 features, as a backstop.

## Run the refit (memory-safe via since-filter)
  BTC_PREDICTIONS_DIR=/var/lib/btc-signal/predictions BTC_REFIT_SINCE=2026-05-25 python3 fit_weights.py
Output: data/fitted_weights.json + console summary.
Pure-Python training is slow and loads into RAM. On the 1 GB box a multi-week window may be slow or OOM.
If so: narrow BTC_REFIT_SINCE to ~5-7 days, OR temporarily resize droplet to 4 GB, OR do the downsampling task below.

## *** KNOWN ISSUE -- DO NOT SKIP (Claude task on return) ***
fit_weights k-fold CV splits by ROW. Each 15-min window logs ~hundreds of near-identical 2s snapshots,
so the same window appears in both train and test folds => LEAKAGE => CV Brier/accuracy/"% improvement"
are INFLATED and NOT trustworthy. (A 2026-05-24 smoke run produced a fake "8.2% better".)
Before deploying ANY weights, implement:
  1. Per-window phase-bucket downsampling in build_dataset (1 row per ticker per ~30-60s bucket). Also fixes RAM/compute.
  2. CV folds split by TICKER (window), never by row.
Only then are metrics honest and weights safe to copy into analyze.py WEIGHTS/INTERCEPT.

## Grouping analysis (your day-of-week / time-of-day question)
  BTC_PREDICTIONS_DIR=/var/lib/btc-signal/predictions python3 time_analysis.py
Already groups by day-of-week, hour-of-day (UTC), session bucket, day x hour heatmap (thin-sample flagged),
day x session, early-phase by day. Promote a grouping to a model feature only if its effect is stable across
enough windows -- thin per-group samples overfit (that is what blew up slot_45 in the smoke run).

## Health checks
  systemctl is-active btc-signal.service btc-analyze-loop.service btc-live-gen.service
  tail -1 newest /var/lib/btc-signal/predictions file -> expect 15 signal_*_raw keys + recent ts
  df -h / ; tail /var/log/btc-gzip-rotate.log
  Backups: restic, B2 bucket msfit-signal-backups, daily 0:10 cron.

## Deploy new weights (only after a TRUSTWORTHY refit)
Copy fitted_weights.json values into analyze.py WEIGHTS + INTERCEPT (feature names match).
Re-mirror current_weights in fit_weights.py to the deployed values. Commit.
The analyze loop re-reads analyze.py every 2s, so changes go live within 2s (no restart).

## Sharing & scaling viewers (dashboard)
Droplet static IP: 159.223.105.39
Current capacity (free ~1 TB/mo transfer; overage $0.01/GiB):
  main dashboard = prediction.json (~20KB) polled every 2s = ~36 MB/hr per open tab.
  => ~40 always-on viewers, ~900 at 1hr/day, ~5,000 casual (10min/day). Sharing with hundreds is fine today, free.
  (closing_gap.html is heavier -- it also pulls the 40KB backtest file.)
Cloudflare (free) -- worth it only to reach thousands, hide origin IP, or add DDoS protection:
  - Requires YOUR OWN domain (a duckdns subdomain cannot be a free CF zone). Register a cheap domain,
    add to Cloudflare, A-record -> 159.223.105.39, proxy ON. Droplet IP is static, so DuckDNS can be dropped.
  - To actually OFFLOAD bandwidth (not just hide IP), the JSON must be edge-cacheable:
      1. Remove the ?t=Date.now() cache-buster on JSON fetches (index.html, closing_gap.html, closing-gap-live.html).
      2. Edge-cache JSON ~2s TTL (CF cache rule, or nginx add_header Cache-Control "max-age=2" on data/*.json).
         Then origin serves ~1 req / 2s regardless of viewers -> effectively unlimited audience.
      3. index.html ~line 79 hardcodes https://msfit.duckdns.org/...; make relative or new-domain or it bypasses CF.
  - Droplet hardening (with Claude on return): nginx realip with CF IP ranges; ufw allow 80/443 only from CF ranges.
  Recommendation: do the CF migration on return with a domain in hand, not before an unattended stretch.
