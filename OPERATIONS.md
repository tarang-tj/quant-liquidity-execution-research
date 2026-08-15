# Operations runbook

This repository is a research and paper-trading bridge. It is not approved for
live capital. The stages below are ordered so that evidence is accumulated
before any stronger permission is considered.

## Stage 0: offline verification

Run from a clean checkout with no broker credentials:

```bash
python -m unittest discover -s tests -v
python -m compileall -q live tests/test_live.py
python live/train_predictor.py --symbol SPY --input tests/fixtures/spy_bars.csv \
  --output /tmp/spy_model.json
```

The model must pass chronological training, provenance, and deployability
checks. Keep the generated artifact outside the repository unless it is an
intentional model release.

## Stage 1: credentialed read-only checks

Create separate Alpaca market-data and paper-trading credentials. Store them in
the environment only; never put them in a file committed to Git:

```bash
export ALPACA_DATA_KEY=...
export ALPACA_DATA_SECRET=...
export ALPACA_PAPER_KEY=...
export ALPACA_PAPER_SECRET=...
python live/preflight.py --symbol SPY --max-training-gap-hours 168 \
  --max-bar-gap-minutes 3
```

Preflight reads the model, completed bars, quote, account/position/open-order/
buying-power state, broker market clock, and the validated exchange trading calendar
(including early-close sessions). It never creates an order. A closed market is
reported as a normal state; malformed or unavailable state fails the check.

For a bounded feed-health sample, capture normalized WebSocket quotes without
touching the broker:

```bash
python live/stream_quotes.py --symbols SPY --max-quotes 100 \
  --timeout-seconds 60 --output runtime/SPY_quotes.jsonl
```

The collector exits after its finite quote/time budget, fsyncs each JSONL event,
and reports whether the stream exhausted or timed out. An empty sample is a
failure; it is not evidence that the feed is healthy.

## Stage 2: causal evaluation and promotion

Use the same feed entitlement and corporate-action adjustment policy for
training, evaluation, and runtime. The model artifact records both and runtime
rejects mismatches or legacy artifacts without provenance. The default
adjustment policy is `all` (splits, dividends, and spin-offs). Evaluate
chronologically and include a transaction-cost stress that is conservative for
the intended order size:

```bash
python live/evaluate_walk_forward.py --symbol SPY --bars 1000 \
  --training-bars 120 --evaluation-bars 120 --transaction-cost-bps 5
python live/promote_model.py --symbol SPY --bars 1000 \
  --target models/SPY_logistic.json
```

For scheduled paper evaluation, the bounded retraining runner repeats this
workflow without ever submitting an order. It writes one durable promotion
report per cycle and exits after the requested count:

```bash
python live/retrain_model.py --symbol SPY --iterations 24 \
  --interval-seconds 3600 --report runtime/SPY_promotions.jsonl \
  --target models/SPY_logistic.json
```

Promotion is atomic and leaves the existing active model untouched when the
candidate fails. Record the evaluation output, model hash, feed, bar window,
and thresholds with the release.

## Stage 3: finite paper observation

Run a bounded read-only monitor before enabling any order submission:

```bash
python live/paper_monitor.py --symbol SPY --iterations 30 \
  --interval-seconds 60 --max-training-gap-hours 168 \
  --max-bar-gap-minutes 3 --min-direction-edge 0.05
```

Review prediction quality, quote freshness, spread, exposure reservations,
daily P&L, rejected samples, and the durable decision journal. If a submission
ever has an unknown network outcome, do not retry; reconcile by client order ID:

```bash
python live/reconcile_paper.py --client-order-id <id>
```

Once later completed bars exist, calculate realized monitor quality with the
read-only scorer. It labels only the immediate next bar and reports pending
records when the bar is not yet available:

```bash
python live/score_predictions.py --symbol SPY \
  --decision-log runtime/paper_monitor_decisions.jsonl --bars 1000 \
  --minimum-scored 20 --minimum-accuracy 0.52 --maximum-brier 0.25 \
  --output runtime/SPY_quality.json
```

Treat a nonzero exit as a hold condition: do not promote or submit based on a
model whose live evidence is insufficient or fails the configured quality gate.
The output report is atomically written and pinned to one model hash. Paper
submission rejects a missing, stale, mixed-model, symbol-mismatched, or failed
report.

Use the combined read-only health snapshot as the machine-checkable session
gate. It verifies the model, live data, broker state, market clock/calendar, fresh
quality report, and decision-journal integrity without submitting:

```bash
python live/health.py --symbol SPY \
  --quality-report runtime/SPY_quality.json \
  --decision-log runtime/paper_monitor_decisions.jsonl \
  --max-training-gap-hours 168 --max-bar-gap-minutes 3
```

Treat a nonzero exit as a hold condition and route the structured JSON to the
operator alerting system.

## Stage 4: explicitly authorized paper submission

Only after the preceding evidence is reviewed may an operator use
`--submit-paper-order --quality-report runtime/SPY_quality.json`. The adapter has a fixed Alpaca paper endpoint, reads
broker state immediately before validation, requires an open market clock, and
has no automatic retry path. The local submission lease is single-host only;
this is not a highly available execution service.

Keep `TRADING_KILL_SWITCH=1` available as the emergency stop. Preserve the
decision journal and broker reconciliation results as operational evidence.

## Requirements before any live-capital design

This repository does not provide a live endpoint or authorize live trading.
Before a separate live-capital system could be considered, it would need at
least: independent risk authorization, multi-host coordination, durable
external audit/WORM storage, order-status reconciliation and idempotency,
corporate-action and trading-calendar handling, entitlement-appropriate
consolidated data, alerting/on-call ownership, disaster recovery, secrets
management, exchange/broker compliance review, and a statistically meaningful
paper-trading record.
