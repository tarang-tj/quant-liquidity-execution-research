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
buying-power state, and broker market clock. It never creates an order. A closed market is
reported as a normal state; malformed or unavailable state fails the check.

## Stage 2: causal evaluation and promotion

Use the same feed entitlement for training, evaluation, and runtime. Evaluate
chronologically and include a transaction-cost stress that is conservative for
the intended order size:

```bash
python live/evaluate_walk_forward.py --symbol SPY --bars 1000 \
  --training-bars 120 --evaluation-bars 120 --transaction-cost-bps 5
python live/promote_model.py --symbol SPY --bars 1000 \
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
  --minimum-scored 20 --minimum-accuracy 0.52 --maximum-brier 0.25
```

Treat a nonzero exit as a hold condition: do not promote or submit based on a
model whose live evidence is insufficient or fails the configured quality gate.

## Stage 4: explicitly authorized paper submission

Only after the preceding evidence is reviewed may an operator use
`--submit-paper-order`. The adapter has a fixed Alpaca paper endpoint, reads
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
