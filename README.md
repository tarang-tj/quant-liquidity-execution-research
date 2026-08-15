# Regime-Aware Liquidity-Risk Optimal Execution

[![Quality checks](https://github.com/tarang-tj/quant-liquidity-execution-research/actions/workflows/quality.yml/badge.svg)](https://github.com/tarang-tj/quant-liquidity-execution-research/actions/workflows/quality.yml)

A portfolio-quality quant research project that studies whether a strictly causal,
two-state HMM liquidity filter can improve execution decisions under changing
market conditions. All data are deterministic synthetic simulations—this is not
a live-trading strategy or a claim about real-market performance.

The project compares TWAP, static Almgren–Chriss-style optimization, and a
posterior-conditioned MPC policy across 1,200 shared market paths. The MPC
propagates today’s causal state posterior through the regime-transition matrix
instead of assuming the same liquidity probability over its whole horizon.

It reports implementation shortfall, 95% VaR/CVaR, calibration, paired
common-path bootstrap contrasts, hard capacity/completion checks, an explicitly
non-deployable oracle diagnostic, a four-case robustness panel, and an
an auditable completed-batch recalibration loop. The live bridge also has a
causal walk-forward evaluator that reports prediction metrics plus a transparent
turnover and transaction-cost stress accounting. Forty-one fast invariant tests cover
causality, forecast dynamics, gradients, constraints, path alignment, and
two-cycle continual-learning timing.

Run the exact reproducibility command recorded in
[`results/复现清单.json`](results/复现清单.json):

```bash
python -m unittest discover -s tests -v && python run_research.py --mode full --seed 20260814 && python run_sensitivity.py --paths 150 --seed 20260814 && python run_stress_test.py --paths 300 --seed 20260814 && python run_continual_learning.py --mode full --seed 20260814 && python make_figures.py
```

Start with [`题目分析报告.md`](题目分析报告.md) for the model contract and
[`results/summary_metrics.csv`](results/summary_metrics.csv) for outcome metrics.

## Live-market and predictive research bridge (paper-only)

`live/` adds a deliberately constrained path from synthetic research to real
US-equity data.  It has an Alpaca adapter for normalized quotes, minute bars,
and optional WebSocket quote streaming; a deterministic logistic next-bar
direction baseline trained on a chronological split; a locally durable,
concurrency-safe quote journal; and fail-closed risk gates. It **cannot submit
a live order**: the
only broker adapter hard-codes Alpaca's paper-trading endpoint, and even a
paper order requires `--submit-paper-order` plus fresh quote, spread, position,
open-order reservations, notional, daily-loss, and kill-switch checks. At
submission time, daily P&L, and full quantity of open orders are read from the
fixed Alpaca paper account before position, so a fill between reads is
over-reserved rather than undercounted; unavailable or malformed broker state
fails closed rather than trusting command-line inputs. A non-blocking local
process lease covers the check-to-submit window. This bridge is deliberately a
single-host paper runner, not a multi-host/HA execution service. Live
predictions use only completed minute bars, never the in-progress bar, and the
decision path rejects model artifacts whose training window extends into the
future relative to the live bars. Market-data REST snapshots use a small,
bounded retry budget for transient transport/5xx/429 failures and fail closed
when that budget is exhausted. Paper submission also reads Alpaca's market
clock immediately before the risk decision and rejects a closed or malformed
session state; notional and buying-power gates use the executable bid/ask side,
not the midpoint; order submission has no automatic retry path.

1. Create separate Alpaca market-data and paper-trading credentials. Put them
   in your shell environment (never commit them); see [`.env.example`](.env.example).
   Commands default to the IEX feed. Where your account is entitled to
   consolidated SIP data, pass `--feed sip` consistently to training,
   walk-forward evaluation, preflight, monitoring, and paper evaluation.
2. Train using historical bars. The command below fetches bars only when data
   credentials are present; `--input` supports an offline CSV with
   `timestamp,open,high,low,close,volume` for reproducible tests. A bundled
   format fixture verifies the offline path. Each model artifact records its
   symbol, timeframe, UTC training range, exact ordered-bar SHA-256, and fit
   configuration hash so a later promotion can be traced to its inputs. Model
   files are rotated atomically, so readers see either complete old or new JSON,
   never a partially written artifact:

   ```bash
   python live/train_predictor.py --symbol SPY --bars 1000
   python live/train_predictor.py --symbol SPY --input tests/fixtures/spy_bars.csv --output /tmp/spy_model.json
   ```

   For recurring retraining, use the promotion gate instead of overwriting the
   active model directly. It leaves the existing artifact untouched unless the
   candidate passes chronological validation, walk-forward accuracy/Brier, and
   cost-aware net-return thresholds:

   ```bash
   python live/promote_model.py --symbol SPY --bars 1000 --target models/SPY_logistic.json
   ```

   Before considering a model for paper use, run a causal walk-forward check.
   Each block is fit only on bars strictly before that block; the result also
   reports a majority-class baseline, gross/net return in basis points,
   turnover, and the assumed transaction-cost stress:

   ```bash
   python live/evaluate_walk_forward.py --symbol SPY --bars 1000 --training-bars 120 --evaluation-bars 120 --transaction-cost-bps 5
   ```

3. Evaluate a fresh quote and print the paper-order recommendation. It does not
   submit anything by default:

   ```bash
   python live/run_paper.py --symbol SPY --max-training-gap-hours 168 \
     --max-bar-gap-minutes 3
   ```

   To avoid forcing a trade when the directional model is nearly indifferent,
   set a minimum probability edge. For example, `--min-direction-edge 0.05`
   journals a `hold` decision whenever the forecast is between 45% and 55%;
   a hold is never submitted and the default `0.0` preserves the baseline
   threshold behavior:

   ```bash
   python live/run_paper.py --symbol SPY --min-direction-edge 0.05
   ```

   Before the first evaluation, run the read-only readiness preflight. It reads
   the model, live data, paper account/risk/buying-power snapshot, broker market clock,
   and the validated exchange trading calendar (including early closes), but never creates an order:

   ```bash
   python live/preflight.py --symbol SPY --max-training-gap-hours 168 \
     --max-bar-gap-minutes 3
   ```

   After the monitor has produced a quality report and journal, the combined
   read-only health snapshot is suitable for a scheduler or alerting adapter:

   ```bash
   python live/health.py --symbol SPY \
     --quality-report runtime/SPY_quality.json \
     --decision-log runtime/paper_monitor_decisions.jsonl \
     --max-training-gap-hours 168 --max-bar-gap-minutes 3
   ```

   It exits nonzero unless model provenance, live data, broker state, market
   clock/calendar, quality evidence, and the local decision hash chain all pass. It
   never creates an order.

   For a finite staged paper observation window, use the read-only monitor. It
   refreshes completed bars and the paper risk snapshot for each iteration,
   appends each prediction to a durable journal, and never submits an order:

   ```bash
   python live/paper_monitor.py --symbol SPY --iterations 30 --interval-seconds 60 \
     --max-training-gap-hours 168 --max-bar-gap-minutes 3 \
     --min-direction-edge 0.05
   ```

   The optional freshness SLA makes each path fail closed when the latest
   completed live bar is more than the specified number of hours after the
   model's recorded training end; omit it when replaying historical fixtures.
   The bar-gap option similarly rejects a missing interval larger than the
   configured number of minutes; choose it according to the symbol and feed's
   expected liquidity rather than treating every no-trade minute as a failure.

   After a subsequent completed bar is available, score the journaled live
   predictions without changing the journal or submitting anything:

   ```bash
   python live/score_predictions.py --symbol SPY \
     --decision-log runtime/paper_monitor_decisions.jsonl --bars 1000 \
     --minimum-scored 20 --minimum-accuracy 0.52 --maximum-brier 0.25 \
     --output runtime/SPY_quality.json
   ```

   The scorer requires the immediate next one-minute bar, reports directional
   accuracy, Brier score, coverage, realized directional return, and leaves
   gaps pending rather than inventing labels. Its quality gate exits nonzero
   until enough scored observations exist and the configured accuracy/Brier
   thresholds pass. The optional output is atomically written and pinned to a
   single model hash; paper submission rejects missing, stale, mixed-model, or
   failed reports.

4. Only after reviewing the logs and model report may you explicitly add
   `--submit-paper-order --quality-report runtime/SPY_quality.json`. This remains
   simulation, not production approval.

Every evaluation records its model hash, completed bars, quote, features, risk
decision, and (only when requested) paper-account state in
`runtime/paper_decisions.jsonl`; it never records secrets. Replay an entry
offline to check that the current model recreates the recorded probability:

```bash
python live/replay_decision.py --model models/SPY_logistic.json --decision-log runtime/paper_decisions.jsonl
```

The journal uses a locked, local integrity chain and is not a replacement for
an external signed/WORM audit archive. Each paper request carries a unique
client-order ID. If its network outcome is unknown, do not retry it; reconcile
first with `python live/reconcile_paper.py --client-order-id <id>`. A verified
broker lookup result is appended as a `reconciliation_result` event to the same
journal (use `--decision-log <path>` only when the journal is non-default).

The free Alpaca IEX feed is single-exchange data, not a consolidated market
feed. The WebSocket adapter uses a bounded reconnect budget and fails closed
when it is exhausted. A production system also needs entitlement-appropriate
consolidated data, replay/gap recovery, corporate-action adjustments, backtests with real
transaction costs, independent model/risk approval, monitoring, and a long
paper-trading record before live capital is considered.

See [`OPERATIONS.md`](OPERATIONS.md) for the staged operator runbook and the
explicit requirements that remain before any live-capital design.
