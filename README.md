# Regime-Aware Liquidity-Risk Optimal Execution

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
auditable completed-batch recalibration loop. Seven fast invariant tests cover
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
direction baseline trained on a chronological split; append-only local quote
logging; and fail-closed risk gates. It **cannot submit a live order**: the
only broker adapter hard-codes Alpaca's paper-trading endpoint, and even a
paper order requires `--submit-paper-order` plus fresh quote, spread, position,
open-order reservations, notional, daily-loss, and kill-switch checks. At
submission time, daily P&L, and full quantity of open orders are read from the
fixed Alpaca paper account before position, so a fill between reads is
over-reserved rather than undercounted; unavailable or malformed broker state
fails closed rather than trusting command-line inputs. A non-blocking local
process lease covers the check-to-submit window. This bridge is deliberately a
single-host paper runner, not a multi-host/HA execution service. Live
predictions use only completed minute bars, never the in-progress bar.

1. Create separate Alpaca market-data and paper-trading credentials. Put them
   in your shell environment (never commit them); see [`.env.example`](.env.example).
2. Train using historical bars. The command below fetches bars only when data
   credentials are present; `--input` supports an offline CSV with
   `timestamp,open,high,low,close,volume` for reproducible tests. A bundled
   format fixture verifies the offline path:

   ```bash
   python live/train_predictor.py --symbol SPY --bars 1000
   python live/train_predictor.py --symbol SPY --input tests/fixtures/spy_bars.csv --output /tmp/spy_model.json
   ```

3. Evaluate a fresh quote and print the paper-order recommendation. It does not
   submit anything by default:

   ```bash
   python live/run_paper.py --symbol SPY
   ```

   Before the first evaluation, run the read-only readiness preflight. It reads
   the model, live data, and paper account/risk snapshot but never creates an
   order:

   ```bash
   python live/preflight.py --symbol SPY
   ```

4. Only after reviewing the logs and model report may you explicitly add
   `--submit-paper-order`. This remains simulation, not production approval.

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
feed. A production system also needs entitlement-appropriate consolidated data,
reconnect/replay handling, corporate-action adjustments, backtests with real
transaction costs, independent model/risk approval, monitoring, and a long
paper-trading record before live capital is considered.
