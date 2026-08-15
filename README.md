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
