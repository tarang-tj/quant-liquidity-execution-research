#!/usr/bin/env python3
"""Deterministic robustness checks for the execution-policy comparison.

This is deliberately a smaller, fixed panel of the full synthetic experiment.
It changes the assumed impact/risk parameters while holding the market paths and
policy feasibility rules constant.  It is a robustness diagnostic, not a
parameter-fitting exercise.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from run_research import DATA, RESULTS, Params, simulate_policy, summarize


SCENARIOS = (
    ("low_risk_base_impact", 0.02, 1.00),
    ("base_case", 0.08, 1.00),
    ("high_risk_base_impact", 0.20, 1.00),
    ("base_risk_high_impact", 0.08, 1.50),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    market = pd.read_csv(DATA / "synthetic_market.csv")
    ids = sorted(market.path_id.unique())[:args.paths]
    if len(ids) < args.paths:
        raise ValueError(f"requested {args.paths} paths but only {len(ids)} are available")
    panel = market[market.path_id.isin(ids)].copy()
    rows = []
    for scenario, risk_aversion, impact_multiplier in SCENARIOS:
        params = replace(
            Params(seed=args.seed, paths=args.paths, bootstrap_reps=200),
            risk_aversion=risk_aversion,
            eta_calm=Params.eta_calm * impact_multiplier,
            eta_stress=Params.eta_stress * impact_multiplier,
        )
        costs = pd.concat(
            [simulate_policy(panel, policy, params)[1] for policy in ("static_ac", "regime_aware_mpc")],
            ignore_index=True,
        )
        report = summarize(costs, params)
        report.insert(0, "scenario", scenario)
        report.insert(1, "risk_aversion", risk_aversion)
        report.insert(2, "impact_multiplier", impact_multiplier)
        rows.append(report)
    output = pd.concat(rows, ignore_index=True)
    if output.max_completion_error.max() > 1e-8 or output.max_constraint_violation.max() > 1e-8:
        raise RuntimeError("sensitivity execution validation failed")
    RESULTS.mkdir(exist_ok=True)
    output.to_csv(RESULTS / "sensitivity.csv", index=False)
    print(f"Wrote {len(output)} sensitivity rows to {RESULTS / 'sensitivity.csv'}")


if __name__ == "__main__":
    main()
