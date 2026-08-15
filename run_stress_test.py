#!/usr/bin/env python3
"""Out-of-distribution liquidity-regime stress test for the execution study."""

from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from run_research import RESULTS, Params, generate_market, simulate_policy, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    base = Params(seed=args.seed, paths=args.paths, bootstrap_reps=200)
    scenarios = (
        ("baseline_regime", base),
        # More frequent/persistent stress and a 50% larger stressed-state impact.
        ("persistent_severe_stress", replace(base, p00=.88, p11=.90, eta_stress=36.0)),
    )
    reports = []
    for name, params in scenarios:
        market = generate_market(params)
        costs = pd.concat(
            [simulate_policy(market, policy, params)[1]
             for policy in ("twap", "static_ac", "regime_aware_mpc", "oracle_mpc")],
            ignore_index=True,
        )
        report = summarize(costs, params)
        report.insert(0, "scenario", name)
        report.insert(1, "p00", params.p00)
        report.insert(2, "p11", params.p11)
        report.insert(3, "eta_stress", params.eta_stress)
        reports.append(report)
    result = pd.concat(reports, ignore_index=True)
    if result.max_completion_error.max() > 1e-8 or result.max_constraint_violation.max() > 1e-8:
        raise RuntimeError("stress test execution validation failed")
    RESULTS.mkdir(exist_ok=True)
    result.to_csv(RESULTS / "stress_test.csv", index=False)
    print(f"Wrote {len(result)} stress-test rows to {RESULTS / 'stress_test.csv'}")


if __name__ == "__main__":
    main()
