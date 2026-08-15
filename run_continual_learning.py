#!/usr/bin/env python3
"""Auditable rolling recalibration for the synthetic liquidity-state filter.

This is a research-training loop, not a live trading process.  Each cycle fits
the observation model only from *completed* labelled synthetic batches, scores
an independently seeded future batch, then appends that completed batch to the
training history.  The execution policy sees the learned filter but never the
latent state for its current decision.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from run_research import RESULTS, Params, filter_diagnostics, generate_market, simulate_policy, summarize


@dataclass(frozen=True)
class CalibratedFilter:
    """A fitted two-state Gaussian HMM observation model for causal filtering."""

    transition: np.ndarray
    emission_means: np.ndarray
    emission_covariances: np.ndarray

    @property
    def stationary_stress(self) -> float:
        p00, p11 = float(self.transition[0, 0]), float(self.transition[1, 1])
        return (1 - p00) / ((1 - p00) + (1 - p11))


FEATURES = ("log_trailing_vol", "log_spread", "log_depth")


def _validate_filter(model: CalibratedFilter) -> None:
    """Fail clearly if a learned snapshot cannot support stable filtering."""
    if model.transition.shape != (2, 2) or not np.isfinite(model.transition).all():
        raise ValueError("learned transition must be a finite 2x2 matrix")
    if np.any(model.transition <= 0) or not np.allclose(model.transition.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("learned transition must be row-stochastic with positive entries")
    if model.emission_means.shape != (2, len(FEATURES)) or not np.isfinite(model.emission_means).all():
        raise ValueError("learned emission means must be finite with shape (2, 3)")
    if model.emission_covariances.shape != (2, len(FEATURES), len(FEATURES)):
        raise ValueError("learned covariances must have shape (2, 3, 3)")
    for state, covariance in enumerate(model.emission_covariances):
        if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, atol=1e-12):
            raise ValueError(f"state {state} covariance must be finite and symmetric")
        if float(np.linalg.eigvalsh(covariance).min()) <= 0:
            raise ValueError(f"state {state} covariance must be positive definite")


def fit_filter(completed_batches: pd.DataFrame, shrinkage: float = 0.08) -> CalibratedFilter:
    """Estimate transition/emission parameters from completed labelled batches.

    Labels are available here because the project is synthetic and the batch has
    already closed.  This makes the training protocol inspectable; a real-market
    replacement would need a separately validated unsupervised/EM estimator.
    """
    if not 0.0 <= shrinkage < 1.0:
        raise ValueError("shrinkage must lie in [0, 1)")
    if completed_batches.regime_true.nunique() != 2:
        raise ValueError("both synthetic regimes must appear in training data")
    transitions = np.ones((2, 2), dtype=float)  # Laplace smoothing.
    for _, path in completed_batches.groupby("path_id", sort=False):
        state = path.sort_values("t").regime_true.to_numpy(dtype=int)
        for before, after in zip(state[:-1], state[1:]):
            transitions[before, after] += 1
    transition = transitions / transitions.sum(axis=1, keepdims=True)
    means, covariances = [], []
    for state in (0, 1):
        values = completed_batches.loc[completed_batches.regime_true == state, FEATURES].to_numpy(dtype=float)
        if len(values) < 2:
            raise ValueError(f"state {state} needs at least two completed observations for covariance estimation")
        mean = values.mean(axis=0)
        empirical = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
        diagonal_target = np.diag(np.diag(empirical))
        covariance = (1 - shrinkage) * empirical + shrinkage * diagonal_target + np.eye(len(FEATURES)) * 1e-6
        means.append(mean); covariances.append(covariance)
    learned = CalibratedFilter(transition=transition, emission_means=np.asarray(means), emission_covariances=np.asarray(covariances))
    _validate_filter(learned)
    return learned


def run_cycles(params: Params, cycles: int, initial_train_paths: int, eval_paths: int) -> pd.DataFrame:
    """Train on completed history and score independent next-batch performance."""
    if min(cycles, initial_train_paths, eval_paths) <= 0:
        raise ValueError("cycles and path counts must be positive")
    initial_seed = params.seed + 10_000
    training = generate_market(replace(params, seed=initial_seed, paths=initial_train_paths))
    training_batches = [{"seed": initial_seed, "paths": initial_train_paths, "path_id_offset": 0}]
    rows, snapshots, holdout_diagnostics = [], [], []
    for cycle in range(1, cycles + 1):
        learned = fit_filter(training)
        evaluation_seed = params.seed + 10_000 + cycle
        evaluation = generate_market(replace(params, seed=evaluation_seed, paths=eval_paths))
        diagnostics = filter_diagnostics(evaluation, learned)
        holdout_diagnostics.append(diagnostics.assign(cycle=cycle))
        target = diagnostics.regime_true.to_numpy(dtype=int)
        predicted = diagnostics.predicted_stress.to_numpy(dtype=int)
        posterior = diagnostics.posterior_stress.to_numpy(dtype=float)
        static_costs = simulate_policy(evaluation, "static_ac", params)[1]
        learned_costs = simulate_policy(evaluation, "regime_aware_mpc", params, belief_model=learned)[1]
        report = summarize(pd.concat([static_costs, learned_costs], ignore_index=True), params).set_index("policy")
        rows.append({
            "cycle": cycle,
            "training_paths": int(training.path_id.nunique()),
            "evaluation_paths": eval_paths,
            "p00_estimate": float(learned.transition[0, 0]),
            "p11_estimate": float(learned.transition[1, 1]),
            "stationary_stress_estimate": learned.stationary_stress,
            "holdout_accuracy": float((predicted == target).mean()),
            "holdout_brier": float(np.mean((posterior - target) ** 2)),
            "static_ac_cvar95_bps": float(report.loc["static_ac", "cvar_95_bps"]),
            "learned_mpc_cvar95_bps": float(report.loc["regime_aware_mpc", "cvar_95_bps"]),
            "learned_minus_static_cvar95_bps": float(report.loc["regime_aware_mpc", "cvar_95_bps"] - report.loc["static_ac", "cvar_95_bps"]),
            "max_completion_error": float(report.max_completion_error.max()),
            "max_constraint_violation": float(report.max_constraint_violation.max()),
        })
        snapshots.append({
            "cycle": cycle,
            "training_batches": training_batches.copy(),
            "holdout_batch": {"seed": evaluation_seed, "paths": eval_paths, "path_id_offset": 0},
            "transition": learned.transition.tolist(),
            "emission_means": learned.emission_means.tolist(),
            "emission_covariances": learned.emission_covariances.tolist(),
        })
        # The just-scored batch becomes eligible only for the next retraining cycle.
        offset = cycle * 1_000_000
        training = pd.concat([training, evaluation.assign(path_id=evaluation.path_id + offset)], ignore_index=True)
        training_batches.append({"seed": evaluation_seed, "paths": eval_paths, "path_id_offset": offset})
    RESULTS.mkdir(exist_ok=True)
    output = pd.DataFrame(rows)
    if output.max_completion_error.max() > 1e-8 or output.max_constraint_violation.max() > 1e-8:
        raise RuntimeError("continual-learning execution validation failed")
    output.to_csv(RESULTS / "online_learning_metrics.csv", index=False)
    pd.concat(holdout_diagnostics, ignore_index=True).to_csv(RESULTS / "online_holdout_diagnostics.csv", index=False)
    learned = fit_filter(training)
    snapshot = {
        "scope": "synthetic completed-batch supervised recalibration; not a live deployment model",
        "cycles_completed": cycles,
        "training_paths_after_last_cycle": int(training.path_id.nunique()),
        "snapshot_schema": "Each cycle records all prior training seed/path batches, the excluded current holdout batch, and full learned transition/emission parameters.",
        "cycle_snapshots": snapshots,
        "final_model_after_all_completed_batches": {
            "transition": learned.transition.tolist(),
            "emission_means": learned.emission_means.tolist(),
            "emission_covariances": learned.emission_covariances.tolist(),
        },
    }
    (RESULTS / "online_filter_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--initial-train-paths", type=int)
    parser.add_argument("--eval-paths", type=int)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    smoke = args.mode == "smoke"
    # q3 is a sequential audit: six independently seeded holdouts are more
    # informative here than making each update batch as large as q2's 1,200-path
    # policy comparison.  The defaults keep the full protocol reproducible in a
    # workstation-scale run while preserving per-cycle feasibility checks.
    params = Params(seed=args.seed, paths=args.eval_paths or (20 if smoke else 40), bootstrap_reps=50 if smoke else 100)
    output = run_cycles(params, args.cycles or (1 if smoke else 6), args.initial_train_paths or (20 if smoke else 50), args.eval_paths or (20 if smoke else 40))
    print(json.dumps({"cycles": len(output), "final_holdout_accuracy": round(float(output.holdout_accuracy.iloc[-1]), 4), "output": str(RESULTS)}, indent=2))


if __name__ == "__main__":
    main()
