#!/usr/bin/env python3
"""Reproducible regime-aware liquidity-risk optimal-execution experiment.

All costs are basis points of parent-order notional.  The only input data are
synthetic, generated deterministically from the seed in the model contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# The per-decision KKT systems are at most 14×14.  Single-threaded BLAS avoids
# thread-launch overhead and makes the seeded run substantially more stable.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class Params:
    seed: int = 20260814
    paths: int = 1200
    horizon: int = 13
    parent_order: int = 100_000
    p00: float = 0.94
    p11: float = 0.82
    eta_calm: float = 7.0
    eta_stress: float = 24.0
    sigma_calm: float = 5.0
    sigma_stress: float = 10.0
    capacity_floor: float = 0.09
    risk_aversion: float = 0.08
    bootstrap_reps: int = 500

    @property
    def transition(self) -> np.ndarray:
        return np.array([[self.p00, 1 - self.p00], [1 - self.p11, self.p11]])

    @property
    def stationary_stress(self) -> float:
        return (1 - self.p00) / ((1 - self.p00) + (1 - self.p11))

    @property
    def emission_means(self) -> np.ndarray:
        return np.array([[1.70, 0.10, 11.42], [2.10, 0.45, 11.10]])

    @property
    def emission_covariances(self) -> np.ndarray:
        return np.array([np.diag([0.20**2, 0.25**2, 0.28**2]), np.diag([0.28**2, 0.32**2, 0.35**2])])


def generate_market(params: Params) -> pd.DataFrame:
    """Generate causal observations and post-decision shocks for every path."""
    rng = np.random.default_rng(params.seed)
    states = np.zeros((params.paths, params.horizon), dtype=int)
    states[:, 0] = rng.random(params.paths) < params.stationary_stress
    for t in range(1, params.horizon):
        prior = states[:, t - 1]
        switch_prob = np.where(prior == 0, 1 - params.p00, params.p11)
        states[:, t] = (rng.random(params.paths) < switch_prob).astype(int)

    records: list[dict[str, float | int]] = []
    for path in range(params.paths):
        for t in range(params.horizon):
            z = states[path, t]
            feature = rng.multivariate_normal(params.emission_means[z], params.emission_covariances[z])
            log_trailing_vol, log_spread, log_depth = feature
            depth = float(np.exp(log_depth))
            cap = max(params.capacity_floor, min(0.18, 0.12 * depth / 100_000.0))
            shock = float(rng.normal(0.0, [params.sigma_calm, params.sigma_stress][z]))
            records.append({
                "path_id": path, "t": t + 1, "regime_true": int(z),
                "log_trailing_vol": float(log_trailing_vol), "log_spread": float(log_spread),
                "log_depth": float(log_depth), "capacity_fraction": cap, "price_shock": shock,
            })
    return pd.DataFrame.from_records(records)


def filter_posteriors(frame: pd.DataFrame, params: Params) -> np.ndarray:
    """Two-state Gaussian-HMM filtering using only observations through t."""
    means, covs, trans = params.emission_means, params.emission_covariances, params.transition
    inv_covs = np.linalg.inv(covs)
    logdets = np.linalg.slogdet(covs)[1]
    posterior = np.array([1 - params.stationary_stress, params.stationary_stress])
    output = []
    for row in frame.sort_values("t").itertuples(index=False):
        y = np.array([row.log_trailing_vol, row.log_spread, row.log_depth])
        prior = posterior @ trans
        log_like = np.array([
            -0.5 * ((y - means[k]) @ inv_covs[k] @ (y - means[k]) + logdets[k] + 3 * np.log(2 * np.pi))
            for k in range(2)
        ])
        log_post = np.log(prior) + log_like
        posterior = np.exp(log_post - logsumexp(log_post))
        output.append(float(posterior[1]))
    return np.asarray(output)


def forecast_stress_probabilities(current_stress_probability: float, params: Params, periods: int) -> np.ndarray:
    """Forecast the filtered state distribution over a remaining MPC horizon.

    The first value is the current causal posterior.  Later values are Markov
    forecasts using no unobserved emissions, states, capacities, or shocks.
    """
    if periods <= 0:
        raise ValueError("periods must be positive")
    distribution = np.array([1.0 - current_stress_probability, current_stress_probability], dtype=float)
    result = np.empty(periods, dtype=float)
    for index in range(periods):
        result[index] = distribution[1]
        distribution = distribution @ params.transition
    return result


def _schedule_objective(v: np.ndarray, eta: np.ndarray, sigma_sq: np.ndarray, remaining: float, risk_aversion: float) -> float:
    future_remaining = remaining - np.cumsum(v)
    return float(np.dot(eta, v * v) + risk_aversion * np.dot(sigma_sq, future_remaining * future_remaining))


def _schedule_gradient(v: np.ndarray, eta: np.ndarray, sigma_sq: np.ndarray, remaining: float, risk_aversion: float) -> np.ndarray:
    future_remaining = remaining - np.cumsum(v)
    return 2 * eta * v - 2 * risk_aversion * np.cumsum((sigma_sq * future_remaining)[::-1])[::-1]


def feasible_allocation(total: float, caps: np.ndarray) -> np.ndarray:
    """Construct a bounded vector that sums exactly to total when feasible."""
    if total > caps.sum() + 1e-10:
        raise ValueError("no feasible bounded allocation")
    out, remaining = np.zeros_like(caps), total
    for index, cap in enumerate(caps):
        future_capacity = caps[index + 1:].sum()
        lower = max(0.0, remaining - future_capacity)
        evenly_spread = remaining / (len(caps) - index)
        out[index] = min(cap, max(lower, evenly_spread))
        remaining -= out[index]
    out[-1] += remaining
    return out


def capped_simplex_projection(vector: np.ndarray, caps: np.ndarray, total: float) -> np.ndarray:
    """Euclidean projection onto {x: 0 <= x <= caps, sum(x)=total}."""
    low, high = float((vector - caps).min()), float(vector.max())
    for _ in range(80):
        threshold = (low + high) / 2
        projected = np.clip(vector - threshold, 0.0, caps)
        if projected.sum() > total:
            low = threshold
        else:
            high = threshold
    projected = np.clip(vector - (low + high) / 2, 0.0, caps)
    # Numerical residual is assigned to a component with available slack.
    residual = total - projected.sum()
    for index in range(len(projected)):
        adjustment = np.clip(residual, -projected[index], caps[index] - projected[index])
        projected[index] += adjustment
        residual -= adjustment
    return projected


def solve_mpc(eta: np.ndarray, sigma_sq: np.ndarray, current_cap: float, remaining: float, floor: float, risk_aversion: float) -> np.ndarray:
    """Solve the small causal box-QP with a deterministic active-set method."""
    periods = len(eta)
    caps = np.full(periods, floor)
    caps[0] = current_cap
    if remaining > caps.sum() + 1e-10:
        raise ValueError(f"infeasible remaining inventory {remaining:.8f}; capacity {caps.sum():.8f}")
    if periods == 1:
        if remaining > current_cap + 1e-10:
            raise ValueError("final-interval cap is infeasible")
        return np.array([remaining])
    lower_now = max(0.0, remaining - caps[1:].sum())
    if current_cap - lower_now < 1e-10:
        return feasible_allocation(remaining, caps)
    lower = np.zeros(periods)
    lower[0] = lower_now
    lower_triangular = np.tril(np.ones((periods, periods)))
    hessian = 2 * np.diag(eta) + 2 * risk_aversion * lower_triangular.T @ np.diag(sigma_sq) @ lower_triangular
    linear = -2 * risk_aversion * remaining * (lower_triangular.T @ sigma_sq)
    fixed = np.full(periods, np.nan)
    tolerance = 1e-10
    for _ in range(periods + 1):
        free = np.flatnonzero(np.isnan(fixed))
        fixed_indices = np.flatnonzero(~np.isnan(fixed))
        fixed_values = fixed[fixed_indices]
        kkt = np.empty((len(free) + 1, len(free) + 1))
        kkt[:-1, :-1] = hessian[np.ix_(free, free)]
        kkt[:-1, -1] = 1.0
        kkt[-1, :-1] = 1.0
        kkt[-1, -1] = 0.0
        rhs = np.empty(len(free) + 1)
        rhs[:-1] = -linear[free]
        if len(fixed_indices):
            rhs[:-1] -= hessian[np.ix_(free, fixed_indices)] @ fixed_values
        rhs[-1] = remaining - fixed_values.sum()
        candidate = np.linalg.solve(kkt, rhs)[:-1]
        below = candidate < lower[free] - tolerance
        above = candidate > caps[free] + tolerance
        if not below.any() and not above.any():
            schedule = fixed.copy()
            schedule[free] = np.clip(candidate, lower[free], caps[free])
            break
        violation = np.maximum(lower[free] - candidate, candidate - caps[free])
        local = int(np.argmax(violation))
        index = free[local]
        fixed[index] = lower[index] if candidate[local] < lower[index] else caps[index]
    else:
        raise RuntimeError("active-set QP failed to converge")
    if abs(schedule.sum() - remaining) > 1e-8 or schedule.min() < -1e-10 or np.any(schedule - caps > 1e-8):
        raise RuntimeError("QP returned a constraint-violating schedule")
    return schedule


def twap_child(remaining: float, current_cap: float, periods_left: int, floor: float) -> float:
    lower_now = max(0.0, remaining - (periods_left - 1) * floor)
    return float(np.clip(remaining / periods_left, lower_now, current_cap))


def simulate_policy(frame: pd.DataFrame, policy: str, params: Params, belief_model: object | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trade a policy path by path, preserving strictly causal information."""
    belief_model = params if belief_model is None else belief_model
    eta_states = np.array([params.eta_calm, params.eta_stress])
    sigma_states = np.array([params.sigma_calm, params.sigma_stress])
    static_p = params.stationary_stress
    trades, costs = [], []
    for path_id, path in frame.groupby("path_id", sort=True):
        path = path.sort_values("t").reset_index(drop=True)
        posterior = filter_posteriors(path, belief_model)
        remaining, is_bps, max_violation = 1.0, 0.0, 0.0
        for index, row in path.iterrows():
            periods_left = len(path) - index
            current_cap = float(row.capacity_fraction)
            if policy == "twap":
                child = twap_child(remaining, current_cap, periods_left, params.capacity_floor)
                posterior_used = np.nan
            else:
                if policy == "static_ac":
                    probabilities = np.full(periods_left, static_p)
                    posterior_used = static_p
                elif policy == "regime_aware_mpc":
                    probabilities = forecast_stress_probabilities(float(posterior[index]), belief_model, periods_left)
                    posterior_used = float(posterior[index])
                elif policy == "oracle_mpc":
                    # Diagnostic lower bound only: this intentionally exposes future
                    # latent states and is never included as a deployable policy.
                    probabilities = path.loc[index:, "regime_true"].to_numpy(dtype=float)
                    posterior_used = np.nan
                else:
                    raise ValueError(f"unknown policy: {policy}")
                eta = (1 - probabilities) * params.eta_calm + probabilities * params.eta_stress
                sigma_sq = (1 - probabilities) * params.sigma_calm**2 + probabilities * params.sigma_stress**2
                child = float(solve_mpc(eta, sigma_sq, current_cap, remaining, params.capacity_floor, params.risk_aversion)[0])
            z = int(row.regime_true)
            post_trade_remaining = remaining - child
            is_bps += eta_states[z] * child**2 + post_trade_remaining * float(row.price_shock)
            max_violation = max(max_violation, max(0.0, child - current_cap), max(0.0, -child))
            trades.append({"policy": policy, "path_id": path_id, "t": int(row.t), "trade_fraction": child,
                           "trade_shares": child * params.parent_order, "remaining_before": remaining,
                           "capacity_fraction": current_cap, "posterior_stress": posterior_used,
                           "regime_true": z})
            remaining -= child
        costs.append({"policy": policy, "path_id": path_id, "implementation_shortfall_bps": is_bps,
                      "completion_error": remaining, "max_constraint_violation": max_violation})
    return pd.DataFrame(trades), pd.DataFrame(costs)


def filter_diagnostics(frame: pd.DataFrame, params: Params) -> pd.DataFrame:
    records = []
    for path_id, path in frame.groupby("path_id", sort=True):
        posterior = filter_posteriors(path, params)
        for row, p in zip(path.sort_values("t").itertuples(index=False), posterior):
            records.append({"path_id": path_id, "t": row.t, "posterior_stress": p, "regime_true": row.regime_true,
                            "predicted_stress": int(p >= 0.5)})
    return pd.DataFrame(records)


def q1_metrics(diagnostics: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    target = diagnostics.regime_true.to_numpy(dtype=int)
    posterior = diagnostics.posterior_stress.to_numpy(dtype=float)
    predicted = diagnostics.predicted_stress.to_numpy(dtype=int)
    stress_recall = float(predicted[target == 1].mean()) if np.any(target == 1) else np.nan
    calm_recall = float((1 - predicted[target == 0]).mean()) if np.any(target == 0) else np.nan
    bins = pd.cut(posterior, bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration = (pd.DataFrame({"bin": bins, "posterior": posterior, "target": target})
                   .groupby("bin", observed=False)
                   .agg(n=("target", "size"), mean_posterior=("posterior", "mean"), observed_stress_rate=("target", "mean"))
                   .reset_index())
    return {
        "accuracy": float((predicted == target).mean()),
        "balanced_accuracy": float((stress_recall + calm_recall) / 2),
        "brier_score": float(np.mean((posterior - target) ** 2)),
        "log_loss": float(-np.mean(target * np.log(np.clip(posterior, 1e-12, 1)) + (1 - target) * np.log(np.clip(1 - posterior, 1e-12, 1)))),
    }, calibration


def write_manifest(params: Params) -> None:
    input_path = DATA / "synthetic_market.csv"
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "random_seed": params.seed,
        "input_files": [{"path": "data/synthetic_market.csv", "sha256": digest, "bytes": input_path.stat().st_size}],
        "runtime": {"name": "python", "version": sys.version.split()[0], "platform": platform.platform(),
                    "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "matplotlib")}},
        "key_parameters": {**asdict(params), "literature_search": {"date": "2026-08-14", "query": "optimal execution Almgren Chriss regime switching liquidity risk", "openalex_candidates": 20, "anysearch_candidates": 10, "cross_validated": 7}},
        "reproduce_command": "python -m unittest discover -s tests -v && python run_research.py --mode full --seed 20260814 && python run_sensitivity.py --paths 150 --seed 20260814 && python run_stress_test.py --paths 300 --seed 20260814 && python run_continual_learning.py --mode full --seed 20260814 && python make_figures.py",
    }
    (RESULTS / "复现清单.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def summarize(costs: pd.DataFrame, params: Params) -> pd.DataFrame:
    rng = np.random.default_rng(params.seed + 77)
    rows = []
    for policy, group in costs.groupby("policy", sort=True):
        values = group.implementation_shortfall_bps.to_numpy()
        var95 = float(np.quantile(values, 0.95))
        cvar95 = float(values[values >= var95].mean())
        boot = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(params.bootstrap_reps)])
        rows.append({"policy": policy, "paths": len(values), "mean_is_bps": float(values.mean()),
                     "std_is_bps": float(values.std(ddof=1)), "var_95_bps": var95, "cvar_95_bps": cvar95,
                     "mean_ci_low_bps": float(np.quantile(boot, 0.025)), "mean_ci_high_bps": float(np.quantile(boot, 0.975)),
                     "max_completion_error": float(group.completion_error.abs().max()),
                     "max_constraint_violation": float(group.max_constraint_violation.max())})
    return pd.DataFrame(rows)


def paired_comparisons(costs: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Paired common-random-number contrasts; negative difference is favorable."""
    pivot = costs.pivot(index="path_id", columns="policy", values="implementation_shortfall_bps")
    comparisons = (("static_ac", "twap"), ("regime_aware_mpc", "twap"), ("regime_aware_mpc", "static_ac"),
                   ("oracle_mpc", "regime_aware_mpc"))
    rng = np.random.default_rng(params.seed + 919)
    rows = []
    for candidate, baseline in comparisons:
        if candidate not in pivot or baseline not in pivot:
            continue
        values = (pivot[candidate] - pivot[baseline]).dropna().to_numpy()
        boot = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(params.bootstrap_reps)])
        rows.append({
            "candidate": candidate, "baseline": baseline, "paths": len(values),
            "mean_difference_bps": float(values.mean()),
            "difference_ci_low_bps": float(np.quantile(boot, 0.025)),
            "difference_ci_high_bps": float(np.quantile(boot, 0.975)),
            "fraction_candidate_lower_cost": float((values < 0).mean()),
        })
    return pd.DataFrame(rows)


def run(params: Params, mode: str) -> None:
    DATA.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    market = generate_market(params)
    market.to_csv(DATA / "synthetic_market.csv", index=False)
    diagnostics = filter_diagnostics(market, params)
    diagnostics.to_csv(RESULTS / "filter_diagnostics.csv", index=False)
    policy_costs, trade_frames = [], []
    for policy in ("twap", "static_ac", "regime_aware_mpc", "oracle_mpc"):
        trades, costs = simulate_policy(market, policy, params)
        policy_costs.append(costs); trade_frames.append(trades)
    costs = pd.concat(policy_costs, ignore_index=True)
    trades = pd.concat(trade_frames, ignore_index=True)
    if costs.completion_error.abs().max() > 1e-8 or costs.max_constraint_violation.max() > 1e-8:
        raise RuntimeError("execution validation failed")
    costs.to_csv(RESULTS / "path_costs.csv", index=False)
    trades.to_csv(RESULTS / "trade_schedules.csv", index=False)
    summarize(costs, params).to_csv(RESULTS / "summary_metrics.csv", index=False)
    paired_comparisons(costs, params).to_csv(RESULTS / "paired_comparisons.csv", index=False)
    metrics, calibration = q1_metrics(diagnostics)
    calibration.to_csv(RESULTS / "filter_calibration.csv", index=False)
    write_manifest(params)
    (RESULTS / "run_summary.json").write_text(json.dumps({"mode": mode, "params": asdict(params), "q1": metrics}, indent=2), encoding="utf-8")
    print(json.dumps({"mode": mode, "paths": params.paths, "q1": {key: round(value, 4) for key, value in metrics.items()}, "output": str(RESULTS)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--paths", type=int)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    defaults = Params(seed=args.seed, paths=args.paths or (12 if args.mode == "smoke" else 1200), bootstrap_reps=50 if args.mode == "smoke" else 500)
    run(defaults, args.mode)


if __name__ == "__main__":
    main()
