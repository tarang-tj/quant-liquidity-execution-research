"""Fast deterministic unit and invariant tests for the research prototype."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from run_research import (
    Params,
    _schedule_gradient,
    _schedule_objective,
    filter_posteriors,
    forecast_stress_probabilities,
    generate_market,
    paired_comparisons,
    simulate_policy,
    solve_mpc,
)
from run_continual_learning import fit_filter, run_cycles


class ResearchInvariantsTest(unittest.TestCase):
    def setUp(self) -> None:
        # Completion is only guaranteed when T * capacity_floor >= 1.
        self.params = Params(paths=4, horizon=13, bootstrap_reps=20)
        self.market = generate_market(self.params)

    def test_filter_is_causal(self) -> None:
        path = self.market[self.market.path_id == 0].copy()
        reference = filter_posteriors(path, self.params)
        changed = path.copy()
        changed.loc[changed.t >= 4, ["log_trailing_vol", "log_spread", "log_depth"]] += 9.0
        candidate = filter_posteriors(changed, self.params)
        np.testing.assert_allclose(reference[:3], candidate[:3], atol=0.0, rtol=0.0)
        self.assertGreater(np.max(np.abs(reference[3:] - candidate[3:])), 0.1)

    def test_markov_forecast_preserves_stationary_distribution(self) -> None:
        forecast = forecast_stress_probabilities(self.params.stationary_stress, self.params, 6)
        np.testing.assert_allclose(forecast, self.params.stationary_stress, atol=1e-12)

    def test_analytic_gradient_matches_central_difference(self) -> None:
        eta = np.array([7.0, 10.0, 14.0])
        sigma_sq = np.array([25.0, 40.0, 60.0])
        point = np.array([0.15, 0.16, 0.19])
        analytic = _schedule_gradient(point, eta, sigma_sq, 0.5, 0.08)
        step = 1e-6
        numerical = np.array([
            (_schedule_objective(point + np.eye(3)[i] * step, eta, sigma_sq, .5, .08)
             - _schedule_objective(point - np.eye(3)[i] * step, eta, sigma_sq, .5, .08)) / (2 * step)
            for i in range(3)
        ])
        np.testing.assert_allclose(analytic, numerical, atol=1e-6)

    def test_mpc_schedule_respects_hard_constraints(self) -> None:
        schedule = solve_mpc(np.array([9.0, 11.0, 12.0]), np.array([30.0, 45.0, 50.0]), .17, .31, .09, .08)
        self.assertAlmostEqual(float(schedule.sum()), .31, places=9)
        self.assertTrue(np.all(schedule >= 0))
        self.assertLessEqual(schedule[0], .17 + 1e-10)
        self.assertTrue(np.all(schedule[1:] <= .09 + 1e-10))

    def test_every_policy_completes_and_paired_output_is_aligned(self) -> None:
        cost_frames = []
        for policy in ("twap", "static_ac", "regime_aware_mpc", "oracle_mpc"):
            trades, costs = simulate_policy(self.market, policy, self.params)
            self.assertEqual(len(trades), self.params.paths * self.params.horizon)
            self.assertLessEqual(float(costs.completion_error.abs().max()), 1e-8)
            self.assertLessEqual(float(costs.max_constraint_violation.max()), 1e-8)
            cost_frames.append(costs)
        contrasts = paired_comparisons(__import__("pandas").concat(cost_frames), self.params)
        self.assertTrue((contrasts.paths == self.params.paths).all())

    def test_completed_batch_fit_is_valid_and_holdout_loop_is_feasible(self) -> None:
        learned = fit_filter(self.market)
        np.testing.assert_allclose(learned.transition.sum(axis=1), 1.0, atol=1e-12)
        self.assertTrue(np.all(np.linalg.eigvalsh(learned.emission_covariances) > 0))
        with tempfile.TemporaryDirectory() as directory:
            report = run_cycles(Params(paths=4, bootstrap_reps=10), cycles=1, initial_train_paths=8, eval_paths=4,
                                output_dir=Path(directory))
            self.assertEqual(len(report), 1)
            self.assertLessEqual(float(report.max_completion_error.max()), 1e-8)
            self.assertLessEqual(float(report.max_constraint_violation.max()), 1e-8)

    def test_two_cycle_snapshots_exclude_current_holdout_and_policy_ignores_labels(self) -> None:
        params = Params(seed=313, paths=6, bootstrap_reps=10)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            report = run_cycles(params, cycles=2, initial_train_paths=10, eval_paths=6, output_dir=output_dir)
            snapshot = json.loads((output_dir / "online_filter_snapshot.json").read_text())
            first, second = snapshot["cycle_snapshots"]
            self.assertEqual(len(first["training_batches"]), 1)
            self.assertEqual(first["holdout_batch"]["seed"], params.seed + 10_001)
            self.assertEqual(len(second["training_batches"]), 2)
            self.assertEqual(second["training_batches"][1]["seed"], first["holdout_batch"]["seed"])
            model = fit_filter(generate_market(Params(seed=99, paths=20)))
            holdout = generate_market(Params(seed=100, paths=4))
            original = simulate_policy(holdout, "regime_aware_mpc", params, belief_model=model)[0]
            altered = holdout.copy(); altered["regime_true"] = 1 - altered["regime_true"]
            candidate = simulate_policy(altered, "regime_aware_mpc", params, belief_model=model)[0]
            np.testing.assert_allclose(original.trade_fraction, candidate.trade_fraction, atol=0.0, rtol=0.0)
            self.assertEqual(len(report), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
