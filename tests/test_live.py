"""Offline tests for the paper-only live-market bridge."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import replace
import asyncio
import json
import hashlib
import multiprocessing
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError
from pathlib import Path

import numpy as np

from live.market_data import AlpacaMarketDataClient, Bar, JsonlEventStore, MarketDataError, Quote
from live.audit import PaperDecision, PaperDecisionLog
from live.paper_broker import AlpacaPaperBroker, PaperRiskState
from live.predictor import (LogisticDirectionModel, causal_training_matrix, live_features,
                            direction_from_probability, train_direction_model, training_config_sha256,
                            training_data_sha256,
                            validate_model_data_alignment)
from live.predictor import validate_paper_model
from live.preflight import run_preflight
from live.health import run_health
from live.paper_monitor import run_monitor
from live.promote_model import promote_if_qualified
from live.retrain_model import run_retraining
from live.reconcile_paper import record_reconciliation_result
from live.replay_decision import replay
from live.run_paper import submit_and_record, validate_quality_report
from live.risk import OrderIntent, PaperSubmissionLease, RiskLimits, validate_paper_order
from live.score_predictions import apply_quality_gate, build_quality_report, score_predictions, write_quality_report
from live.stream_quotes import collect_quotes
from live.walk_forward import walk_forward_evaluate


def append_in_child(log_path: str, decision: PaperDecision, start: object) -> None:
    start.wait()
    PaperDecisionLog(Path(log_path)).append(decision)


def append_quotes_in_child(log_path: str, quote: Quote, start: object) -> None:
    start.wait()
    from live.market_data import JsonlEventStore
    JsonlEventStore(Path(log_path)).append_quotes([quote])


def reconcile_in_child(log_path: str, client_order_id: str, start: object, outcomes: object) -> None:
    start.wait()
    try:
        record_reconciliation_result(PaperDecisionLog(Path(log_path)), client_order_id, {
            "id": "paper-order", "status": "filled", "client_order_id": client_order_id,
        })
    except ValueError:
        outcomes.put(False)
    else:
        outcomes.put(True)


def hold_submission_lease(lock_path: str, acquired: object, release: object) -> None:
    with PaperSubmissionLease(Path(lock_path)):
        acquired.set()
        release.wait(timeout=10)


class LiveBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
        returns = rng.normal(0.0001, 0.002, 180)
        close = 100 * np.exp(np.cumsum(returns))
        self.bars = [Bar("SPY", start + timedelta(minutes=i), close[i] * .999, close[i] * 1.001,
                         close[i] * .998, close[i], float(10_000 + i % 11)) for i in range(len(close))]
        self.now = self.bars[-1].timestamp + timedelta(seconds=1)
        self.quote = Quote("SPY", self.bars[-1].close - .01, self.bars[-1].close + .01, 100, 100, self.now, "fixture")

    def test_training_is_chronological_and_serializable(self) -> None:
        x, y = causal_training_matrix(self.bars)
        self.assertEqual(len(x), len(y))
        model = train_direction_model(self.bars, feed="iex")
        self.assertEqual(model.feature_mean.shape, (4,))
        self.assertGreaterEqual(model.report.validation_observations, 20)
        self.assertEqual(model.report.training_symbol, "SPY")
        self.assertEqual(model.report.training_timeframe, "1Min")
        self.assertEqual(model.report.training_adjustment, "all")
        self.assertEqual(model.report.training_start, self.bars[0].timestamp.isoformat())
        self.assertEqual(model.report.training_end, self.bars[-1].timestamp.isoformat())
        self.assertEqual(model.report.training_data_sha256, training_data_sha256(self.bars))
        self.assertEqual(model.report.training_config_sha256, training_config_sha256(
            lookback=20, validation_fraction=.30, learning_rate=.08, iterations=1_500, l2=.02,
            timeframe="1Min"))
        self.assertEqual(len(model.report.training_data_sha256), 64)
        self.assertTrue(0 <= model.predict_probability(live_features(self.bars)) <= 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.to_json(path)
            restored = LogisticDirectionModel.from_json(path)
            self.assertAlmostEqual(model.predict_probability(x[-1]), restored.predict_probability(x[-1]))

            raw = json.loads(path.read_text(encoding="utf-8"))
            for key in ("training_symbol", "training_timeframe", "training_start", "training_end",
                        "training_data_sha256", "training_config_sha256", "created_at"):
                raw["report"].pop(key)
            path.write_text(json.dumps(raw), encoding="utf-8")
            legacy = LogisticDirectionModel.from_json(path)
            self.assertIsNone(legacy.report.training_data_sha256)

    def test_direction_gate_abstains_only_with_a_configured_edge(self) -> None:
        self.assertEqual(direction_from_probability(0.80), "buy")
        self.assertEqual(direction_from_probability(0.20), "sell")
        self.assertEqual(direction_from_probability(0.50), "buy")
        self.assertEqual(direction_from_probability(0.52, 0.05), "hold")
        self.assertEqual(direction_from_probability(0.55, 0.05), "hold")
        self.assertEqual(direction_from_probability(0.56, 0.05), "buy")
        self.assertEqual(direction_from_probability(0.44, 0.05), "sell")
        with self.assertRaises(ValueError):
            direction_from_probability(0.5, 0.5)
        with self.assertRaises(ValueError):
            direction_from_probability(float("nan"), 0.05)

    def test_live_prediction_scoring_requires_the_immediate_next_bar(self) -> None:
        model = train_direction_model(self.bars, feed="iex")
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            model.to_json(model_path)
            decision = PaperDecision.create(
                model_path=model_path, symbol="SPY", quote=self.quote,
                bars=self.bars[:-1], features=live_features(self.bars[:-1]).tolist(),
                probability=0.8, side="buy", risk_approved=False, risk_reason="test",
                risk_source="test", broker_position=None, broker_daily_pnl=None,
            )
            scored = score_predictions([decision], self.bars)
            self.assertEqual(scored["scored"], 1)
            self.assertEqual(scored["directional_scored"], 1)
            self.assertEqual(scored["pending"], 0)
            self.assertAlmostEqual(float(scored["accuracy"]), 1.0)
            pending = score_predictions([decision], self.bars[:-1])
            self.assertEqual(pending["scored"], 0)
            self.assertEqual(pending["pending"], 1)
            mismatched = score_predictions([replace(decision, symbol="QQQ")], self.bars, symbol="SPY")
            self.assertEqual(mismatched["scored"], 0)
            self.assertEqual(mismatched["symbol_mismatch"], 1)
            self.assertTrue(apply_quality_gate(scored, minimum_scored=1)["quality_gate"]["passed"])
            self.assertFalse(apply_quality_gate(pending, minimum_scored=1)["quality_gate"]["passed"])
            report = build_quality_report([decision], scored, symbol="SPY", minimum_scored=1,
                                          generated_at=self.now)
            self.assertTrue(report["quality_gate"]["passed"])
            self.assertEqual(report["model_sha256"], decision.model_sha256)
            report_path = Path(directory) / "quality.json"
            write_quality_report(report, report_path)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["report_type"],
                             "live_quality_gate")

    def test_quality_report_is_fresh_and_pinned_before_paper_submission(self) -> None:
        model = train_direction_model(self.bars, feed="iex")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model_path, report_path = folder / "model.json", folder / "quality.json"
            model.to_json(model_path)
            report = {
                "report_type": "live_quality_gate", "symbol": "SPY",
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "generated_at": self.now.isoformat(),
                "quality_gate": {"passed": True},
            }
            write_quality_report(report, report_path)
            checked = validate_quality_report(report_path, symbol="SPY",
                                              model_sha256=report["model_sha256"], now=self.now,
                                              max_age_seconds=60)
            self.assertTrue(checked["quality_gate"]["passed"])
            for mutation in ({"symbol": "QQQ"}, {"model_sha256": "0" * 64},
                             {"quality_gate": {"passed": False}}):
                invalid = dict(report)
                invalid.update(mutation)
                write_quality_report(invalid, report_path)
                with self.assertRaises(ValueError):
                    validate_quality_report(report_path, symbol="SPY",
                                            model_sha256=report["model_sha256"], now=self.now,
                                            max_age_seconds=60)

    def test_health_snapshot_requires_quality_and_valid_journal_without_submission(self) -> None:
        model = train_direction_model(self.bars, feed="iex")
        deployable = replace(model, report=replace(model.report, validation_observations=100,
                                                   deployable_for_paper=True))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model_path, report_path, log_path = folder / "model.json", folder / "quality.json", folder / "decisions.jsonl"
            deployable.to_json(model_path)
            decision = PaperDecision.create(
                model_path=model_path, symbol="SPY", quote=self.quote, bars=self.bars,
                features=live_features(self.bars).tolist(), probability=.8, side="buy",
                risk_approved=False, risk_reason="fixture", risk_source="fixture",
                broker_position=0, broker_daily_pnl=0,
            )
            journal = PaperDecisionLog(log_path)
            committed = journal.append(decision)
            report = build_quality_report([committed], {"scored": 1, "accuracy": 1.0, "brier": 0.0},
                                          symbol="SPY", minimum_scored=1,
                                          generated_at=datetime.now(timezone.utc))
            write_quality_report(report, report_path)
            data_client = unittest.mock.Mock()
            data_client.latest_quote.return_value = self.quote
            data_client.bars.return_value = self.bars[-100:]
            broker = unittest.mock.Mock()
            broker.risk_state.return_value = PaperRiskState(0, 0, 0, 0, 1_000)
            broker.market_clock.return_value = True
            broker.trading_calendar.return_value = []
            health = run_health("SPY", model_path, report_path, log_path,
                                data_client=data_client, broker=broker)
            self.assertTrue(health["ready_for_paper_session"])
            self.assertTrue(health["paper_only"])
            self.assertFalse(health["order_submission_attempted"])
            self.assertTrue(all(check["ok"] for check in health["checks"]))

    def test_model_rotation_is_atomic_and_preserves_previous_artifact_on_failure(self) -> None:
        model = train_direction_model(self.bars)
        replacement = train_direction_model(self.bars, iterations=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.to_json(path)
            previous_bytes = path.read_bytes()
            with patch("live.predictor.os.replace", side_effect=OSError("simulated rotation failure")):
                with self.assertRaises(OSError):
                    replacement.to_json(path)
            self.assertEqual(path.read_bytes(), previous_bytes)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            restored = LogisticDirectionModel.from_json(path)
            self.assertAlmostEqual(restored.predict_probability(live_features(self.bars)),
                                   model.predict_probability(live_features(self.bars)))

    def test_failed_model_promotion_preserves_active_artifact(self) -> None:
        model = train_direction_model(self.bars)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "active.json"
            model.to_json(target)
            previous_bytes = target.read_bytes()
            report = promote_if_qualified(
                self.bars, symbol="SPY", target_path=target, training_bars=120,
                evaluation_bars=20, minimum_accuracy=1.0, minimum_net_return_bps=1e12,
            )
            self.assertFalse(report["promoted"])
            self.assertEqual(target.read_bytes(), previous_bytes)
            self.assertIn("rejection_reason", report)

    def test_retraining_loop_is_bounded_and_auditable_without_submission(self) -> None:
        data_client = unittest.mock.Mock()
        data_client.bars.return_value = self.bars
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "active.json"
            report_path = Path(directory) / "promotion.jsonl"
            promoted = {"promoted": True, "symbol": "SPY", "checks": {"all": True}}
            with patch("live.retrain_model.promote_if_qualified", return_value=promoted) as promote:
                reports = run_retraining(
                    "SPY", target, iterations=2, interval_seconds=0, bars=180,
                    data_client=data_client, report_path=report_path,
                    sleep_fn=lambda _: self.fail("bounded zero interval should not sleep"),
                )
            self.assertEqual(len(reports), 2)
            self.assertEqual(promote.call_count, 2)
            self.assertEqual(data_client.bars.call_count, 2)
            self.assertTrue(all(report["paper_only"] for report in reports))
            self.assertTrue(all(not report["order_submission_attempted"] for report in reports))
            persisted = [json.loads(line) for line in report_path.read_text().splitlines()]
            self.assertEqual([report["cycle"] for report in persisted], [0, 1])
            self.assertEqual([report["schema"] for report in persisted], ["paper_model_promotion.v1"] * 2)

    def test_read_only_preflight_checks_data_model_and_paper_broker(self) -> None:
        model = train_direction_model(self.bars, feed="iex")
        deployable = replace(model, report=replace(model.report, validation_observations=100, deployable_for_paper=True))
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            deployable.to_json(model_path)
            data_client = unittest.mock.Mock()
            data_client.latest_quote.return_value = self.quote
            data_client.bars.return_value = self.bars[-100:]
            broker = unittest.mock.Mock()
            broker.risk_state.return_value = PaperRiskState(1, -2, 3, 4, 1000)
            broker.market_clock.return_value = True
            broker.trading_calendar.return_value = [{"date": "2026-01-02", "open": "09:30", "close": "16:00"}]
            report = run_preflight("SPY", model_path, data_client=data_client, broker=broker)
        self.assertTrue(report["ready_for_paper_evaluation"])
        self.assertTrue(report["paper_only"])
        self.assertFalse(report["order_submission_attempted"])
        data_client.latest_quote.assert_called_once_with("SPY")
        data_client.bars.assert_called_once_with("SPY", limit=100)
        broker.risk_state.assert_called_once_with("SPY")
        broker.market_clock.assert_called_once_with()
        broker.trading_calendar.assert_called_once()
        self.assertFalse(hasattr(broker, "submit_market_order") and broker.submit_market_order.called)

    def test_finite_read_only_monitor_records_each_sample_without_submission(self) -> None:
        model = train_direction_model(self.bars, feed="iex")
        deployable = replace(model, report=replace(model.report, validation_observations=100, deployable_for_paper=True))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model_path, log_path = folder / "model.json", folder / "monitor.jsonl"
            deployable.to_json(model_path)
            data_client = unittest.mock.Mock()
            data_client.latest_quote.side_effect = [self.quote, self.quote]
            data_client.bars.side_effect = [self.bars[-100:], self.bars[-100:]]
            broker = unittest.mock.Mock()
            broker.risk_state.side_effect = [PaperRiskState(0, 0, 0, 0), PaperRiskState(1, -2, 0, 0)]
            sleeps: list[float] = []
            samples = run_monitor(
                "SPY", model_path, iterations=2, interval_seconds=3,
                data_client=data_client, broker=broker,
                decision_log=PaperDecisionLog(log_path), sleep_fn=sleeps.append,
                now_fn=lambda: self.now,
            )
            self.assertEqual(len(samples), 2)
            self.assertEqual(sleeps, [3])
            self.assertTrue(all(sample["paper_only"] and not sample["submitted"] for sample in samples))
            self.assertEqual(len(PaperDecisionLog(log_path).read()), 2)
            self.assertEqual(data_client.latest_quote.call_count, 2)
            self.assertEqual(broker.risk_state.call_count, 2)
            self.assertFalse(hasattr(broker, "submit_market_order") and broker.submit_market_order.called)

    def test_monitor_rejects_nonfinite_interval(self) -> None:
        with self.assertRaises(ValueError):
            run_monitor("SPY", Path("missing-model.json"), iterations=1, interval_seconds=float("nan"),
                        data_client=unittest.mock.Mock(), broker=unittest.mock.Mock())

    def test_model_deserialization_rejects_nonfinite_or_invalid_deployable_parameters(self) -> None:
        model = train_direction_model(self.bars)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.to_json(path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["feature_scale"][0] = 0
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                LogisticDirectionModel.from_json(path)

    def test_model_deserialization_rejects_invalid_provenance(self) -> None:
        model = train_direction_model(self.bars)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.to_json(path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["report"]["training_data_sha256"] = "not-a-digest"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                LogisticDirectionModel.from_json(path)
            model.to_json(path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["report"]["training_start"] = "2026-01-02T14:30:00"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                LogisticDirectionModel.from_json(path)
            model.to_json(path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["report"]["training_start"], raw["report"]["training_end"] = (
                raw["report"]["training_end"], raw["report"]["training_start"])
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                LogisticDirectionModel.from_json(path)
            model.to_json(path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["report"]["validation_observations"] = 100
            raw["report"]["deployable_for_paper"] = True
            raw["report"]["validation_accuracy"] = 0.51
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                LogisticDirectionModel.from_json(path)
            model.to_json(path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["intercept"] = True
            raw["report"]["deployable_for_paper"] = 1
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                LogisticDirectionModel.from_json(path)

    def test_future_bar_does_not_change_earlier_features(self) -> None:
        baseline, _ = causal_training_matrix(self.bars)
        changed = self.bars.copy()
        last = changed[-1]
        changed[-1] = Bar(last.symbol, last.timestamp, last.open, last.close * 10 * 1.001, last.low,
                          last.close * 10, last.volume * 10)
        candidate, _ = causal_training_matrix(changed)
        np.testing.assert_allclose(baseline[:-1], candidate[:-1], atol=0, rtol=0)

    def test_training_rejects_mixed_or_nonchronological_bars(self) -> None:
        mixed = self.bars.copy()
        mixed[0] = Bar("QQQ", mixed[0].timestamp, mixed[0].open, mixed[0].high,
                       mixed[0].low, mixed[0].close, mixed[0].volume)
        with self.assertRaises(ValueError):
            train_direction_model(mixed)
        unsorted = self.bars.copy()
        unsorted[0], unsorted[1] = unsorted[1], unsorted[0]
        with self.assertRaises(ValueError):
            train_direction_model(unsorted)

    def test_model_data_alignment_rejects_future_and_stale_training_windows(self) -> None:
        model = train_direction_model(self.bars)
        future_report = replace(model.report, training_end=(self.bars[-1].timestamp + timedelta(minutes=1)).isoformat())
        with self.assertRaises(ValueError):
            validate_model_data_alignment(replace(model, report=future_report), self.bars)
        old_report = replace(model.report, training_end=(self.bars[-2].timestamp).isoformat())
        with self.assertRaises(ValueError):
            validate_model_data_alignment(replace(model, report=old_report), self.bars, max_training_gap_seconds=1)
        with self.assertRaises(ValueError):
            validate_model_data_alignment(model, self.bars, max_training_gap_seconds=float("nan"))

    def test_model_data_alignment_rejects_excessive_live_bar_gap(self) -> None:
        gapped = self.bars[:120]
        gapped = gapped[:60] + [replace(bar, timestamp=bar.timestamp + timedelta(minutes=2))
                                for bar in gapped[60:]]
        model = train_direction_model(self.bars[:120])
        with self.assertRaises(ValueError):
            validate_model_data_alignment(model, gapped, max_bar_gap_seconds=60)

    def test_paper_model_requires_matching_complete_provenance(self) -> None:
        model = train_direction_model(self.bars)
        with self.assertRaises(ValueError):
            validate_paper_model(model, "QQQ")
        with self.assertRaises(ValueError):
            validate_paper_model(model, "SPY", "5Min")
        legacy_report = replace(model.report, training_symbol=None, training_timeframe=None,
                                training_start=None, training_end=None, training_data_sha256=None,
                                training_config_sha256=None, created_at=None)
        legacy = replace(model, report=legacy_report)
        with self.assertRaises(ValueError):
            validate_paper_model(legacy, "SPY")

    def test_paper_model_requires_matching_market_data_feed_provenance(self) -> None:
        model = train_direction_model(self.bars, feed="sip")
        model = replace(model, report=replace(model.report, validation_observations=100,
                                             deployable_for_paper=True))
        validate_paper_model(model, "SPY", feed="sip")
        with self.assertRaises(ValueError):
            validate_paper_model(model, "SPY", feed="iex")
        legacy = replace(model, report=replace(model.report, training_feed=None))
        with self.assertRaises(ValueError):
            validate_paper_model(legacy, "SPY", feed="sip")

    def test_paper_model_requires_matching_corporate_action_adjustment(self) -> None:
        model = train_direction_model(self.bars, feed="iex", adjustment="all")
        model = replace(model, report=replace(model.report, validation_observations=100,
                                             deployable_for_paper=True))
        validate_paper_model(model, "SPY", feed="iex", adjustment="all")
        with self.assertRaises(ValueError):
            validate_paper_model(model, "SPY", feed="iex", adjustment="raw")
        legacy = replace(model, report=replace(model.report, training_adjustment=None))
        with self.assertRaises(ValueError):
            validate_paper_model(legacy, "SPY", feed="iex", adjustment="all")

    def test_walk_forward_evaluation_is_causal_and_against_baseline(self) -> None:
        summary = walk_forward_evaluate(self.bars, training_bars=120, evaluation_bars=20, iterations=300)
        self.assertEqual(summary.symbol, "SPY")
        self.assertEqual(summary.predictions, len(self.bars) - 1 - 120)
        self.assertEqual(summary.evaluation_block_size, 20)
        self.assertEqual(summary.evaluation_predictions, summary.predictions)
        self.assertGreaterEqual(summary.retrain_blocks, 1)
        self.assertTrue(0 <= summary.accuracy <= 1)
        self.assertTrue(0 <= summary.brier <= 1)
        self.assertTrue(0.5 <= summary.majority_baseline_accuracy <= 1)
        self.assertEqual(summary.transaction_cost_bps, 5.0)
        self.assertGreater(summary.turnover_units, 0)
        self.assertLessEqual(summary.net_return_bps, summary.gross_return_bps)
        self.assertGreaterEqual(summary.max_drawdown_bps, 0)
        changed = self.bars.copy()
        for index in range(121, len(changed)):
            bar = changed[index]
            changed[index] = Bar(bar.symbol, bar.timestamp, bar.open, bar.high * 10, bar.low,
                                  bar.close * 10, bar.volume * 10)
        baseline_model = train_direction_model(self.bars[:120], iterations=300)
        changed_model = train_direction_model(changed[:120], iterations=300)
        np.testing.assert_allclose(live_features(self.bars[:121]), live_features(changed[:121]), atol=0, rtol=0)
        self.assertAlmostEqual(
            baseline_model.predict_probability(live_features(self.bars[:121])),
            changed_model.predict_probability(live_features(changed[:121])), places=12)
        for kwargs in ({"validation_fraction": 0}, {"learning_rate": float("nan")},
                       {"iterations": 0}, {"l2": -1}, {"lookback": "20"},
                       {"transaction_cost_bps": float("nan")}, {"transaction_cost_bps": -1}):
            with self.assertRaises(ValueError):
                walk_forward_evaluate(self.bars, training_bars=120, evaluation_bars=20, **kwargs)

    def test_risk_gates_are_fail_closed(self) -> None:
        allowed = validate_paper_order(OrderIntent("SPY", "buy", 2), self.quote, 0, 0, RiskLimits(), self.now)
        self.assertTrue(allowed.approved)
        stale = Quote("SPY", self.quote.bid_price, self.quote.ask_price, 100, 100, self.now - timedelta(seconds=10), "fixture")
        self.assertFalse(validate_paper_order(OrderIntent("SPY", "buy", 2), stale, 0, 0, RiskLimits(), self.now).approved)
        future = Quote("SPY", self.quote.bid_price, self.quote.ask_price, 100, 100, self.now + timedelta(seconds=1), "fixture")
        self.assertFalse(validate_paper_order(OrderIntent("SPY", "buy", 2), future, 0, 0, RiskLimits(), self.now).approved)
        for invalid_pnl in (float("nan"), float("inf"), float("-inf")):
            self.assertFalse(validate_paper_order(OrderIntent("SPY", "buy", 2), self.quote, 0, invalid_pnl, RiskLimits(), self.now).approved)
        self.assertFalse(validate_paper_order(OrderIntent("SPY", "buy", float("nan")), self.quote, 0, 0,
                                              RiskLimits(), self.now).approved)
        self.assertFalse(validate_paper_order(OrderIntent("SPY", "buy", 1), self.quote, 0, 0,
                                              RiskLimits(max_position_shares=float("nan")), self.now).approved)
        self.assertFalse(validate_paper_order(OrderIntent("SPY", "buy", 1), self.quote, 0, 0,
                                              RiskLimits(max_position_shares=10), self.now,
                                              pending_buy_quantity=10).approved)
        self.assertFalse(validate_paper_order(OrderIntent("SPY", "sell", 1), self.quote, 0, 0,
                                              RiskLimits(max_position_shares=10), self.now,
                                              pending_sell_quantity=10).approved)
        old = os.environ.get("TRADING_KILL_SWITCH")
        try:
            os.environ["TRADING_KILL_SWITCH"] = "1"
            self.assertFalse(validate_paper_order(OrderIntent("SPY", "buy", 2), self.quote, 0, 0, RiskLimits(), self.now).approved)
        finally:
            if old is None:
                os.environ.pop("TRADING_KILL_SWITCH", None)
            else:
                os.environ["TRADING_KILL_SWITCH"] = old

    def test_risk_gate_rejects_closed_or_invalid_market_session(self) -> None:
        closed = validate_paper_order(OrderIntent("SPY", "buy", 1), self.quote, 0, 0,
                                      RiskLimits(), self.now, market_open=False)
        self.assertFalse(closed.approved)
        self.assertEqual(closed.reason, "market is closed")
        invalid = validate_paper_order(OrderIntent("SPY", "buy", 1), self.quote, 0, 0,
                                       RiskLimits(), self.now, market_open=1)  # type: ignore[arg-type]
        self.assertFalse(invalid.approved)
        self.assertEqual(invalid.reason, "market session state is invalid")

    def test_risk_gate_enforces_broker_buying_power(self) -> None:
        rejected = validate_paper_order(OrderIntent("SPY", "buy", 2), self.quote, 0, 0,
                                        RiskLimits(), self.now, buying_power=1.0)
        self.assertFalse(rejected.approved)
        self.assertEqual(rejected.reason, "order exceeds broker buying power")
        invalid = validate_paper_order(OrderIntent("SPY", "buy", 1), self.quote, 0, 0,
                                       RiskLimits(), self.now, buying_power=float("nan"))
        self.assertFalse(invalid.approved)
        self.assertEqual(invalid.reason, "buying power is invalid")

    def test_risk_notional_uses_executable_side_not_midpoint(self) -> None:
        quote = Quote("SPY", 99.0, 101.0, 100, 100, self.now, "fixture")
        limits = RiskLimits(max_order_notional=100.0, max_spread_bps=300.0)
        rejected = validate_paper_order(OrderIntent("SPY", "buy", 1), quote, 0, 0, limits, self.now)
        self.assertFalse(rejected.approved)
        self.assertEqual(rejected.reason, "order notional limit exceeded")
        accepted = validate_paper_order(OrderIntent("SPY", "sell", 1), quote, 0, 0, limits, self.now)
        self.assertTrue(accepted.approved)

    def test_quote_schema_normalizes_provider_fields(self) -> None:
        quote = Quote.from_alpaca({"S": "SPY", "bp": 100.0, "ap": 100.02, "bs": 10, "as": 12,
                                   "t": "2026-01-02T14:30:00Z"})
        self.assertEqual(quote.symbol, "SPY")
        self.assertAlmostEqual(quote.spread_bps, 2.0, places=3)

    def test_market_data_feed_is_explicitly_limited_to_supported_alpaca_feeds(self) -> None:
        with self.assertRaises(ValueError):
            AlpacaMarketDataClient(key="data-key", secret="data-secret", feed="unknown")
        with self.assertRaises(ValueError):
            AlpacaMarketDataClient(key="data-key", secret="data-secret", adjustment="unknown")
        client = AlpacaMarketDataClient(key="data-key", secret="data-secret", feed="sip", adjustment="split")
        self.assertEqual(client.feed, "sip")
        self.assertEqual(client.adjustment, "split")

    def test_market_data_retries_transient_transport_failure_with_bound(self) -> None:
        client = AlpacaMarketDataClient(key="data-key", secret="data-secret", max_retries=1,
                                        retry_backoff_seconds=0)
        response = unittest.mock.Mock()
        response.__enter__ = lambda _: response
        response.__exit__ = lambda *_args: None
        response.read.return_value = b'{"quote":{"S":"SPY","bp":100,"ap":100.02,"bs":1,"as":2,"t":"2026-01-02T14:30:00Z"}}'
        with patch("live.market_data.urlopen", side_effect=[URLError("temporary"), response]) as open_mock:
            quote = client.latest_quote("SPY")
        self.assertEqual(quote.symbol, "SPY")
        self.assertEqual(open_mock.call_count, 2)

    def test_market_data_excludes_current_partial_bar(self) -> None:
        client = AlpacaMarketDataClient(key="data-key", secret="data-secret")
        now = datetime(2026, 1, 2, 14, 32, 25, tzinfo=timezone.utc)
        payload = {"bars": [
            {"t": "2026-01-02T14:30:00Z", "o": 100, "h": 101, "l": 99, "c": 100, "v": 10},
            {"t": "2026-01-02T14:31:00Z", "o": 100, "h": 101, "l": 99, "c": 100, "v": 10},
            {"t": "2026-01-02T14:32:00Z", "o": 100, "h": 999, "l": 1, "c": 999, "v": 999},
        ]}
        with patch.object(client, "_get_json", return_value=payload) as get_json:
            bars = client.bars("SPY", limit=3, completed_before=now)
        self.assertEqual([bar.timestamp.minute for bar in bars], [30, 31])
        self.assertEqual(get_json.call_args.args[1]["adjustment"], "all")

    def test_quote_event_store_serializes_concurrent_durable_appends(self) -> None:
        from live.market_data import JsonlEventStore
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quotes.jsonl"
            start = multiprocessing.Event()
            processes = [multiprocessing.Process(
                target=append_quotes_in_child,
                args=(str(path), replace(self.quote, timestamp=self.now + timedelta(microseconds=index)), start),
            ) for index in range(8)]
            for process in processes:
                process.start()
            start.set()
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 8)
            records = [json.loads(line) for line in lines]
            self.assertTrue(all(record["symbol"] == "SPY" for record in records))
            # A subsequent append must observe the complete prior file.
            JsonlEventStore(path).append_quotes([self.quote])
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 9)

    def test_websocket_stream_reconnects_with_a_bounded_budget(self) -> None:
        quote_message = json.dumps({"T": "q", "S": "SPY", "bp": 100.0, "ap": 100.02,
                                    "bs": 10, "as": 12, "t": "2026-01-02T14:30:00Z"})

        class FakeSocket:
            def __init__(self) -> None:
                self.sent: list[str] = []
                self.yielded = False

            async def __aenter__(self) -> "FakeSocket":
                return self

            async def __aexit__(self, *_args: object) -> None:
                return None

            async def send(self, message: str) -> None:
                self.sent.append(message)

            async def recv(self) -> str:
                return json.dumps([{"T": "success"}])

            def __aiter__(self) -> "FakeSocket":
                return self

            async def __anext__(self) -> str:
                if not self.yielded:
                    self.yielded = True
                    return quote_message
                raise ConnectionError("simulated disconnect")

        sockets: list[FakeSocket] = []

        class FakeWebsockets:
            def connect(self, *_args: object, **_kwargs: object) -> FakeSocket:
                socket = FakeSocket()
                sockets.append(socket)
                return socket

        async def collect() -> list[Quote]:
            client = AlpacaMarketDataClient(key="data-key", secret="data-secret")
            output: list[Quote] = []
            try:
                async for quote in client.stream_quotes(["spy", "SPY"], max_reconnects=1,
                                                        reconnect_backoff_seconds=0):
                    output.append(quote)
            except MarketDataError:
                pass
            return output

        with patch.dict(sys.modules, {"websockets": FakeWebsockets()}):
            quotes = asyncio.run(collect())
        self.assertEqual(len(quotes), 2)
        self.assertEqual(len(sockets), 2)
        self.assertEqual(json.loads(sockets[0].sent[1])["quotes"], ["SPY"])

    def test_bounded_quote_collector_durably_records_stream_and_never_submits(self) -> None:
        fixture_quote, fixture_now = self.quote, self.now

        class FakeClient:
            async def stream_quotes(self, symbols: list[str], *, max_reconnects: int) -> object:
                self.symbols = symbols
                self.max_reconnects = max_reconnects
                for index in range(3):
                    yield replace(fixture_quote, timestamp=fixture_now + timedelta(microseconds=index))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "quotes.jsonl"
            client = FakeClient()
            report = asyncio.run(collect_quotes(
                client, ["spy", "SPY"],
                JsonlEventStore(output),
                max_quotes=2, timeout_seconds=5, max_reconnects=1,
            ))
            self.assertEqual(client.symbols, ["SPY"])
            self.assertEqual(client.max_reconnects, 1)
            self.assertEqual(report["quotes_written"], 2)
            self.assertFalse(report["timed_out"])
            self.assertTrue(report["paper_only"])
            self.assertFalse(report["order_submission_attempted"])
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)

    def test_quote_collector_rejects_out_of_order_or_unrequested_stream_events(self) -> None:
        fixture_quote, fixture_now = self.quote, self.now

        class FakeClient:
            def __init__(self, quotes: list[Quote]) -> None:
                self.quotes = quotes

            async def stream_quotes(self, symbols: list[str], *, max_reconnects: int) -> object:
                for quote in self.quotes:
                    yield quote

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "quotes.jsonl"
            client = FakeClient([
                replace(fixture_quote, timestamp=fixture_now),
                replace(fixture_quote, timestamp=fixture_now - timedelta(seconds=1)),
            ])
            with self.assertRaises(MarketDataError):
                asyncio.run(collect_quotes(client, ["SPY"], JsonlEventStore(output),
                                           max_quotes=2, timeout_seconds=5))
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)

            wrong_symbol = FakeClient([replace(fixture_quote, symbol="QQQ")])
            with self.assertRaises(MarketDataError):
                asyncio.run(collect_quotes(wrong_symbol, ["SPY"], JsonlEventStore(output),
                                           max_quotes=1, timeout_seconds=5))

    def test_bar_schema_rejects_invalid_provider_payload(self) -> None:
        with self.assertRaises(MarketDataError):
            Bar.from_alpaca("SPY", {"t": "2026-01-02T14:30:00Z", "o": 100, "h": 99, "l": 101, "c": 100, "v": 1})

    def test_paper_broker_endpoint_cannot_be_overridden(self) -> None:
        broker = AlpacaPaperBroker(key="paper-key", secret="paper-secret")
        broker.base_url = "https://api.alpaca.markets"  # Must have no effect even if a caller assigns it.
        with patch("live.paper_broker.urlopen") as open_mock:
            open_mock.return_value.__enter__.return_value.read.return_value = b'{"id":"paper-order"}'
            response = broker.submit_market_order(OrderIntent("SPY", "buy", 1), "paper-test-order")
        self.assertEqual(response["id"], "paper-order")
        self.assertEqual(open_mock.call_args.args[0].full_url, "https://paper-api.alpaca.markets/v2/orders")
        self.assertEqual(json.loads(open_mock.call_args.args[0].data.decode("utf-8"))["client_order_id"], "paper-test-order")

    def test_paper_broker_market_clock_requires_boolean_state(self) -> None:
        broker = AlpacaPaperBroker(key="paper-key", secret="paper-secret")
        with patch.object(broker, "_get_json", return_value={"is_open": True}):
            self.assertTrue(broker.market_clock())
        with patch.object(broker, "_get_json", return_value={"is_open": "true"}):
            with self.assertRaises(MarketDataError):
                broker.market_clock()

    def test_paper_trading_calendar_validates_sessions_and_range(self) -> None:
        broker = AlpacaPaperBroker(key="paper-key", secret="paper-secret")
        with patch.object(broker, "_get_json", return_value=[
            {"date": "2026-01-02", "open": "09:30", "close": "16:00"},
        ]) as get_json:
            sessions = broker.trading_calendar(datetime(2026, 1, 1).date(), datetime(2026, 1, 3).date())
        self.assertEqual(sessions[0]["date"], "2026-01-02")
        self.assertIn("start=2026-01-01", get_json.call_args.args[0])
        with self.assertRaises(ValueError):
            broker.trading_calendar(datetime(2026, 1, 3).date(), datetime(2026, 1, 1).date())
        with patch.object(broker, "_get_json", return_value=[{"date": "bad", "open": "", "close": ""}]):
            with self.assertRaises(MarketDataError):
                broker.trading_calendar(datetime(2026, 1, 1).date(), datetime(2026, 1, 3).date())
        with patch.object(broker, "_get_json", return_value=[
            {"date": "2026-01-02", "open": "not-a-time", "close": "16:00"},
        ]):
            with self.assertRaises(MarketDataError):
                broker.trading_calendar(datetime(2026, 1, 1).date(), datetime(2026, 1, 3).date())
        with patch.object(broker, "_get_json", return_value=[
            {"date": "2026-01-02", "open": "16:00", "close": "09:30"},
        ]):
            with self.assertRaises(MarketDataError):
                broker.trading_calendar(datetime(2026, 1, 1).date(), datetime(2026, 1, 3).date())

    def test_reconciliation_uses_fixed_paper_endpoint(self) -> None:
        broker = AlpacaPaperBroker(key="paper-key", secret="paper-secret")
        with patch("live.paper_broker.urlopen") as open_mock:
            open_mock.return_value.__enter__.return_value.read.return_value = b'{"id":"paper-order","status":"new","client_order_id":"paper-test-order"}'
            response = broker.order_by_client_order_id("paper-test-order")
        self.assertEqual(response["status"], "new")
        self.assertEqual(open_mock.call_args.args[0].full_url,
                         "https://paper-api.alpaca.markets/v2/orders:by_client_order_id?client_order_id=paper-test-order")

    def test_paper_risk_state_is_broker_sourced_and_fails_closed(self) -> None:
        broker = AlpacaPaperBroker(key="paper-key", secret="paper-secret")
        with patch.object(broker, "_get_json", side_effect=[
            {"equity": "1000", "last_equity": "1025", "buying_power": "900"},
            [{"symbol": "SPY", "side": "buy", "qty": "4", "filled_qty": "1"}], {"qty": "3"},
        ]):
            self.assertEqual(broker.risk_state("SPY"), PaperRiskState(3, -25.0, 4, 0, 900))
        with patch.object(broker, "_get_json", side_effect=MarketDataError("unavailable")):
            with self.assertRaises(MarketDataError):
                broker.risk_state("SPY")

    def test_invalid_open_order_fails_closed(self) -> None:
        broker = AlpacaPaperBroker(key="paper-key", secret="paper-secret")
        with patch.object(broker, "_get_json", side_effect=[
            {"equity": "1000", "last_equity": "1025", "buying_power": "900"},
            [{"symbol": "SPY", "side": "buy", "qty": "1", "filled_qty": "2"}], {"qty": "3"},
        ]):
            with self.assertRaises(MarketDataError):
                broker.risk_state("SPY")

    def test_open_order_query_limit_fails_closed(self) -> None:
        broker = AlpacaPaperBroker(key="paper-key", secret="paper-secret")
        orders = [{"symbol": "SPY", "side": "buy", "qty": "1", "filled_qty": "0"}] * 500
        with patch.object(broker, "_get_json", side_effect=[
            {"equity": "1000", "last_equity": "1025", "buying_power": "900"}, orders, {"qty": "3"},
        ]):
            with self.assertRaises(MarketDataError):
                broker.risk_state("SPY")

    def test_fractional_broker_positions_cannot_bypass_position_limit(self) -> None:
        limits = RiskLimits(max_position_shares=10)
        self.assertFalse(validate_paper_order(OrderIntent("SPY", "buy", 1), self.quote, 9.9, 0,
                                              limits, self.now).approved)
        self.assertFalse(validate_paper_order(OrderIntent("SPY", "sell", 1), self.quote, -9.9, 0,
                                              limits, self.now).approved)

    def test_decision_log_replays_exactly_and_detects_model_change(self) -> None:
        model = train_direction_model(self.bars)
        features = live_features(self.bars)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model_path, log_path = folder / "model.json", folder / "decisions.jsonl"
            model.to_json(model_path)
            PaperDecisionLog(log_path).append(PaperDecision.create(
                model_path=model_path, symbol="SPY", quote=self.quote, bars=self.bars, features=features.tolist(),
                probability=model.predict_probability(features), side="buy", risk_approved=True,
                risk_reason="fixture", risk_source="fixture", broker_position=None, broker_daily_pnl=None))
            self.assertAlmostEqual(replay(model_path, log_path), model.predict_probability(features))
            model_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                replay(model_path, log_path)

    def test_decision_log_detects_record_tampering_and_feature_mismatch(self) -> None:
        model = train_direction_model(self.bars)
        features = live_features(self.bars)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model_path, log_path = folder / "model.json", folder / "decisions.jsonl"
            model.to_json(model_path)
            record = PaperDecision.create(model_path=model_path, symbol="SPY", quote=self.quote, bars=self.bars,
                                          features=features.tolist(), probability=model.predict_probability(features),
                                          side="buy", risk_approved=True, risk_reason="fixture",
                                          risk_source="fixture", broker_position=None, broker_daily_pnl=None,
                                          client_order_id="paper-failure-test")
            PaperDecisionLog(log_path).append(record)
            log_path.write_text(log_path.read_text(encoding="utf-8").replace("fixture", "edited", 1), encoding="utf-8")
            with self.assertRaises(ValueError):
                PaperDecisionLog(log_path).read()
            log_path.unlink()
            PaperDecisionLog(log_path).append(replace(record, features=[0.0] * 4))
            with self.assertRaises(ValueError):
                replay(model_path, log_path)

    def test_decision_log_reads_legacy_record_without_new_optional_field(self) -> None:
        model = train_direction_model(self.bars)
        features = live_features(self.bars)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model_path, log_path = folder / "model.json", folder / "legacy.jsonl"
            model.to_json(model_path)
            decision = PaperDecision.create(
                model_path=model_path, symbol="SPY", quote=self.quote, bars=self.bars, features=features.tolist(),
                probability=model.predict_probability(features), side="buy", risk_approved=True,
                risk_reason="fixture", risk_source="fixture", broker_position=None, broker_daily_pnl=None)
            legacy = dict(decision.__dict__) if hasattr(decision, "__dict__") else {
                field: getattr(decision, field) for field in decision.__dataclass_fields__
            }
            legacy.pop("broker_order_status")
            legacy["record_hash"] = None
            legacy["record_hash"] = hashlib.sha256(
                json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            log_path.write_text(json.dumps(legacy, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            records = PaperDecisionLog(log_path).read()
            self.assertEqual(len(records), 1)
            self.assertIsNone(records[0].broker_order_status)
            self.assertAlmostEqual(replay(model_path, log_path), model.predict_probability(features))

    def test_failed_paper_submission_writes_terminal_outcome(self) -> None:
        model = train_direction_model(self.bars)
        features = live_features(self.bars)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model_path, log_path = folder / "model.json", folder / "decisions.jsonl"
            model.to_json(model_path)
            record = PaperDecision.create(model_path=model_path, symbol="SPY", quote=self.quote, bars=self.bars,
                                          features=features.tolist(), probability=model.predict_probability(features),
                                          side="buy", risk_approved=True, risk_reason="fixture",
                                          risk_source="fixture", broker_position=None, broker_daily_pnl=None,
                                          client_order_id="paper-failure-test")
            journal = PaperDecisionLog(log_path)
            record = journal.append(record)
            broker = AlpacaPaperBroker(key="paper-key", secret="paper-secret")
            observed_pre_network: list[str] = []

            def network_failure(*_args: object) -> None:
                observed_pre_network.extend(event.event_kind for event in journal.read())
                raise MarketDataError("network failure")

            with patch.object(broker, "submit_market_order", side_effect=network_failure):
                with self.assertRaises(MarketDataError):
                    submit_and_record(broker, OrderIntent("SPY", "buy", 1), record, journal)
            events = journal.read()
            self.assertEqual(observed_pre_network, ["decision", "submission_attempt_started"])
            self.assertEqual([event.event_kind for event in events],
                             ["decision", "submission_attempt_started", "submission_outcome"])
            self.assertEqual(events[1].submission_state, "unknown_reconciliation_required")
            self.assertEqual(events[-1].submission_error, "MarketDataError")
            self.assertEqual(events[-1].submission_state, "unknown_reconciliation_required")

    def test_reconciliation_appends_verified_terminal_result(self) -> None:
        model = train_direction_model(self.bars)
        features = live_features(self.bars)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model_path, log_path = folder / "model.json", folder / "decisions.jsonl"
            model.to_json(model_path)
            journal = PaperDecisionLog(log_path)
            decision = journal.append(PaperDecision.create(
                model_path=model_path, symbol="SPY", quote=self.quote, bars=self.bars, features=features.tolist(),
                probability=model.predict_probability(features), side="buy", risk_approved=True,
                risk_reason="fixture", risk_source="fixture", broker_position=None, broker_daily_pnl=None,
                client_order_id="paper-reconcile-test"))
            journal.append(replace(decision, event_kind="submission_attempt_started",
                                   submission_state="unknown_reconciliation_required"))
            result = record_reconciliation_result(journal, "paper-reconcile-test", {
                "id": "paper-order", "status": "filled", "client_order_id": "paper-reconcile-test",
            })
            self.assertEqual(result.event_kind, "reconciliation_result")
            self.assertEqual(result.submission_state, "reconciled:filled")
            self.assertEqual(result.broker_order_status, "filled")
            self.assertEqual([event.event_kind for event in journal.read()],
                             ["decision", "submission_attempt_started", "reconciliation_result"])
            with self.assertRaises(ValueError):
                record_reconciliation_result(journal, "paper-reconcile-test", {
                    "id": "other-order", "status": "filled", "client_order_id": "other-client",
                })

    def test_submission_and_reconciliation_race_resolves_only_once(self) -> None:
        model = train_direction_model(self.bars)
        features = live_features(self.bars)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model_path, log_path = folder / "model.json", folder / "decisions.jsonl"
            model.to_json(model_path)
            journal = PaperDecisionLog(log_path)
            decision = journal.append(PaperDecision.create(
                model_path=model_path, symbol="SPY", quote=self.quote, bars=self.bars, features=features.tolist(),
                probability=model.predict_probability(features), side="buy", risk_approved=True,
                risk_reason="fixture", risk_source="fixture", broker_position=None, broker_daily_pnl=None,
                client_order_id="paper-submit-race"))
            broker = AlpacaPaperBroker(key="paper-key", secret="paper-secret")

            def reconcile_before_response(*_args: object) -> dict[str, object]:
                record_reconciliation_result(journal, "paper-submit-race", {
                    "id": "paper-order", "status": "filled", "client_order_id": "paper-submit-race",
                })
                return {"id": "paper-order"}

            with patch.object(broker, "submit_market_order", side_effect=reconcile_before_response):
                self.assertEqual(submit_and_record(broker, OrderIntent("SPY", "buy", 1), decision, journal),
                                 {"id": "paper-order"})
            self.assertEqual([event.event_kind for event in journal.read()],
                             ["decision", "submission_attempt_started", "reconciliation_result"])

    def test_concurrent_reconciliation_commits_only_one_result(self) -> None:
        model = train_direction_model(self.bars)
        features = live_features(self.bars)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model_path, log_path = folder / "model.json", folder / "decisions.jsonl"
            model.to_json(model_path)
            journal = PaperDecisionLog(log_path)
            decision = journal.append(PaperDecision.create(
                model_path=model_path, symbol="SPY", quote=self.quote, bars=self.bars, features=features.tolist(),
                probability=model.predict_probability(features), side="buy", risk_approved=True,
                risk_reason="fixture", risk_source="fixture", broker_position=None, broker_daily_pnl=None,
                client_order_id="paper-concurrent-reconcile"))
            journal.append(replace(decision, event_kind="submission_attempt_started",
                                   submission_state="unknown_reconciliation_required"))
            context = multiprocessing.get_context("spawn")
            start, outcomes = context.Event(), context.Queue()
            workers = [context.Process(target=reconcile_in_child,
                                       args=(str(log_path), "paper-concurrent-reconcile", start, outcomes))
                       for _ in range(2)]
            for worker in workers:
                worker.start()
            start.set()
            for worker in workers:
                worker.join(timeout=10)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual(sorted(outcomes.get(timeout=2) for _ in workers), [False, True])
            self.assertEqual([event.event_kind for event in journal.read()],
                             ["decision", "submission_attempt_started", "reconciliation_result"])

    def test_concurrent_writers_preserve_a_single_hash_chain(self) -> None:
        model = train_direction_model(self.bars)
        features = live_features(self.bars)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model_path, log_path = folder / "model.json", folder / "decisions.jsonl"
            model.to_json(model_path)
            record = PaperDecision.create(model_path=model_path, symbol="SPY", quote=self.quote, bars=self.bars,
                                          features=features.tolist(), probability=model.predict_probability(features),
                                          side="buy", risk_approved=True, risk_reason="fixture",
                                          risk_source="fixture", broker_position=None, broker_daily_pnl=None)
            context = multiprocessing.get_context("spawn")
            start = context.Event()
            workers = [context.Process(target=append_in_child, args=(str(log_path), record, start)) for _ in range(2)]
            for worker in workers:
                worker.start()
            start.set()
            for worker in workers:
                worker.join(timeout=10)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual(len(PaperDecisionLog(log_path).read()), 2)

    def test_submission_lease_excludes_another_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("spawn")
            acquired, release = context.Event(), context.Event()
            worker = context.Process(target=hold_submission_lease,
                                     args=(str(Path(directory) / "submission.lock"), acquired, release))
            worker.start()
            self.assertTrue(acquired.wait(timeout=10))
            with self.assertRaises(RuntimeError):
                with PaperSubmissionLease(Path(directory) / "submission.lock"):
                    pass
            release.set()
            worker.join(timeout=10)
            self.assertEqual(worker.exitcode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
