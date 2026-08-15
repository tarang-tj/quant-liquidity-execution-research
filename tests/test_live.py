"""Offline tests for the paper-only live-market bridge."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import replace
import json
import hashlib
import multiprocessing
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from live.market_data import AlpacaMarketDataClient, Bar, MarketDataError, Quote
from live.audit import PaperDecision, PaperDecisionLog
from live.paper_broker import AlpacaPaperBroker, PaperRiskState
from live.predictor import LogisticDirectionModel, causal_training_matrix, live_features, train_direction_model
from live.preflight import run_preflight
from live.paper_monitor import run_monitor
from live.reconcile_paper import record_reconciliation_result
from live.replay_decision import replay
from live.run_paper import submit_and_record
from live.risk import OrderIntent, PaperSubmissionLease, RiskLimits, validate_paper_order


def append_in_child(log_path: str, decision: PaperDecision, start: object) -> None:
    start.wait()
    PaperDecisionLog(Path(log_path)).append(decision)


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
        model = train_direction_model(self.bars)
        self.assertEqual(model.feature_mean.shape, (4,))
        self.assertGreaterEqual(model.report.validation_observations, 20)
        self.assertTrue(0 <= model.predict_probability(live_features(self.bars)) <= 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.to_json(path)
            restored = LogisticDirectionModel.from_json(path)
            self.assertAlmostEqual(model.predict_probability(x[-1]), restored.predict_probability(x[-1]))

    def test_read_only_preflight_checks_data_model_and_paper_broker(self) -> None:
        model = train_direction_model(self.bars)
        deployable = replace(model, report=replace(model.report, deployable_for_paper=True))
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            deployable.to_json(model_path)
            data_client = unittest.mock.Mock()
            data_client.latest_quote.return_value = self.quote
            data_client.bars.return_value = self.bars[-100:]
            broker = unittest.mock.Mock()
            broker.risk_state.return_value = PaperRiskState(1, -2, 3, 4)
            report = run_preflight("SPY", model_path, data_client=data_client, broker=broker)
        self.assertTrue(report["ready_for_paper_evaluation"])
        self.assertTrue(report["paper_only"])
        self.assertFalse(report["order_submission_attempted"])
        data_client.latest_quote.assert_called_once_with("SPY")
        data_client.bars.assert_called_once_with("SPY", limit=100)
        broker.risk_state.assert_called_once_with("SPY")
        self.assertFalse(hasattr(broker, "submit_market_order") and broker.submit_market_order.called)

    def test_finite_read_only_monitor_records_each_sample_without_submission(self) -> None:
        model = train_direction_model(self.bars)
        deployable = replace(model, report=replace(model.report, deployable_for_paper=True))
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

    def test_future_bar_does_not_change_earlier_features(self) -> None:
        baseline, _ = causal_training_matrix(self.bars)
        changed = self.bars.copy()
        last = changed[-1]
        changed[-1] = Bar(last.symbol, last.timestamp, last.open, last.close * 10 * 1.001, last.low,
                          last.close * 10, last.volume * 10)
        candidate, _ = causal_training_matrix(changed)
        np.testing.assert_allclose(baseline[:-1], candidate[:-1], atol=0, rtol=0)

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

    def test_quote_schema_normalizes_provider_fields(self) -> None:
        quote = Quote.from_alpaca({"S": "SPY", "bp": 100.0, "ap": 100.02, "bs": 10, "as": 12,
                                   "t": "2026-01-02T14:30:00Z"})
        self.assertEqual(quote.symbol, "SPY")
        self.assertAlmostEqual(quote.spread_bps, 2.0, places=3)

    def test_market_data_excludes_current_partial_bar(self) -> None:
        client = AlpacaMarketDataClient(key="data-key", secret="data-secret")
        now = datetime(2026, 1, 2, 14, 32, 25, tzinfo=timezone.utc)
        payload = {"bars": [
            {"t": "2026-01-02T14:30:00Z", "o": 100, "h": 101, "l": 99, "c": 100, "v": 10},
            {"t": "2026-01-02T14:31:00Z", "o": 100, "h": 101, "l": 99, "c": 100, "v": 10},
            {"t": "2026-01-02T14:32:00Z", "o": 100, "h": 999, "l": 1, "c": 999, "v": 999},
        ]}
        with patch.object(client, "_get_json", return_value=payload):
            bars = client.bars("SPY", limit=3, completed_before=now)
        self.assertEqual([bar.timestamp.minute for bar in bars], [30, 31])

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
            {"equity": "1000", "last_equity": "1025"},
            [{"symbol": "SPY", "side": "buy", "qty": "4", "filled_qty": "1"}], {"qty": "3"},
        ]):
            self.assertEqual(broker.risk_state("SPY"), PaperRiskState(3, -25.0, 4, 0))
        with patch.object(broker, "_get_json", side_effect=MarketDataError("unavailable")):
            with self.assertRaises(MarketDataError):
                broker.risk_state("SPY")

    def test_invalid_open_order_fails_closed(self) -> None:
        broker = AlpacaPaperBroker(key="paper-key", secret="paper-secret")
        with patch.object(broker, "_get_json", side_effect=[
            {"equity": "1000", "last_equity": "1025"},
            [{"symbol": "SPY", "side": "buy", "qty": "1", "filled_qty": "2"}], {"qty": "3"},
        ]):
            with self.assertRaises(MarketDataError):
                broker.risk_state("SPY")

    def test_open_order_query_limit_fails_closed(self) -> None:
        broker = AlpacaPaperBroker(key="paper-key", secret="paper-secret")
        orders = [{"symbol": "SPY", "side": "buy", "qty": "1", "filled_qty": "0"}] * 500
        with patch.object(broker, "_get_json", side_effect=[
            {"equity": "1000", "last_equity": "1025"}, orders, {"qty": "3"},
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
