"""Offline tests for the paper-only live-market bridge."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from live.market_data import AlpacaMarketDataClient, Bar, MarketDataError, Quote
from live.paper_broker import AlpacaPaperBroker, PaperRiskState
from live.predictor import LogisticDirectionModel, causal_training_matrix, live_features, train_direction_model
from live.risk import OrderIntent, RiskLimits, validate_paper_order


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
            response = broker.submit_market_order(OrderIntent("SPY", "buy", 1))
        self.assertEqual(response["id"], "paper-order")
        self.assertEqual(open_mock.call_args.args[0].full_url, "https://paper-api.alpaca.markets/v2/orders")

    def test_paper_risk_state_is_broker_sourced_and_fails_closed(self) -> None:
        broker = AlpacaPaperBroker(key="paper-key", secret="paper-secret")
        with patch.object(broker, "_get_json", side_effect=[{"equity": "1000", "last_equity": "1025"}, {"qty": "3"}]):
            self.assertEqual(broker.risk_state("SPY"), PaperRiskState(3, -25.0))
        with patch.object(broker, "_get_json", side_effect=MarketDataError("unavailable")):
            with self.assertRaises(MarketDataError):
                broker.risk_state("SPY")

    def test_fractional_broker_positions_cannot_bypass_position_limit(self) -> None:
        limits = RiskLimits(max_position_shares=10)
        self.assertFalse(validate_paper_order(OrderIntent("SPY", "buy", 1), self.quote, 9.9, 0,
                                              limits, self.now).approved)
        self.assertFalse(validate_paper_order(OrderIntent("SPY", "sell", 1), self.quote, -9.9, 0,
                                              limits, self.now).approved)


if __name__ == "__main__":
    unittest.main(verbosity=2)
