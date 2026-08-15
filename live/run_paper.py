#!/usr/bin/env python3
"""Evaluate a live quote against a validated model; submission is opt-in and paper-only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.market_data import AlpacaMarketDataClient, JsonlEventStore
from live.paper_broker import AlpacaPaperBroker
from live.predictor import LogisticDirectionModel, live_features
from live.risk import OrderIntent, RiskLimits, validate_paper_order


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--history-bars", type=int, default=100)
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--submit-paper-order", action="store_true", help="explicitly submit a qualifying paper order")
    args = parser.parse_args()
    client = AlpacaMarketDataClient()
    quote = client.latest_quote(args.symbol)
    JsonlEventStore(ROOT / "runtime" / "quotes.jsonl").append_quotes([quote])
    model_path = args.model or ROOT / "models" / f"{args.symbol.upper()}_logistic.json"
    model = LogisticDirectionModel.from_json(model_path)
    if not model.report.deployable_for_paper:
        raise SystemExit("model did not pass the minimum chronological paper-research quality gate")
    probability = model.predict_probability(live_features(client.bars(args.symbol, args.history_bars)))
    side = "buy" if probability >= .5 else "sell"
    intent = OrderIntent(args.symbol.upper(), side, args.quantity)
    broker = AlpacaPaperBroker() if args.submit_paper_order else None
    if broker is None:
        decision = validate_paper_order(intent, quote, 0, 0.0, RiskLimits())
        risk_source = "preview only; no broker account read"
    else:
        state = broker.risk_state(intent.symbol)
        decision = validate_paper_order(intent, quote, state.current_position, state.daily_pnl, RiskLimits())
        risk_source = "Alpaca paper account and position"
    report = {"symbol": quote.symbol, "mid_price": quote.mid_price, "spread_bps": quote.spread_bps,
              "probability_next_bar_up": probability, "proposed_side": side, "risk": decision.reason,
              "risk_source": risk_source, "submitted": False, "environment": "paper-only"}
    if args.submit_paper_order and decision.approved:
        assert broker is not None
        order = broker.submit_market_order(intent)
        report["submitted"] = True
        report["paper_order_id"] = order.get("id")
    print(report)


if __name__ == "__main__":
    main()
