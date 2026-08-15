#!/usr/bin/env python3
"""Evaluate a live quote against a validated model; submission is opt-in and paper-only."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import uuid4

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.market_data import AlpacaMarketDataClient, JsonlEventStore
from live.audit import PaperDecision, PaperDecisionLog
from live.paper_broker import AlpacaPaperBroker
from live.predictor import LogisticDirectionModel, live_features
from live.risk import OrderIntent, PaperSubmissionLease, RiskLimits, validate_paper_order


ROOT = Path(__file__).resolve().parents[1]


def _already_reconciled(decision_log: PaperDecisionLog, client_order_id: str,
                        expected_paper_order_id: str | None = None) -> bool:
    """Recognize a competing recovery only when it resolved this same request."""
    latest = next((event for event in reversed(decision_log.read())
                   if event.client_order_id == client_order_id), None)
    if latest is None or latest.event_kind != "reconciliation_result":
        return False
    return expected_paper_order_id is None or latest.paper_order_id == expected_paper_order_id


def submit_and_record(broker: AlpacaPaperBroker, intent: OrderIntent, decision: PaperDecision,
                      decision_log: PaperDecisionLog) -> dict[str, object]:
    """Persist an uncertain attempt before network I/O, then record its outcome."""
    # This fsync'd record intentionally says "unknown" before sending: a
    # process crash after the request leaves this host must never make retrying
    # look safe. Reconcile by client_order_id first.
    assert decision.client_order_id is not None
    decision_log.transition_latest(
        decision.client_order_id,
        "not_requested",
        lambda prior: replace(prior, timestamp=datetime.now(timezone.utc).isoformat(),
                              event_kind="submission_attempt_started",
                              submission_state="unknown_reconciliation_required"),
    )
    try:
        order = broker.submit_market_order(intent, decision.client_order_id)
    except Exception as exc:
        try:
            decision_log.transition_latest(
                decision.client_order_id,
                "unknown_reconciliation_required",
                lambda prior: replace(prior, timestamp=datetime.now(timezone.utc).isoformat(),
                                      event_kind="submission_outcome", submission_error=type(exc).__name__,
                                      submission_state="unknown_reconciliation_required"),
            )
        except ValueError:
            if not _already_reconciled(decision_log, decision.client_order_id):
                raise
        raise
    paper_order_id = order.get("id")
    if not isinstance(paper_order_id, str) or not paper_order_id:
        raise ValueError("paper broker submission response lacks a non-empty id; reconcile before retrying")
    try:
        decision_log.transition_latest(
            decision.client_order_id,
            "unknown_reconciliation_required",
            lambda prior: replace(prior, timestamp=datetime.now(timezone.utc).isoformat(),
                                  event_kind="submission_outcome", submitted=True, paper_order_id=paper_order_id,
                                  submission_state="submitted"),
        )
    except ValueError:
        if not _already_reconciled(decision_log, decision.client_order_id, paper_order_id):
            raise
    return order


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
    bars = client.bars(args.symbol, args.history_bars)
    features = live_features(bars)
    probability = model.predict_probability(features)
    side = "buy" if probability >= .5 else "sell"
    intent = OrderIntent(args.symbol.upper(), side, args.quantity)
    broker = AlpacaPaperBroker() if args.submit_paper_order else None
    lease = PaperSubmissionLease(ROOT / "runtime" / "paper_submission.lock") if broker else nullcontext()
    with lease:
        if broker is None:
            decision = validate_paper_order(intent, quote, 0, 0.0, RiskLimits())
            risk_source = "preview only; no broker account read"
            broker_position = broker_daily_pnl = broker_pending_buy_quantity = broker_pending_sell_quantity = None
        else:
            state = broker.risk_state(intent.symbol)
            decision = validate_paper_order(intent, quote, state.current_position, state.daily_pnl, RiskLimits(),
                                            pending_buy_quantity=state.pending_buy_quantity,
                                            pending_sell_quantity=state.pending_sell_quantity)
            risk_source = "Alpaca paper account, position, and open orders"
            broker_position, broker_daily_pnl = state.current_position, state.daily_pnl
            broker_pending_buy_quantity, broker_pending_sell_quantity = (
                state.pending_buy_quantity, state.pending_sell_quantity)
        report = {"symbol": quote.symbol, "mid_price": quote.mid_price, "spread_bps": quote.spread_bps,
                  "probability_next_bar_up": probability, "proposed_side": side, "risk": decision.reason,
                  "risk_source": risk_source, "submitted": False, "environment": "paper-only"}
        audit = PaperDecision.create(model_path=model_path, symbol=quote.symbol, quote=quote, bars=bars,
                                     features=features.tolist(), probability=probability, side=side,
                                     risk_approved=decision.approved, risk_reason=decision.reason,
                                     risk_source=risk_source, broker_position=broker_position,
                                     broker_daily_pnl=broker_daily_pnl,
                                     broker_pending_buy_quantity=broker_pending_buy_quantity,
                                     broker_pending_sell_quantity=broker_pending_sell_quantity,
                                     client_order_id=f"research-{uuid4().hex}")
        decision_log = PaperDecisionLog(ROOT / "runtime" / "paper_decisions.jsonl")
        audit = decision_log.append(audit)  # Must succeed before any possible network submission.
        if args.submit_paper_order and decision.approved:
            assert broker is not None
            order = submit_and_record(broker, intent, audit, decision_log)
            report["submitted"] = True
            report["paper_order_id"] = order.get("id")
    print(report)


if __name__ == "__main__":
    main()
