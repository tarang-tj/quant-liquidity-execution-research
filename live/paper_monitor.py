#!/usr/bin/env python3
"""Finite, read-only paper-market monitor for staged model evaluation.

This module deliberately has no order-submission dependency.  Each completed
sample is written to the same tamper-evident local decision journal used by the
one-shot runner, but every sample is marked ``not_requested`` and ``submitted``
is always false.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from math import isfinite
from pathlib import Path
import sys
import time
from typing import Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.audit import PaperDecision, PaperDecisionLog, file_sha256
from live.market_data import AlpacaMarketDataClient
from live.paper_broker import AlpacaPaperBroker
from live.predictor import (LogisticDirectionModel, live_features, validate_model_data_alignment,
                            direction_from_probability, validate_paper_model)
from live.risk import OrderIntent, RiskDecision, RiskLimits, validate_paper_order


ROOT = Path(__file__).resolve().parents[1]


def evaluate_read_only_once(
    symbol: str,
    model_path: Path,
    *,
    data_client: AlpacaMarketDataClient,
    broker: AlpacaPaperBroker,
    decision_log: PaperDecisionLog,
    history_bars: int = 100,
    quantity: int = 1,
    now_fn: Callable[[], datetime] | None = None,
    model: LogisticDirectionModel | None = None,
    expected_model_hash: str | None = None,
    max_training_gap_seconds: float | None = None,
    max_bar_gap_seconds: float | None = None,
    min_direction_edge: float = 0.0,
    feed: str = "iex",
) -> dict[str, object]:
    """Fetch one fresh snapshot, score it, and append evidence without POSTing."""
    active_model = model or LogisticDirectionModel.from_json(model_path)
    validate_paper_model(active_model, symbol, feed=feed)
    if expected_model_hash is not None and file_sha256(model_path) != expected_model_hash:
        raise RuntimeError("model file changed during monitor; refusing to mix model versions")
    quote = data_client.latest_quote(symbol)
    bars = data_client.bars(symbol, limit=history_bars)
    validate_model_data_alignment(active_model, bars, max_training_gap_seconds, max_bar_gap_seconds)
    features = live_features(bars)
    probability = active_model.predict_probability(features)
    side = direction_from_probability(probability, min_direction_edge)
    intent = OrderIntent(symbol.upper(), side, quantity)
    state = broker.risk_state(intent.symbol)
    now = now_fn() if now_fn is not None else None
    decision = (RiskDecision(False, "predictive signal abstained: below minimum direction edge")
                if side == "hold" else validate_paper_order(
                    intent, quote, state.current_position, state.daily_pnl, RiskLimits(),
                    now=now,  # type: ignore[arg-type]
                    pending_buy_quantity=state.pending_buy_quantity,
                    pending_sell_quantity=state.pending_sell_quantity,
                ))
    if expected_model_hash is not None and file_sha256(model_path) != expected_model_hash:
        raise RuntimeError("model file changed during monitor; refusing to journal mixed model evidence")
    audit = PaperDecision.create(
        model_path=model_path,
        symbol=quote.symbol,
        quote=quote,
        bars=bars,
        features=features.tolist(),
        probability=probability,
        side=side,
        risk_approved=decision.approved,
        risk_reason=decision.reason,
        risk_source="Alpaca paper account, position, and open orders (read-only monitor)",
        broker_position=state.current_position,
        broker_daily_pnl=state.daily_pnl,
        broker_pending_buy_quantity=state.pending_buy_quantity,
        broker_pending_sell_quantity=state.pending_sell_quantity,
        client_order_id=None,
        model_hash=expected_model_hash,
    )
    committed = decision_log.append(audit)
    return {
        "timestamp": committed.timestamp,
        "symbol": quote.symbol,
        "probability_next_bar_up": probability,
        "proposed_side": side,
        "risk_approved": decision.approved,
        "risk_reason": decision.reason,
        "submitted": False,
        "paper_only": True,
        "order_submission_attempted": False,
        "record_hash": committed.record_hash,
    }


def run_monitor(
    symbol: str,
    model_path: Path,
    *,
    iterations: int,
    interval_seconds: float,
    history_bars: int = 100,
    quantity: int = 1,
    data_client: AlpacaMarketDataClient | None = None,
    broker: AlpacaPaperBroker | None = None,
    decision_log: PaperDecisionLog | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] | None = None,
    max_training_gap_seconds: float | None = None,
    max_bar_gap_seconds: float | None = None,
    min_direction_edge: float = 0.0,
    feed: str = "iex",
) -> list[dict[str, object]]:
    """Run a finite paper monitor; unbounded daemon operation is not supported."""
    if not isinstance(iterations, int) or isinstance(iterations, bool) or not 1 <= iterations <= 10_000:
        raise ValueError("iterations must be an integer between 1 and 10,000")
    if not isfinite(interval_seconds) or interval_seconds < 0 or interval_seconds > 86_400:
        raise ValueError("interval_seconds must be between 0 and 86,400")
    if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
        raise ValueError("quantity must be a positive whole number")
    client = data_client or AlpacaMarketDataClient(feed=feed)
    paper_broker = broker or AlpacaPaperBroker()
    log = decision_log or PaperDecisionLog(ROOT / "runtime" / "paper_monitor_decisions.jsonl")
    # Pair the bytes used to load the model with the hash that will be
    # journaled.  If an atomic model rotation overlaps startup, fail closed
    # instead of labeling version A predictions as version B evidence.
    pinned_model_hash = file_sha256(model_path)
    model = LogisticDirectionModel.from_json(model_path)
    validate_paper_model(model, symbol, feed=feed)
    if file_sha256(model_path) != pinned_model_hash:
        raise RuntimeError("model file changed during monitor startup; refusing to mix model versions")
    samples: list[dict[str, object]] = []
    for index in range(iterations):
        if file_sha256(model_path) != pinned_model_hash:
            raise RuntimeError("model file changed during monitor; refusing to mix model versions")
        samples.append(evaluate_read_only_once(
            symbol,
            model_path,
            data_client=client,
            broker=paper_broker,
            decision_log=log,
            history_bars=history_bars,
            quantity=quantity,
            now_fn=now_fn,
            model=model,
            expected_model_hash=pinned_model_hash,
            max_training_gap_seconds=max_training_gap_seconds,
            max_bar_gap_seconds=max_bar_gap_seconds,
            min_direction_edge=min_direction_edge,
            feed=feed,
        ))
        if index + 1 < iterations and interval_seconds:
            sleep_fn(interval_seconds)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite read-only Alpaca paper-market monitor")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--feed", choices=("iex", "sip"), default="iex",
                        help="Alpaca market-data feed; SIP requires the appropriate entitlement")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--iterations", type=int, default=1,
                        help="finite number of snapshots; defaults to one")
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--history-bars", type=int, default=100)
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--max-training-gap-hours", type=float,
                        help="optional model freshness SLA; fail closed when exceeded")
    parser.add_argument("--max-bar-gap-minutes", type=float,
                        help="optional live-bar continuity SLA; fail closed when exceeded")
    parser.add_argument("--min-direction-edge", type=float, default=0.0,
                        help="abstain (hold) unless probability is this far from 0.5; default 0")
    parser.add_argument("--decision-log", type=Path)
    args = parser.parse_args()
    model_path = args.model or ROOT / "models" / f"{args.symbol.upper()}_logistic.json"
    samples = run_monitor(
        args.symbol,
        model_path,
        iterations=args.iterations,
        interval_seconds=args.interval_seconds,
        history_bars=args.history_bars,
        quantity=args.quantity,
        decision_log=PaperDecisionLog(args.decision_log) if args.decision_log else None,
        max_training_gap_seconds=(None if args.max_training_gap_hours is None
                                  else args.max_training_gap_hours * 3_600),
        max_bar_gap_seconds=(None if args.max_bar_gap_minutes is None
                             else args.max_bar_gap_minutes * 60),
        min_direction_edge=args.min_direction_edge,
        feed=args.feed,
    )
    for sample in samples:
        print(json.dumps(sample, sort_keys=True))


if __name__ == "__main__":
    main()
