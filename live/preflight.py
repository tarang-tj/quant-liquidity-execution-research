#!/usr/bin/env python3
"""Read-only paper-trading readiness check; it never creates an order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.market_data import AlpacaMarketDataClient
from live.paper_broker import AlpacaPaperBroker
from live.predictor import LogisticDirectionModel, validate_model_data_alignment, validate_paper_model


ROOT = Path(__file__).resolve().parents[1]


def _result(check: str, ok: bool, detail: str) -> dict[str, object]:
    return {"check": check, "ok": ok, "detail": detail}


def run_preflight(symbol: str, model_path: Path, history_bars: int = 100,
                  data_client: AlpacaMarketDataClient | None = None,
                  broker: AlpacaPaperBroker | None = None,
                  max_training_gap_seconds: float | None = None,
                  max_bar_gap_seconds: float | None = None,
                  feed: str = "iex") -> dict[str, object]:
    """Verify read-only prerequisites for paper execution without any order path."""
    checks: list[dict[str, object]] = []
    model: LogisticDirectionModel | None = None
    try:
        model = LogisticDirectionModel.from_json(model_path)
        validate_paper_model(model, symbol)
        checks.append(_result("model", True, "chronological validation and provenance gate passed"))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        checks.append(_result("model", False, f"model unavailable or invalid: {type(exc).__name__}"))

    try:
        client = data_client or AlpacaMarketDataClient(feed=feed)
        quote = client.latest_quote(symbol)
        bars = client.bars(symbol, limit=history_bars)
        if len(bars) < 21:
            checks.append(_result("market_data", False, "fewer than 21 completed one-minute bars"))
        else:
            checks.append(_result("market_data", True,
                                  f"quote and {len(bars)} completed one-minute bars received for {quote.symbol}"))
            if model is not None:
                try:
                    validate_model_data_alignment(model, bars, max_training_gap_seconds, max_bar_gap_seconds)
                    checks.append(_result("model_data_alignment", True,
                                          "model training window does not extend beyond live bars"))
                except ValueError as exc:
                    checks.append(_result("model_data_alignment", False, str(exc)))
    except Exception as exc:
        checks.append(_result("market_data", False, f"read-only data check failed: {type(exc).__name__}"))

    try:
        paper_broker = broker or AlpacaPaperBroker()
        state = paper_broker.risk_state(symbol)
        checks.append(_result("paper_broker", True,
                              "paper account, position, and open-order reservation snapshot received "
                              f"(position={state.current_position}, pending_buy={state.pending_buy_quantity}, "
                              f"pending_sell={state.pending_sell_quantity})"))
        market_open = paper_broker.market_clock()
        checks.append(_result("market_clock", True,
                              "paper market is open" if market_open else
                              "paper market is closed; no submission should be attempted"))
    except Exception as exc:
        checks.append(_result("paper_broker", False, f"read-only paper-broker check failed: {type(exc).__name__}"))

    return {"symbol": symbol.upper(), "model": str(model_path), "paper_only": True,
            "order_submission_attempted": False, "ready_for_paper_evaluation": all(check["ok"] for check in checks),
            "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Alpaca paper-execution readiness check")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--feed", choices=("iex", "sip"), default="iex",
                        help="Alpaca market-data feed; SIP requires the appropriate entitlement")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--history-bars", type=int, default=100)
    parser.add_argument("--max-training-gap-hours", type=float,
                        help="optional model freshness SLA; fail closed when exceeded")
    parser.add_argument("--max-bar-gap-minutes", type=float,
                        help="optional live-bar continuity SLA; fail closed when exceeded")
    args = parser.parse_args()
    model_path = args.model or ROOT / "models" / f"{args.symbol.upper()}_logistic.json"
    gap_seconds = None if args.max_training_gap_hours is None else args.max_training_gap_hours * 3_600
    bar_gap_seconds = None if args.max_bar_gap_minutes is None else args.max_bar_gap_minutes * 60
    report = run_preflight(args.symbol, model_path, args.history_bars,
                           feed=args.feed, max_training_gap_seconds=gap_seconds,
                           max_bar_gap_seconds=bar_gap_seconds)
    print(json.dumps(report, sort_keys=True))
    if not report["ready_for_paper_evaluation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
