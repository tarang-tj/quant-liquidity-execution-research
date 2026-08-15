#!/usr/bin/env python3
"""Score journaled live predictions once their next completed bar exists.

This is a read-only, causal evaluator.  It never changes the decision journal
and never submits an order.  A prediction is scored only when the immediately
following one-minute bar is available; gaps remain pending instead of being
silently treated as a target.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.audit import PaperDecision, PaperDecisionLog
from live.market_data import AlpacaMarketDataClient, Bar


def _summary(scored: list[tuple[PaperDecision, Bar, float]]) -> dict[str, object]:
    directional = [(decision, bar, return_bps) for decision, bar, return_bps in scored
                   if decision.proposed_side in {"buy", "sell"}]
    correct = [((decision.proposed_side == "buy") == (bar.close > float(decision.completed_bars[-1]["close"])))
               for decision, bar, _ in directional]
    brier = [((decision.probability_next_bar_up - (1.0 if bar.close > float(decision.completed_bars[-1]["close"])
                else 0.0)) ** 2) for decision, bar, _ in scored]
    returns = [return_bps if decision.proposed_side == "buy" else -return_bps
               for decision, _, return_bps in directional]
    return {
        "scored": len(scored),
        "directional_scored": len(directional),
        "hold_scored": len(scored) - len(directional),
        "coverage": (len(directional) / len(scored)) if scored else None,
        "accuracy": (sum(correct) / len(correct)) if correct else None,
        "brier": (sum(brier) / len(brier)) if brier else None,
        "mean_directional_return_bps": (sum(returns) / len(returns)) if returns else None,
    }


def score_predictions(decisions: Iterable[PaperDecision], bars: Iterable[Bar]) -> dict[str, object]:
    """Return causal quality metrics and pending counts for journaled decisions."""
    ordered_bars = sorted(bars, key=lambda bar: bar.timestamp)
    by_timestamp = {bar.timestamp: bar for bar in ordered_bars}
    scored: list[tuple[PaperDecision, Bar, float]] = []
    pending = 0
    invalid = 0
    for decision in decisions:
        if decision.event_kind != "decision":
            continue
        if not decision.completed_bars:
            invalid += 1
            continue
        try:
            raw_timestamp = decision.completed_bars[-1]["timestamp"]
            raw_close = decision.completed_bars[-1]["close"]
            cutoff = datetime.fromisoformat(str(raw_timestamp).replace("Z", "+00:00"))
            previous_close = float(raw_close)
            probability = float(decision.probability_next_bar_up)
        except (KeyError, TypeError, ValueError):
            invalid += 1
            continue
        if cutoff.tzinfo is None or not isfinite(previous_close) or previous_close <= 0:
            invalid += 1
            continue
        if not isfinite(probability) or not 0 <= probability <= 1:
            invalid += 1
            continue
        expected_timestamp = cutoff.astimezone(timezone.utc) + timedelta(minutes=1)
        next_bar = by_timestamp.get(expected_timestamp)
        if next_bar is None:
            pending += 1
            continue
        return_bps = (next_bar.close / previous_close - 1.0) * 10_000
        if not isfinite(return_bps):
            invalid += 1
            continue
        scored.append((decision, next_bar, return_bps))
    result = _summary(scored)
    result.update({"pending": pending, "invalid": invalid, "available_bars": len(ordered_bars)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score journaled live predictions without submitting orders")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--feed", choices=("iex", "sip"), default="iex")
    parser.add_argument("--decision-log", type=Path, required=True)
    parser.add_argument("--bars", type=int, default=1_000)
    args = parser.parse_args()
    decisions = PaperDecisionLog(args.decision_log).read()
    bars = AlpacaMarketDataClient(feed=args.feed).bars(args.symbol, limit=args.bars)
    print(json.dumps(score_predictions(decisions, bars), sort_keys=True))


if __name__ == "__main__":
    main()
