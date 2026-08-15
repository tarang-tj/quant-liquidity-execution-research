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
import os
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


def score_predictions(decisions: Iterable[PaperDecision], bars: Iterable[Bar],
                      symbol: str | None = None) -> dict[str, object]:
    """Return causal quality metrics and pending counts for journaled decisions."""
    ordered_bars = sorted(bars, key=lambda bar: bar.timestamp)
    by_key = {(bar.symbol.upper(), bar.timestamp): bar for bar in ordered_bars}
    target_symbol = symbol.upper() if symbol is not None else None
    scored: list[tuple[PaperDecision, Bar, float]] = []
    pending = 0
    invalid = 0
    symbol_mismatch = 0
    for decision in decisions:
        if decision.event_kind != "decision":
            continue
        decision_symbol = decision.symbol.upper()
        if target_symbol is not None and decision_symbol != target_symbol:
            symbol_mismatch += 1
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
        next_bar = by_key.get((decision_symbol, expected_timestamp))
        if next_bar is None:
            pending += 1
            continue
        return_bps = (next_bar.close / previous_close - 1.0) * 10_000
        if not isfinite(return_bps):
            invalid += 1
            continue
        scored.append((decision, next_bar, return_bps))
    result = _summary(scored)
    result.update({"pending": pending, "invalid": invalid, "symbol_mismatch": symbol_mismatch,
                   "available_bars": len(ordered_bars)})
    return result


def apply_quality_gate(metrics: dict[str, object], *, minimum_scored: int = 20,
                       minimum_accuracy: float = 0.52, maximum_brier: float = 0.25) -> dict[str, object]:
    """Attach an explicit evidence gate used by scheduled monitoring."""
    if (type(minimum_scored) is not int or minimum_scored < 1 or
            type(minimum_accuracy) not in (int, float) or
            not isfinite(float(minimum_accuracy)) or not 0 <= minimum_accuracy <= 1 or
            type(maximum_brier) not in (int, float) or
            not isfinite(float(maximum_brier)) or not 0 <= maximum_brier <= 1):
        raise ValueError("invalid live-quality thresholds")
    scored = metrics.get("scored")
    accuracy = metrics.get("accuracy")
    brier = metrics.get("brier")
    checks = {
        "minimum_scored": isinstance(scored, int) and scored >= minimum_scored,
        "minimum_accuracy": isinstance(accuracy, (int, float)) and isfinite(float(accuracy))
        and float(accuracy) >= minimum_accuracy,
        "maximum_brier": isinstance(brier, (int, float)) and isfinite(float(brier))
        and float(brier) <= maximum_brier,
    }
    result = dict(metrics)
    result["quality_gate"] = {
        "passed": all(checks.values()),
        "checks": checks,
        "minimum_scored": minimum_scored,
        "minimum_accuracy": minimum_accuracy,
        "maximum_brier": maximum_brier,
    }
    return result


def build_quality_report(decisions: Iterable[PaperDecision], metrics: dict[str, object], *,
                         symbol: str, minimum_scored: int = 20,
                         minimum_accuracy: float = 0.52, maximum_brier: float = 0.25,
                         generated_at: datetime | None = None) -> dict[str, object]:
    """Build a durable, model-pinned quality report for the paper gate."""
    target_symbol = symbol.upper()
    matching = [decision for decision in decisions
                if decision.event_kind == "decision" and decision.symbol.upper() == target_symbol]
    model_hashes = sorted({decision.model_sha256 for decision in matching if decision.model_sha256})
    report = apply_quality_gate(
        metrics,
        minimum_scored=minimum_scored,
        minimum_accuracy=minimum_accuracy,
        maximum_brier=maximum_brier,
    )
    checks = dict(report["quality_gate"]["checks"])
    checks["single_model"] = len(model_hashes) == 1
    report["quality_gate"] = dict(report["quality_gate"], passed=all(checks.values()), checks=checks)
    report.update({
        "report_type": "live_quality_gate",
        "symbol": target_symbol,
        "model_sha256": model_hashes[0] if len(model_hashes) == 1 else None,
        "model_hashes": model_hashes,
        "generated_at": (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
    })
    return report


def write_quality_report(report: dict[str, object], path: Path) -> None:
    """Atomically write a quality report so readers never see partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score journaled live predictions without submitting orders")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--feed", choices=("iex", "sip"), default="iex")
    parser.add_argument("--decision-log", type=Path, required=True)
    parser.add_argument("--bars", type=int, default=1_000)
    parser.add_argument("--minimum-scored", type=int, default=20)
    parser.add_argument("--minimum-accuracy", type=float, default=0.52)
    parser.add_argument("--maximum-brier", type=float, default=0.25)
    parser.add_argument("--output", type=Path,
                        help="optional path for an atomic model-pinned quality report")
    args = parser.parse_args()
    decisions = PaperDecisionLog(args.decision_log).read()
    bars = AlpacaMarketDataClient(feed=args.feed).bars(args.symbol, limit=args.bars)
    report = build_quality_report(
        decisions,
        score_predictions(decisions, bars, symbol=args.symbol),
        symbol=args.symbol,
        minimum_scored=args.minimum_scored,
        minimum_accuracy=args.minimum_accuracy,
        maximum_brier=args.maximum_brier,
    )
    if args.output:
        write_quality_report(report, args.output)
    print(json.dumps(report, sort_keys=True))
    if not report["quality_gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
