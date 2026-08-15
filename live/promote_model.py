#!/usr/bin/env python3
"""Gate and atomically promote a predictive model for paper evaluation."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.audit import file_sha256
from live.market_data import AlpacaMarketDataClient, Bar
from live.predictor import LogisticDirectionModel, train_direction_model, validate_paper_model
from live.train_predictor import read_csv
from live.walk_forward import walk_forward_evaluate


def promote_if_qualified(
    bars: list[Bar],
    *,
    symbol: str,
    target_path: Path,
    timeframe: str = "1Min",
    training_bars: int = 120,
    evaluation_bars: int = 120,
    transaction_cost_bps: float = 5.0,
    minimum_accuracy: float = 0.52,
    maximum_brier: float = 0.25,
    minimum_net_return_bps: float = 0.0,
) -> dict[str, Any]:
    """Train a candidate, gate it, and replace ``target_path`` only on success.

    The walk-forward evaluation is completed before the candidate is fitted for
    the final artifact. A rejected candidate is never written, preserving the
    last known-good model. The returned report contains no credentials.
    """
    for name, value in (("minimum_accuracy", minimum_accuracy), ("maximum_brier", maximum_brier),
                        ("minimum_net_return_bps", minimum_net_return_bps)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
            raise ValueError(f"{name} must be numeric")
    if not 0 <= minimum_accuracy <= 1:
        raise ValueError("minimum_accuracy must be between 0 and 1")
    if not 0 <= maximum_brier <= 1:
        raise ValueError("maximum_brier must be between 0 and 1")
    walk = walk_forward_evaluate(
        bars, training_bars=training_bars, evaluation_bars=evaluation_bars,
        timeframe=timeframe, transaction_cost_bps=transaction_cost_bps,
    )
    candidate = train_direction_model(bars, timeframe=timeframe)
    checks = {
        "model_validation_gate": bool(candidate.report.deployable_for_paper),
        "walk_forward_accuracy": walk.accuracy >= minimum_accuracy,
        "walk_forward_brier": walk.brier <= maximum_brier,
        "walk_forward_net_return": walk.net_return_bps >= minimum_net_return_bps,
    }
    report: dict[str, Any] = {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "target_path": str(target_path),
        "promoted": False,
        "checks": checks,
        "walk_forward": walk.as_dict(),
        "candidate_model_sha256": None,
        "candidate_data_sha256": candidate.report.training_data_sha256,
    }
    if not all(checks.values()):
        report["rejection_reason"] = "candidate did not satisfy every promotion gate"
        return report
    validate_paper_model(candidate, symbol, timeframe)
    candidate.to_json(target_path)
    report["candidate_model_sha256"] = file_sha256(target_path)
    report["promoted"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate and promote a paper-only predictive model")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--input", type=Path, help="CSV with timestamp,open,high,low,close,volume")
    parser.add_argument("--bars", type=int, default=1_000)
    parser.add_argument("--training-bars", type=int, default=120)
    parser.add_argument("--evaluation-bars", type=int, default=120)
    parser.add_argument("--timeframe", default="1Min")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--minimum-accuracy", type=float, default=0.52)
    parser.add_argument("--maximum-brier", type=float, default=0.25)
    parser.add_argument("--minimum-net-return-bps", type=float, default=0.0)
    parser.add_argument("--target", type=Path)
    args = parser.parse_args()
    bars: list[Bar] = (read_csv(args.input, args.symbol) if args.input
                       else AlpacaMarketDataClient().bars(args.symbol, limit=args.bars, timeframe=args.timeframe))
    target = args.target or Path(__file__).resolve().parents[1] / "models" / f"{args.symbol.upper()}_logistic.json"
    report = promote_if_qualified(
        bars, symbol=args.symbol, target_path=target, timeframe=args.timeframe,
        training_bars=args.training_bars, evaluation_bars=args.evaluation_bars,
        transaction_cost_bps=args.transaction_cost_bps, minimum_accuracy=args.minimum_accuracy,
        maximum_brier=args.maximum_brier, minimum_net_return_bps=args.minimum_net_return_bps,
    )
    print(json.dumps(report, sort_keys=True))
    if not report["promoted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
