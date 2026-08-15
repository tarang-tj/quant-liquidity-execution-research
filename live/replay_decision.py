#!/usr/bin/env python3
"""Verify a recorded paper decision with no network or credentials."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.audit import PaperDecisionLog, file_sha256
from live.market_data import Bar
from live.predictor import LogisticDirectionModel, live_features


def bar_from_record(record: dict[str, object]) -> Bar:
    return Bar(str(record["symbol"]), datetime.fromisoformat(str(record["timestamp"])), float(record["open"]),
               float(record["high"]), float(record["low"]), float(record["close"]), float(record["volume"]))


def replay(model_path: Path, decision_log: Path, index: int = -1) -> float:
    decision = PaperDecisionLog(decision_log).read()[index]
    if file_sha256(model_path) != decision.model_sha256:
        raise ValueError("model hash differs from the model used for this decision")
    bars = [bar_from_record(record) for record in decision.completed_bars]
    cutoff = datetime.fromisoformat(decision.completed_before)
    if cutoff.tzinfo is None or any(bar.timestamp >= cutoff for bar in bars):
        raise ValueError("decision evidence includes a bar not completed by its recorded cutoff")
    features = live_features(bars)
    if len(decision.features) != len(features) or any(abs(left - right) > 1e-12
                                                       for left, right in zip(decision.features, features)):
        raise ValueError("replayed features differ from the recorded decision")
    probability = LogisticDirectionModel.from_json(model_path).predict_probability(features)
    if abs(probability - decision.probability_next_bar_up) > 1e-12:
        raise ValueError("replayed probability differs from the recorded decision")
    return probability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--decision-log", type=Path, required=True)
    parser.add_argument("--index", type=int, default=-1)
    args = parser.parse_args()
    print({"replayed_probability_next_bar_up": replay(args.model, args.decision_log, args.index), "verified": True})


if __name__ == "__main__":
    main()
