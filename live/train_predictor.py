#!/usr/bin/env python3
"""Train and chronologically validate the paper-only direction baseline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.market_data import AlpacaMarketDataClient, Bar
from live.predictor import train_direction_model


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path, symbol: str) -> list[Bar]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [Bar(symbol.upper(), datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).astimezone(timezone.utc),
                float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row["volume"])) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--input", type=Path, help="CSV with timestamp,open,high,low,close,volume")
    parser.add_argument("--bars", type=int, default=1_000)
    parser.add_argument("--timeframe", default="1Min", help="timeframe represented by the training CSV/bars")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bars = (read_csv(args.input, args.symbol) if args.input
            else AlpacaMarketDataClient().bars(args.symbol, limit=args.bars, timeframe=args.timeframe))
    model = train_direction_model(bars, timeframe=args.timeframe)
    output = args.output or ROOT / "models" / f"{args.symbol.upper()}_logistic.json"
    model.to_json(output)
    print({"model": str(output), "report": model.report})


if __name__ == "__main__":
    main()
