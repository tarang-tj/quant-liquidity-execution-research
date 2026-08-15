#!/usr/bin/env python3
"""Run a causal walk-forward evaluation without submitting orders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.market_data import AlpacaMarketDataClient, Bar
from live.train_predictor import read_csv
from live.walk_forward import walk_forward_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal walk-forward paper-model evaluation")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--feed", choices=("iex", "sip"), default="iex",
                        help="Alpaca market-data feed; SIP requires the appropriate entitlement")
    parser.add_argument("--adjustment", choices=("raw", "split", "dividend", "all"), default="all",
                        help="corporate-action adjustment applied to historical bars")
    parser.add_argument("--input", type=Path, help="CSV with timestamp,open,high,low,close,volume")
    parser.add_argument("--bars", type=int, default=1_000)
    parser.add_argument("--training-bars", type=int, default=120)
    parser.add_argument("--evaluation-bars", type=int, default=120)
    parser.add_argument("--timeframe", default="1Min")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0,
                        help="cost charged per unit of position turnover")
    args = parser.parse_args()
    bars: list[Bar] = (read_csv(args.input, args.symbol) if args.input
                       else AlpacaMarketDataClient(feed=args.feed, adjustment=args.adjustment).bars(
                           args.symbol, limit=args.bars, timeframe=args.timeframe))
    summary = walk_forward_evaluate(
        bars, training_bars=args.training_bars, evaluation_bars=args.evaluation_bars,
        timeframe=args.timeframe, transaction_cost_bps=args.transaction_cost_bps,
        feed=args.feed, adjustment=args.adjustment,
    )
    print(json.dumps(summary.as_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
