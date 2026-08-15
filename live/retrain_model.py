#!/usr/bin/env python3
"""Run a bounded, auditable paper-model retraining/promotion loop.

This is deliberately a finite operator job rather than an unbounded daemon.
Each cycle fetches a fresh historical window, evaluates a candidate causally,
and atomically promotes it only when every existing promotion gate passes.
There is no order-submission dependency in this module.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from math import isfinite
import os
from pathlib import Path
import sys
import time
from typing import Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.market_data import AlpacaMarketDataClient
from live.promote_model import promote_if_qualified


ROOT = Path(__file__).resolve().parents[1]


def _append_report(path: Path, report: dict[str, object]) -> None:
    """Append one complete report and fsync it before the cycle is returned."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - POSIX is the supported target.
            fcntl = None
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_retraining(
    symbol: str,
    target_path: Path,
    *,
    iterations: int,
    interval_seconds: float,
    bars: int = 1_000,
    training_bars: int = 120,
    evaluation_bars: int = 120,
    transaction_cost_bps: float = 5.0,
    minimum_accuracy: float = 0.52,
    maximum_brier: float = 0.25,
    minimum_net_return_bps: float = 0.0,
    data_client: AlpacaMarketDataClient | None = None,
    report_path: Path | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    feed: str = "iex",
    adjustment: str = "all",
) -> list[dict[str, object]]:
    """Run a finite number of fresh-data promotion cycles.

    Rejected candidates leave the active model untouched.  Data, validation,
    and promotion failures are raised so an operator/scheduler can fail closed;
    successful and rejected promotion reports are persisted when ``report_path``
    is supplied.
    """
    if not isinstance(iterations, int) or isinstance(iterations, bool) or not 1 <= iterations <= 10_000:
        raise ValueError("iterations must be an integer between 1 and 10,000")
    if not isfinite(interval_seconds) or interval_seconds < 0 or interval_seconds > 86_400:
        raise ValueError("interval_seconds must be between 0 and 86,400")
    if not isinstance(bars, int) or isinstance(bars, bool) or not 2 <= bars <= 10_000:
        raise ValueError("bars must be an integer between 2 and 10,000")
    if feed not in {"iex", "sip"}:
        raise ValueError("feed must be either 'iex' or 'sip'")
    if adjustment not in {"raw", "split", "dividend", "all"}:
        raise ValueError("adjustment must be one of 'raw', 'split', 'dividend', or 'all'")
    client = data_client or AlpacaMarketDataClient(feed=feed, adjustment=adjustment)
    reports: list[dict[str, object]] = []
    for cycle in range(iterations):
        ordered_bars = client.bars(symbol, limit=bars, timeframe="1Min")
        report = promote_if_qualified(
            ordered_bars,
            symbol=symbol,
            target_path=target_path,
            training_bars=training_bars,
            evaluation_bars=evaluation_bars,
            transaction_cost_bps=transaction_cost_bps,
            minimum_accuracy=minimum_accuracy,
            maximum_brier=maximum_brier,
            minimum_net_return_bps=minimum_net_return_bps,
            feed=feed,
            adjustment=adjustment,
        )
        report = {
            **report,
            "schema": "paper_model_promotion.v1",
            "cycle": cycle,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "feed": feed,
            "adjustment": adjustment,
            "bar_count": len(ordered_bars),
            "order_submission_attempted": False,
            "paper_only": True,
        }
        if report_path is not None:
            _append_report(report_path, report)
        reports.append(report)
        if cycle + 1 < iterations and interval_seconds:
            sleep_fn(interval_seconds)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded, auditable paper-model retraining loop")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--feed", choices=("iex", "sip"), default="iex",
                        help="Alpaca market-data feed; SIP requires the appropriate entitlement")
    parser.add_argument("--adjustment", choices=("raw", "split", "dividend", "all"), default="all",
                        help="corporate-action adjustment applied to historical bars")
    parser.add_argument("--iterations", type=int, default=1,
                        help="finite number of promotion cycles; never runs unbounded")
    parser.add_argument("--interval-seconds", type=float, default=3_600.0)
    parser.add_argument("--bars", type=int, default=1_000)
    parser.add_argument("--training-bars", type=int, default=120)
    parser.add_argument("--evaluation-bars", type=int, default=120)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--minimum-accuracy", type=float, default=0.52)
    parser.add_argument("--maximum-brier", type=float, default=0.25)
    parser.add_argument("--minimum-net-return-bps", type=float, default=0.0)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--report", type=Path,
                        help="optional JSONL path for durable cycle reports")
    args = parser.parse_args()
    target = args.target or ROOT / "models" / f"{args.symbol.upper()}_logistic.json"
    reports = run_retraining(
        args.symbol,
        target,
        iterations=args.iterations,
        interval_seconds=args.interval_seconds,
        bars=args.bars,
        training_bars=args.training_bars,
        evaluation_bars=args.evaluation_bars,
        transaction_cost_bps=args.transaction_cost_bps,
        minimum_accuracy=args.minimum_accuracy,
        maximum_brier=args.maximum_brier,
        minimum_net_return_bps=args.minimum_net_return_bps,
        report_path=args.report,
        feed=args.feed,
        adjustment=args.adjustment,
    )
    for report in reports:
        print(json.dumps(report, sort_keys=True))
    if not all(bool(report.get("promoted")) for report in reports):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
