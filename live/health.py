#!/usr/bin/env python3
"""Read-only operator health snapshot for a paper-trading session.

This combines the existing preflight checks with model-quality evidence and
journal verification.  It has no order-submission dependency and always
returns a structured report suitable for a scheduler or alerting adapter.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.audit import PaperDecisionLog, file_sha256
from live.market_data import AlpacaMarketDataClient
from live.paper_broker import AlpacaPaperBroker
from live.preflight import run_preflight
from live.predictor import LogisticDirectionModel
from live.run_paper import validate_quality_report


ROOT = Path(__file__).resolve().parents[1]


def _check(name: str, ok: bool, detail: str) -> dict[str, object]:
    return {"check": name, "ok": ok, "detail": detail}


def run_health(
    symbol: str,
    model_path: Path,
    quality_report_path: Path,
    decision_log_path: Path,
    *,
    history_bars: int = 100,
    data_client: AlpacaMarketDataClient | None = None,
    broker: AlpacaPaperBroker | None = None,
    max_training_gap_seconds: float | None = None,
    max_bar_gap_seconds: float | None = None,
    max_quality_age_seconds: float = 86_400.0,
    feed: str = "iex",
) -> dict[str, object]:
    """Return one fail-closed health decision without submitting an order."""
    checks: list[dict[str, object]] = []
    try:
        model = LogisticDirectionModel.from_json(model_path)
        model_hash = file_sha256(model_path)
        quality = validate_quality_report(
            quality_report_path,
            symbol=symbol,
            model_sha256=model_hash,
            max_age_seconds=max_quality_age_seconds,
        )
        checks.append(_check("quality_evidence", True,
                             f"passed report pinned to model {model_hash[:12]}"))
    except Exception as exc:
        model = None
        quality = None
        checks.append(_check("quality_evidence", False,
                             f"quality evidence unavailable or invalid: {type(exc).__name__}"))

    preflight = run_preflight(
        symbol,
        model_path,
        history_bars=history_bars,
        data_client=data_client,
        broker=broker,
        max_training_gap_seconds=max_training_gap_seconds,
        max_bar_gap_seconds=max_bar_gap_seconds,
        feed=feed,
    )
    checks.extend(preflight["checks"])

    try:
        events = PaperDecisionLog(decision_log_path).read()
        if not events:
            raise ValueError("journal is empty")
        checks.append(_check("decision_journal", True,
                             f"verified {len(events)} hash-chained events"))
    except (OSError, ValueError) as exc:
        checks.append(_check("decision_journal", False,
                             f"journal unavailable or invalid: {type(exc).__name__}"))

    ready = all(bool(check.get("ok")) for check in checks)
    return {
        "schema": "paper_health.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol.upper(),
        "model": str(model_path),
        "quality_report": str(quality_report_path),
        "decision_log": str(decision_log_path),
        "paper_only": True,
        "order_submission_attempted": False,
        "ready_for_paper_session": ready,
        "checks": checks,
        "quality_gate": quality.get("quality_gate") if isinstance(quality, dict) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only paper-session health snapshot")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--feed", choices=("iex", "sip"), default="iex")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--decision-log", type=Path, required=True)
    parser.add_argument("--history-bars", type=int, default=100)
    parser.add_argument("--max-training-gap-hours", type=float)
    parser.add_argument("--max-bar-gap-minutes", type=float)
    parser.add_argument("--max-quality-age-hours", type=float, default=24.0)
    args = parser.parse_args()
    model_path = args.model or ROOT / "models" / f"{args.symbol.upper()}_logistic.json"
    report = run_health(
        args.symbol,
        model_path,
        args.quality_report,
        args.decision_log,
        history_bars=args.history_bars,
        max_training_gap_seconds=(None if args.max_training_gap_hours is None
                                  else args.max_training_gap_hours * 3_600),
        max_bar_gap_seconds=(None if args.max_bar_gap_minutes is None
                             else args.max_bar_gap_minutes * 60),
        max_quality_age_seconds=args.max_quality_age_hours * 3_600,
        feed=args.feed,
    )
    print(json.dumps(report, sort_keys=True))
    if not report["ready_for_paper_session"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
