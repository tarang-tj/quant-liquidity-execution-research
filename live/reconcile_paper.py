#!/usr/bin/env python3
"""Reconcile an uncertain Alpaca paper order by idempotency key; never retry it."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.audit import PaperDecision, PaperDecisionLog
from live.paper_broker import AlpacaPaperBroker


ROOT = Path(__file__).resolve().parents[1]


def _required_text(order: dict[str, object], field: str) -> str:
    value = order.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"broker reconciliation response is missing a non-empty {field}")
    return value


def record_reconciliation_result(
    decision_log: PaperDecisionLog, client_order_id: str, order: dict[str, object]
) -> PaperDecision:
    """Append a resolved broker lookup for a previously uncertain request.

    The broker's response must identify the same idempotency key.  This makes a
    mismatched or malformed lookup fail closed without changing the journal.
    """
    broker_client_order_id = _required_text(order, "client_order_id")
    paper_order_id = _required_text(order, "id")
    status = _required_text(order, "status")
    if broker_client_order_id != client_order_id:
        raise ValueError("broker reconciliation response client_order_id does not match request")

    return decision_log.transition_latest(
        client_order_id,
        "unknown_reconciliation_required",
        lambda prior: replace(
            prior,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_kind="reconciliation_result",
            submitted=True,
            paper_order_id=paper_order_id,
            submission_error=None,
            submission_state=f"reconciled:{status}",
            broker_order_status=status,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-order-id", required=True)
    parser.add_argument("--decision-log", type=Path, default=ROOT / "runtime" / "paper_decisions.jsonl")
    args = parser.parse_args()
    order = AlpacaPaperBroker().order_by_client_order_id(args.client_order_id)
    record = record_reconciliation_result(PaperDecisionLog(args.decision_log), args.client_order_id, order)
    print({"client_order_id": record.client_order_id, "paper_order_id": record.paper_order_id,
           "status": record.broker_order_status, "journal_event": record.event_kind})


if __name__ == "__main__":
    main()
