#!/usr/bin/env python3
"""Issue a short-lived independent approval for one paper order."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live.risk_approval import issue_risk_approval, write_risk_approval

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approve", action="store_true", required=True)
    parser.add_argument("--client-order-id", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--side", choices=("buy", "sell"), required=True)
    parser.add_argument("--quantity", type=int, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--quote-timestamp", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--ttl-seconds", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    approval = issue_risk_approval(client_order_id=args.client_order_id, symbol=args.symbol,
        side=args.side, quantity=args.quantity, model_sha256=args.model_sha256,
        quote_timestamp=args.quote_timestamp, reason=args.reason, ttl_seconds=args.ttl_seconds)
    write_risk_approval(approval, args.output)
    print({"approval_id": approval.approval_id, "expires_at": approval.expires_at, "output": str(args.output)})

if __name__ == "__main__":
    main()
