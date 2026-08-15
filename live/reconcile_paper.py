#!/usr/bin/env python3
"""Look up an uncertain Alpaca paper order by its idempotency key; never retries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.paper_broker import AlpacaPaperBroker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-order-id", required=True)
    args = parser.parse_args()
    order = AlpacaPaperBroker().order_by_client_order_id(args.client_order_id)
    print({"client_order_id": args.client_order_id, "paper_order_id": order.get("id"), "status": order.get("status")})


if __name__ == "__main__":
    main()
