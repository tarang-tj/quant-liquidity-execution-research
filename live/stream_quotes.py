#!/usr/bin/env python3
"""Collect a bounded Alpaca quote stream into the local normalized event store.

This is a read-only ingestion building block.  It deliberately has a finite
quote and wall-clock budget so a failed provider cannot become an unobserved
infinite process.  It never evaluates an order and has no broker dependency.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.market_data import AlpacaMarketDataClient, JsonlEventStore, MarketDataError, Quote


async def collect_quotes(
    client: AlpacaMarketDataClient,
    symbols: list[str],
    store: JsonlEventStore,
    *,
    max_quotes: int = 100,
    timeout_seconds: float = 60.0,
    max_reconnects: int = 3,
) -> dict[str, object]:
    """Collect at most ``max_quotes`` before the wall-clock timeout expires.

    The stream is dependency-injected through ``client`` so offline tests can
    exercise the lifecycle without credentials or a network connection.
    """
    if (not isinstance(symbols, list) or not symbols or
            any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols)):
        raise ValueError("symbols must be a non-empty list of non-empty strings")
    if (not isinstance(max_quotes, int) or isinstance(max_quotes, bool) or
            not 1 <= max_quotes <= 100_000):
        raise ValueError("max_quotes must be an integer between 1 and 100,000")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or
            timeout_seconds <= 0 or timeout_seconds > 86_400):
        raise ValueError("timeout_seconds must be between 0 and 86,400")
    if (not isinstance(max_reconnects, int) or isinstance(max_reconnects, bool) or
            max_reconnects < 0 or max_reconnects > 100):
        raise ValueError("max_reconnects must be an integer between 0 and 100")

    normalized_symbols = sorted({symbol.upper() for symbol in symbols})
    stream = client.stream_quotes(normalized_symbols, max_reconnects=max_reconnects)
    iterator = stream.__aiter__()
    collected: list[Quote] = []
    latest_by_symbol: dict[str, datetime] = {}
    duplicate_timestamps = 0
    largest_gap_seconds = 0.0
    deadline = asyncio.get_running_loop().time() + float(timeout_seconds)
    timed_out = False
    try:
        while len(collected) < max_quotes:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                timed_out = True
                break
            try:
                quote = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
            except asyncio.TimeoutError:
                timed_out = True
                break
            except StopAsyncIteration:
                break
            if not isinstance(quote, Quote):
                raise TypeError("quote stream yielded a non-Quote value")
            if quote.symbol.upper() not in normalized_symbols:
                raise MarketDataError("quote stream yielded a symbol that was not requested")
            previous = latest_by_symbol.get(quote.symbol.upper())
            if previous is not None:
                gap_seconds = (quote.timestamp - previous).total_seconds()
                if gap_seconds < 0:
                    raise MarketDataError("quote stream timestamp moved backwards")
                if gap_seconds == 0:
                    duplicate_timestamps += 1
                largest_gap_seconds = max(largest_gap_seconds, gap_seconds)
            latest_by_symbol[quote.symbol.upper()] = quote.timestamp
            collected.append(quote)
            # Append one event at a time so a process interruption loses at
            # most the in-flight quote and never an acknowledged batch.
            store.append_quotes([quote])
    finally:
        closer = getattr(stream, "aclose", None)
        if closer is not None:
            await closer()

    timestamps = [quote.timestamp for quote in collected]
    return {
        "schema": "quote_stream.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": normalized_symbols,
        "quotes_written": len(collected),
        "first_quote_at": min(timestamps).isoformat() if timestamps else None,
        "last_quote_at": max(timestamps).isoformat() if timestamps else None,
        "symbols_seen": sorted(latest_by_symbol),
        "duplicate_timestamps": duplicate_timestamps,
        "max_interquote_gap_seconds": largest_gap_seconds,
        "stream_integrity": "passed",
        "timed_out": timed_out,
        "stream_exhausted": not timed_out and len(collected) < max_quotes,
        "paper_only": True,
        "order_submission_attempted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded, read-only Alpaca WebSocket quote collector")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="append-only normalized quote JSONL path")
    parser.add_argument("--feed", choices=("iex", "sip"), default="iex")
    parser.add_argument("--max-quotes", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-reconnects", type=int, default=3)
    args = parser.parse_args()
    client = AlpacaMarketDataClient(feed=args.feed)
    report = asyncio.run(collect_quotes(
        client,
        args.symbols,
        JsonlEventStore(args.output),
        max_quotes=args.max_quotes,
        timeout_seconds=args.timeout_seconds,
        max_reconnects=args.max_reconnects,
    ))
    print(json.dumps(report, sort_keys=True))
    if report["quotes_written"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
