"""Provider-neutral market-data types and an Alpaca REST/WebSocket adapter."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import AsyncIterator, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class MarketDataError(RuntimeError):
    """A recoverable market-data provider or schema error."""


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    timestamp: datetime
    source: str

    def __post_init__(self) -> None:
        if (not self.symbol or not all(isfinite(value) for value in (self.bid_price, self.ask_price,
                self.bid_size, self.ask_size)) or self.bid_price <= 0 or self.ask_price <= 0
                or self.ask_price < self.bid_price):
            raise ValueError("quote must have positive, ordered bid/ask prices")
        if self.bid_size < 0 or self.ask_size < 0:
            raise ValueError("quote sizes cannot be negative")
        if self.timestamp.tzinfo is None:
            raise ValueError("quote timestamp must be timezone-aware")

    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread_bps(self) -> float:
        return 10_000.0 * (self.ask_price - self.bid_price) / self.mid_price

    @property
    def imbalance(self) -> float:
        total = self.bid_size + self.ask_size
        return 0.0 if total <= 0 else (self.bid_size - self.ask_size) / total

    @classmethod
    def from_alpaca(cls, payload: dict[str, object], source: str = "alpaca") -> "Quote":
        quote = payload.get("quote", payload)
        if not isinstance(quote, dict):
            raise MarketDataError("latest quote response does not contain an object")
        try:
            timestamp = datetime.fromisoformat(str(quote["t"]).replace("Z", "+00:00"))
            return cls(str(quote["S"]), float(quote["bp"]), float(quote["ap"]), float(quote.get("bs", 0)),
                       float(quote.get("as", 0)), timestamp.astimezone(timezone.utc), source)
        except (KeyError, TypeError, ValueError) as exc:
            raise MarketDataError(f"invalid Alpaca quote schema: {quote!r}") from exc


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        values = (self.open, self.high, self.low, self.close, self.volume)
        if not self.symbol or not all(isfinite(value) for value in values):
            raise ValueError("bar values must be finite")
        if min(self.open, self.high, self.low, self.close) <= 0 or self.volume < 0:
            raise ValueError("bar prices must be positive and volume non-negative")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("bar high/low must bound open and close")
        if self.timestamp.tzinfo is None:
            raise ValueError("bar timestamp must be timezone-aware")

    @classmethod
    def from_alpaca(cls, symbol: str, payload: dict[str, object]) -> "Bar":
        try:
            timestamp = datetime.fromisoformat(str(payload["t"]).replace("Z", "+00:00"))
            return cls(symbol, timestamp.astimezone(timezone.utc), float(payload["o"]), float(payload["h"]),
                       float(payload["l"]), float(payload["c"]), float(payload["v"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise MarketDataError(f"invalid Alpaca bar schema: {payload!r}") from exc


class JsonlEventStore:
    """Durable append-only normalized event store.

    A single exclusive lock covers the whole batch so separate quote-stream
    workers cannot interleave JSON lines.  The file is flushed and fsynced
    before the lock is released; this is local durability, not a replicated
    or tamper-evident event archive.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def append_quotes(self, quotes: Iterable[Quote]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        records: list[str] = []
        for quote in quotes:
            record = asdict(quote)
            record["timestamp"] = quote.timestamp.isoformat()
            records.append(json.dumps(record, sort_keys=True) + "\n")
        if not records:
            return
        import fcntl  # POSIX target: this research system runs on macOS/Linux.
        with self.path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.writelines(records)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class AlpacaMarketDataClient:
    """Minimal authenticated Alpaca adapter with no SDK dependency.

    Set ALPACA_DATA_KEY and ALPACA_DATA_SECRET.  The default IEX feed is useful
    for paper research but is not consolidated-market coverage.
    """

    base_url = "https://data.alpaca.markets"

    def __init__(self, key: str | None = None, secret: str | None = None, feed: str = "iex",
                 adjustment: str = "all", max_retries: int = 2,
                 retry_backoff_seconds: float = 0.5) -> None:
        self.key = key or os.environ.get("ALPACA_DATA_KEY")
        self.secret = secret or os.environ.get("ALPACA_DATA_SECRET")
        self.feed = feed
        self.adjustment = adjustment
        if not self.key or not self.secret:
            raise MarketDataError("set ALPACA_DATA_KEY and ALPACA_DATA_SECRET; never put credentials in source files")
        if feed not in {"iex", "sip"}:
            raise ValueError("feed must be either 'iex' or 'sip'")
        if adjustment not in {"raw", "split", "dividend", "all"}:
            raise ValueError("adjustment must be one of 'raw', 'split', 'dividend', or 'all'")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be an integer between 0 and 5")
        if (isinstance(retry_backoff_seconds, bool) or not isinstance(retry_backoff_seconds, (int, float))
                or not isfinite(retry_backoff_seconds) or retry_backoff_seconds < 0
                or retry_backoff_seconds > 30):
            raise ValueError("retry_backoff_seconds must be between 0 and 30 seconds")
        self.max_retries = max_retries
        self.retry_backoff_seconds = float(retry_backoff_seconds)

    def _get_json(self, path: str, query: dict[str, str] | None = None) -> dict[str, object]:
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        request = Request(url, headers={"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret, "Accept": "application/json"})
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=15) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                transient = exc.code == 429 or 500 <= exc.code <= 599
                if not transient or attempt >= self.max_retries:
                    raise MarketDataError(
                        f"Alpaca market-data HTTP {exc.code}; check entitlement, symbol, and feed") from exc
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt >= self.max_retries:
                    raise MarketDataError("Alpaca market-data request failed") from exc
            time.sleep(self.retry_backoff_seconds * (2 ** attempt))
        if not isinstance(parsed, dict):
            raise MarketDataError("Alpaca market-data response must be an object")
        return parsed

    def latest_quote(self, symbol: str) -> Quote:
        payload = self._get_json(f"/v2/stocks/{symbol.upper()}/quotes/latest", {"feed": self.feed})
        return Quote.from_alpaca(payload)

    def bars(self, symbol: str, limit: int = 1000, timeframe: str = "1Min",
             completed_before: datetime | None = None) -> list[Bar]:
        """Return only completed one-minute bars, excluding the current minute.

        Alpaca timestamps a minute bar at its opening minute.  A bar at or after
        the current minute may still be changing, so using it would leak
        intra-minute close/high/low/volume into a decision.  This bridge
        intentionally supports only 1Min live features until other intervals
        have equally explicit completion rules.
        """
        if not 2 <= limit <= 10_000:
            raise ValueError("limit must be between 2 and 10,000")
        if timeframe != "1Min":
            raise ValueError("paper live features currently require timeframe='1Min'")
        cutoff = completed_before or datetime.now(timezone.utc)
        if cutoff.tzinfo is None:
            raise ValueError("completed_before must be timezone-aware")
        cutoff = cutoff.astimezone(timezone.utc).replace(second=0, microsecond=0)
        payload = self._get_json(f"/v2/stocks/{symbol.upper()}/bars", {
            "timeframe": timeframe, "limit": str(limit), "feed": self.feed,
            "adjustment": self.adjustment,
        })
        raw_bars = payload.get("bars")
        if not isinstance(raw_bars, list):
            raise MarketDataError("bar response does not contain a bars list")
        if not all(isinstance(raw, dict) for raw in raw_bars):
            raise MarketDataError("bar response contains a non-object bar")
        bars = [Bar.from_alpaca(symbol.upper(), raw) for raw in raw_bars]
        bars = [bar for bar in bars if bar.timestamp < cutoff]
        if len(bars) < 2:
            raise MarketDataError("provider returned fewer than two completed bars")
        return sorted(bars, key=lambda bar: bar.timestamp)

    async def stream_quotes(self, symbols: list[str], *, max_reconnects: int = 3,
                            reconnect_backoff_seconds: float = 1.0) -> AsyncIterator[Quote]:
        """Yield quotes with bounded reconnects; authentication failures fail closed.

        The reconnect budget is intentionally finite so a broken feed cannot
        turn into an unobserved infinite loop. Callers should supervise the
        exhausted stream and decide whether to halt trading or page an operator.
        """
        if (not isinstance(symbols, list) or not symbols or
                any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols)):
            raise ValueError("symbols must be a non-empty list of non-empty strings")
        if (not isinstance(max_reconnects, int) or isinstance(max_reconnects, bool) or max_reconnects < 0):
            raise ValueError("max_reconnects must be a non-negative integer")
        if (isinstance(reconnect_backoff_seconds, bool) or
                not isinstance(reconnect_backoff_seconds, (int, float)) or
                not isfinite(reconnect_backoff_seconds) or reconnect_backoff_seconds < 0 or
                reconnect_backoff_seconds > 300):
            raise ValueError("reconnect_backoff_seconds must be between 0 and 300 seconds")
        normalized_symbols = sorted({symbol.upper() for symbol in symbols})
        try:
            import websockets
        except ImportError as exc:
            raise MarketDataError("install requirements-live.txt to use WebSocket streaming") from exc
        endpoint = f"wss://stream.data.alpaca.markets/v2/{self.feed}"
        reconnects = 0
        while True:
            try:
                async with websockets.connect(endpoint, open_timeout=15, ping_interval=20) as socket:
                    await socket.send(json.dumps({"action": "auth", "key": self.key, "secret": self.secret}))
                    auth = json.loads(await asyncio.wait_for(socket.recv(), timeout=10))
                    auth_items = auth if isinstance(auth, list) else [auth]
                    if not any(isinstance(item, dict) and item.get("T") == "success" for item in auth_items):
                        raise MarketDataError("Alpaca WebSocket authentication failed")
                    await socket.send(json.dumps({"action": "subscribe", "quotes": normalized_symbols}))
                    async for raw_message in socket:
                        messages = json.loads(raw_message)
                        for item in messages if isinstance(messages, list) else [messages]:
                            if isinstance(item, dict) and item.get("T") == "q":
                                yield Quote.from_alpaca(item, source="alpaca-websocket")
                    raise ConnectionError("Alpaca quote stream closed")
            except MarketDataError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # provider-specific WebSocket errors have unstable types
                if reconnects >= max_reconnects:
                    raise MarketDataError("Alpaca quote stream exhausted reconnect budget") from exc
                delay = min(float(reconnect_backoff_seconds) * (2 ** reconnects), 300.0)
                reconnects += 1
                await asyncio.sleep(delay)
