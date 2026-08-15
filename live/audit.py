"""Append-only local integrity evidence for paper-trading decisions.

This module deliberately stores no credentials, account numbers, or HTTP
headers.  A decision captures the completed bars and quote that were actually
used, so it can be deterministically replayed with the recorded model hash.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from live.market_data import Bar, Quote


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bar_record(bar: Bar) -> dict[str, object]:
    return {"symbol": bar.symbol, "timestamp": bar.timestamp.isoformat(), "open": bar.open,
            "high": bar.high, "low": bar.low, "close": bar.close, "volume": bar.volume}


def quote_record(quote: Quote) -> dict[str, object]:
    record = asdict(quote)
    record["timestamp"] = quote.timestamp.isoformat()
    return record


@dataclass(frozen=True, slots=True)
class PaperDecision:
    timestamp: str
    model_path: str
    model_sha256: str
    symbol: str
    quote: dict[str, object]
    completed_bars: list[dict[str, object]]
    features: list[float]
    probability_next_bar_up: float
    proposed_side: str
    risk_approved: bool
    risk_reason: str
    risk_source: str
    broker_position: float | None
    broker_daily_pnl: float | None
    submitted: bool
    paper_order_id: str | None
    completed_before: str
    event_kind: str = "decision"
    submission_error: str | None = None
    client_order_id: str | None = None
    submission_state: str = "not_requested"
    previous_hash: str | None = None
    record_hash: str | None = None

    @classmethod
    def create(cls, *, model_path: Path, symbol: str, quote: Quote, bars: list[Bar], features: list[float],
               probability: float, side: str, risk_approved: bool, risk_reason: str, risk_source: str,
               broker_position: float | None, broker_daily_pnl: float | None, submitted: bool = False,
               paper_order_id: str | None = None, client_order_id: str | None = None) -> "PaperDecision":
        cutoff = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        if any(bar.timestamp >= cutoff for bar in bars):
            raise ValueError("decision evidence contains an incomplete or future minute bar")
        return cls(datetime.now(timezone.utc).isoformat(), str(model_path.resolve()), file_sha256(model_path),
                   symbol, quote_record(quote), [bar_record(bar) for bar in bars], features, probability, side,
                   risk_approved, risk_reason, risk_source, broker_position, broker_daily_pnl, submitted,
                   paper_order_id, cutoff.isoformat(), client_order_id=client_order_id)


class PaperDecisionLog:
    """Append-only, single-writer JSONL journal with an unauthenticated hash chain.

    The chain catches accidental or casual local edits.  It is not an external
    trust anchor: a writer who controls the whole file can recompute it.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    @staticmethod
    def _hash_record(decision: PaperDecision) -> str:
        payload = asdict(decision)
        payload["record_hash"] = None
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def append(self, decision: PaperDecision) -> PaperDecision:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        import fcntl  # POSIX target: this research system runs on macOS/Linux.
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                existing = self._parse_lines(handle.read().splitlines())
                previous_hash = existing[-1].record_hash if existing else None
                committed = replace(decision, previous_hash=previous_hash, record_hash=None)
                committed = replace(committed, record_hash=self._hash_record(committed))
                handle.seek(0, os.SEEK_END)
                handle.write(json.dumps(asdict(committed), sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return committed

    @classmethod
    def _parse_lines(cls, lines: list[str]) -> list[PaperDecision]:
        decisions: list[PaperDecision] = []
        expected_previous_hash: str | None = None
        for line_number, line in enumerate(lines, start=1):
            try:
                raw: Any = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("record is not an object")
                decision = PaperDecision(**raw)
                if decision.previous_hash != expected_previous_hash or not decision.record_hash:
                    raise ValueError("broken decision-log hash chain")
                if decision.record_hash != cls._hash_record(decision):
                    raise ValueError("decision-log record hash does not match")
                expected_previous_hash = decision.record_hash
                decisions.append(decision)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid paper decision at line {line_number}") from exc
        return decisions

    def read(self) -> list[PaperDecision]:
        if not self.path.exists():
            raise FileNotFoundError(f"paper decision log does not exist: {self.path}")
        import fcntl
        with self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                return self._parse_lines(handle.read().splitlines())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
