"""Independent, short-lived HMAC approval artifacts for paper orders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets

SCHEMA = "risk_approval.v1"
KEY_ENV = "TRADING_RISK_APPROVAL_KEY"


def _canonical(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _key(key: bytes | None = None) -> bytes:
    value = key if key is not None else os.environ.get(KEY_ENV, "").encode()
    if len(value) < 32:
        raise ValueError(f"{KEY_ENV} must contain at least 32 bytes")
    return value


def _utc(value: datetime | str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class RiskApproval:
    schema: str
    approval_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: int
    model_sha256: str
    quote_timestamp: str
    issued_at: str
    expires_at: str
    reason: str
    approved: bool
    signature: str


def issue_risk_approval(*, client_order_id: str, symbol: str, side: str, quantity: int,
                        model_sha256: str, quote_timestamp: datetime | str, reason: str,
                        ttl_seconds: int = 60, key: bytes | None = None,
                        issued_at: datetime | None = None) -> RiskApproval:
    if not isinstance(client_order_id, str) or not client_order_id or len(client_order_id) > 48:
        raise ValueError("client_order_id must be non-empty and at most 48 characters")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol is required")
    if side not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise ValueError("quantity must be a positive integer")
    if not isinstance(model_sha256, str) or len(model_sha256) != 64 or any(c not in "0123456789abcdef" for c in model_sha256):
        raise ValueError("model_sha256 must be a lowercase SHA-256 hex digest")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason is required")
    if not isinstance(ttl_seconds, int) or not 1 <= ttl_seconds <= 300:
        raise ValueError("ttl_seconds must be between 1 and 300")
    issued = (issued_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    quote = _utc(quote_timestamp)
    payload = {
        "schema": SCHEMA, "approval_id": secrets.token_urlsafe(18),
        "client_order_id": client_order_id, "symbol": symbol.upper(), "side": side,
        "quantity": quantity, "model_sha256": model_sha256, "quote_timestamp": quote.isoformat(),
        "issued_at": issued.isoformat(), "expires_at": (issued + timedelta(seconds=ttl_seconds)).isoformat(),
        "reason": reason.strip(), "approved": True,
    }
    signature = hmac.new(_key(key), _canonical(payload), hashlib.sha256).hexdigest()
    return RiskApproval(**payload, signature=signature)


def write_risk_approval(approval: RiskApproval, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    temp.write_text(json.dumps(asdict(approval), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def validate_risk_approval(path: Path, *, client_order_id: str, symbol: str, side: str,
                           quantity: int, model_sha256: str, quote_timestamp: datetime | str,
                           now: datetime | None = None, key: bytes | None = None) -> RiskApproval:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        signature = raw.pop("signature")
    except (OSError, json.JSONDecodeError, KeyError, AttributeError) as exc:
        raise ValueError("risk approval is unavailable or invalid") from exc
    if not isinstance(raw, dict) or not isinstance(signature, str):
        raise ValueError("risk approval is malformed")
    expected = hmac.new(_key(key), _canonical(raw), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("risk approval signature is invalid")
    try:
        approval = RiskApproval(**raw, signature=signature)
    except TypeError as exc:
        raise ValueError("risk approval schema is invalid") from exc
    if approval.schema != SCHEMA or approval.approved is not True:
        raise ValueError("risk approval is not approved")
    if (approval.client_order_id != client_order_id or approval.symbol != symbol.upper() or
            approval.side != side or approval.quantity != quantity or
            approval.model_sha256 != model_sha256 or approval.quote_timestamp != _utc(quote_timestamp).isoformat()):
        raise ValueError("risk approval does not match the proposed order")
    reference = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    issued, expires = _utc(approval.issued_at), _utc(approval.expires_at)
    if expires <= issued or reference < issued or reference >= expires:
        raise ValueError("risk approval is expired or not yet valid")
    return approval
