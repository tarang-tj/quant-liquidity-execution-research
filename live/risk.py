"""Fail-closed paper-execution risk gates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
import os

from live.market_data import Quote


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_position_shares: int = 10
    max_order_notional: float = 1_000.0
    max_spread_bps: float = 20.0
    max_quote_age_seconds: float = 5.0
    max_daily_loss: float = 100.0


@dataclass(frozen=True, slots=True)
class OrderIntent:
    symbol: str
    side: str
    quantity: int


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    reason: str


def validate_paper_order(intent: OrderIntent, quote: Quote, current_position: float, daily_pnl: float,
                         limits: RiskLimits = RiskLimits(), now: datetime | None = None) -> RiskDecision:
    """Check fresh price, exposure, liquidity, loss, and emergency stop before any paper submission."""
    now = now or datetime.now(timezone.utc)
    if os.environ.get("TRADING_KILL_SWITCH", "0") == "1":
        return RiskDecision(False, "TRADING_KILL_SWITCH is enabled")
    if not isinstance(intent.quantity, int) or isinstance(intent.quantity, bool) or intent.quantity <= 0:
        return RiskDecision(False, "quantity must be a positive whole number")
    if not all(isfinite(value) for value in (daily_pnl, current_position, limits.max_position_shares,
            limits.max_daily_loss, limits.max_order_notional, limits.max_spread_bps, limits.max_quote_age_seconds)):
        return RiskDecision(False, "risk input or limit is non-finite")
    if min(limits.max_position_shares, limits.max_order_notional, limits.max_spread_bps,
           limits.max_quote_age_seconds, limits.max_daily_loss) < 0:
        return RiskDecision(False, "risk limits cannot be negative")
    if quote.timestamp > now:
        return RiskDecision(False, "quote timestamp is in the future")
    if intent.side not in {"buy", "sell"}:
        return RiskDecision(False, "invalid side")
    if not isfinite(quote.spread_bps) or (now - quote.timestamp).total_seconds() > limits.max_quote_age_seconds:
        return RiskDecision(False, "quote is stale")
    if quote.spread_bps > limits.max_spread_bps:
        return RiskDecision(False, "spread exceeds risk limit")
    if daily_pnl <= -abs(limits.max_daily_loss):
        return RiskDecision(False, "daily loss limit reached")
    signed_quantity = intent.quantity if intent.side == "buy" else -intent.quantity
    if abs(current_position + signed_quantity) > limits.max_position_shares:
        return RiskDecision(False, "position limit exceeded")
    if intent.quantity * quote.mid_price > limits.max_order_notional:
        return RiskDecision(False, "order notional limit exceeded")
    return RiskDecision(True, "paper order passed all risk gates")
