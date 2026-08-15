"""Intentionally paper-only Alpaca order adapter."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from math import isfinite
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from live.market_data import MarketDataError
from live.risk import OrderIntent


class AlpacaPaperBroker:
    """Never exposes a configurable endpoint: orders can only reach Alpaca paper trading."""

    def __init__(self, key: str | None = None, secret: str | None = None) -> None:
        self.key = key or os.environ.get("ALPACA_PAPER_KEY")
        self.secret = secret or os.environ.get("ALPACA_PAPER_SECRET")
        if not self.key or not self.secret:
            raise MarketDataError("set ALPACA_PAPER_KEY and ALPACA_PAPER_SECRET for paper orders")

    def _get_json(self, path: str) -> object:
        request = Request(f"https://paper-api.alpaca.markets{path}", headers={
            "APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret, "Accept": "application/json"})
        try:
            with urlopen(request, timeout=15) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise MarketDataError(f"Alpaca paper-account HTTP {exc.code}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise MarketDataError("Alpaca paper-account request failed") from exc
        if not isinstance(parsed, (dict, list)):
            raise MarketDataError("Alpaca paper-account response must be an object or array")
        return parsed

    def risk_state(self, symbol: str) -> "PaperRiskState":
        """Read broker source-of-truth immediately before paper submission.

        Daily P&L is account equity minus last equity; if Alpaca cannot provide
        either account or position data, the caller must not submit an order.
        """
        account = self._get_json("/v2/account")
        if not isinstance(account, dict):
            raise MarketDataError("paper account response must be an object")
        try:
            daily_pnl = float(account["equity"]) - float(account["last_equity"])
            if not isfinite(daily_pnl):
                raise ValueError("non-finite daily P&L")
        except (KeyError, TypeError, ValueError) as exc:
            raise MarketDataError("paper account lacks finite equity and last_equity") from exc
        # Read and conservatively reserve open orders *before* position.  If an
        # order fills between these calls, it is either represented in the later
        # position or still reserved at its full original quantity (often both),
        # which can reject extra trades but cannot undercount this race.
        open_orders = self._get_json("/v2/orders?" + urlencode({
            "status": "open", "symbols": symbol.upper(), "limit": "500",
        }))
        if not isinstance(open_orders, list):
            raise MarketDataError("paper open-orders response must be an array")
        if len(open_orders) >= 500:
            raise MarketDataError("paper open-orders query reached its 500-order safety limit")
        pending_buy_quantity = pending_sell_quantity = 0.0
        for order in open_orders:
            if not isinstance(order, dict):
                raise MarketDataError("paper open-orders response contains a non-object order")
            try:
                if str(order["symbol"]).upper() != symbol.upper() or order["side"] not in {"buy", "sell"}:
                    raise ValueError("unexpected order symbol or side")
                quantity, filled = float(order["qty"]), float(order["filled_qty"])
                if not all(isfinite(value) for value in (quantity, filled)) or quantity <= 0 or filled < 0 or filled > quantity:
                    raise ValueError("invalid order quantities")
            except (KeyError, TypeError, ValueError) as exc:
                raise MarketDataError("paper open order lacks valid symbol, side, qty, or filled_qty") from exc
            if order["side"] == "buy":
                pending_buy_quantity += quantity
            else:
                pending_sell_quantity += quantity
        try:
            position = self._get_json(f"/v2/positions/{symbol.upper()}")
            if not isinstance(position, dict):
                raise MarketDataError("paper position response must be an object")
            current_position = float(position["qty"])
            if not isfinite(current_position):
                raise ValueError("non-finite position quantity")
        except (TypeError, ValueError) as exc:
            raise MarketDataError("paper position lacks a finite quantity") from exc
        except MarketDataError as exc:
            # A 404 means no current position; all other account failures must
            # remain fail-closed.  urllib exposes the status through __cause__.
            if isinstance(exc.__cause__, HTTPError) and exc.__cause__.code == 404:
                current_position = 0
            else:
                raise
        return PaperRiskState(current_position=current_position, daily_pnl=daily_pnl,
                              pending_buy_quantity=pending_buy_quantity,
                              pending_sell_quantity=pending_sell_quantity)

    def submit_market_order(self, intent: OrderIntent, client_order_id: str) -> dict[str, object]:
        if not client_order_id or len(client_order_id) > 48:
            raise ValueError("client_order_id must be a non-empty string of at most 48 characters")
        payload = json.dumps({"symbol": intent.symbol.upper(), "qty": str(intent.quantity), "side": intent.side,
                              "type": "market", "time_in_force": "day",
                              "client_order_id": client_order_id}).encode("utf-8")
        # Deliberately local and non-configurable: no instance/class endpoint can
        # be changed into a live-trading URL by a caller.
        request = Request("https://paper-api.alpaca.markets/v2/orders", data=payload, method="POST",
                          headers={"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret,
                                   "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urlopen(request, timeout=15) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise MarketDataError(f"Alpaca paper-order HTTP {exc.code}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise MarketDataError("Alpaca paper-order request failed") from exc
        if not isinstance(parsed, dict):
            raise MarketDataError("Alpaca paper-order response must be an object")
        return parsed

    def order_by_client_order_id(self, client_order_id: str) -> dict[str, object]:
        """Reconcile an uncertain submission before any human retries it."""
        if not client_order_id:
            raise ValueError("client_order_id is required for reconciliation")
        order = self._get_json("/v2/orders:by_client_order_id?" + urlencode({"client_order_id": client_order_id}))
        if not isinstance(order, dict):
            raise MarketDataError("paper order lookup response must be an object")
        return order


@dataclass(frozen=True, slots=True)
class PaperRiskState:
    current_position: float
    daily_pnl: float
    pending_buy_quantity: float = 0.0
    pending_sell_quantity: float = 0.0
