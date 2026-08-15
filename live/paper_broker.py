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

    def _get_json(self, path: str) -> dict[str, object]:
        request = Request(f"https://paper-api.alpaca.markets{path}", headers={
            "APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret, "Accept": "application/json"})
        try:
            with urlopen(request, timeout=15) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise MarketDataError(f"Alpaca paper-account HTTP {exc.code}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise MarketDataError("Alpaca paper-account request failed") from exc
        if not isinstance(parsed, dict):
            raise MarketDataError("Alpaca paper-account response must be an object")
        return parsed

    def risk_state(self, symbol: str) -> "PaperRiskState":
        """Read broker source-of-truth immediately before paper submission.

        Daily P&L is account equity minus last equity; if Alpaca cannot provide
        either account or position data, the caller must not submit an order.
        """
        account = self._get_json("/v2/account")
        try:
            daily_pnl = float(account["equity"]) - float(account["last_equity"])
            if not isfinite(daily_pnl):
                raise ValueError("non-finite daily P&L")
        except (KeyError, TypeError, ValueError) as exc:
            raise MarketDataError("paper account lacks finite equity and last_equity") from exc
        try:
            position = self._get_json(f"/v2/positions/{symbol.upper()}")
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
        return PaperRiskState(current_position=current_position, daily_pnl=daily_pnl)

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
        return self._get_json("/v2/orders:by_client_order_id?" + urlencode({"client_order_id": client_order_id}))


@dataclass(frozen=True, slots=True)
class PaperRiskState:
    current_position: float
    daily_pnl: float
