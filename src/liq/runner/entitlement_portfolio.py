"""Connect explicit dividend receivables to the existing long-only cash ledger."""

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal, localcontext
from typing import Literal

from liq.core import CashMovement
from liq.core.enums import Currency
from liq.metrics.entitlements import Entitlement, EntitlementBook
from liq.sim.long_only_ledger import LongOnlyLedger


class EntitlementPortfolio:
    """Own a receivables book and ledger for sequential, single-owner replay.

    Use the ledger for trades; use this adapter for dividend settlement and total
    NAV. Concurrent mutation and external dividend credits are unsupported.
    Prepared book copies prevent cash rejection from consuming a receivable.
    """

    def __init__(self, *, initial_cash: Decimal, reporting_basis: Literal["gross", "net"]) -> None:
        self._book = EntitlementBook(currency="USD", reporting_basis=reporting_basis)
        self._ledger = LongOnlyLedger(initial_cash)

    @property
    def ledger(self) -> LongOnlyLedger:
        return self._ledger

    def accrue(self, item: Entitlement, *, at: datetime, unit_basis: str) -> None:
        self._ledger.validate_timestamp(at)
        self._book.accrue(item, at=at, unit_basis=unit_basis)
        self._ledger.advance_timestamp(at)

    def settle(self, action_id: str, *, at: datetime) -> None:
        candidate, amount = self._book.settled_copy(action_id, at=at)
        movement = CashMovement(
            timestamp=at,
            amount=amount,
            currency=Currency.USD,
            movement_type="dividend",
            description=action_id,
        )
        self._ledger.apply_cash_movement(action_id, movement)
        self._book = candidate

    def mark(self, marks: Mapping[str, Decimal], *, at: datetime) -> Decimal:
        receivables = self._book.value(at=at)
        equity = self._ledger.mark(marks, at)
        precision = 1 + sum(
            len(value.as_tuple().digits) + abs(int(value.as_tuple().exponent))
            for value in (equity, receivables)
        )
        with localcontext(prec=precision):
            return equity + receivables
