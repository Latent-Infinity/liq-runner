from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from liq.metrics.entitlements import (
    CashRates,
    Entitlement,
    EntitlementError,
    IncompleteEntitlements,
)
from liq.runner.entitlement_portfolio import EntitlementPortfolio
from liq.sim.long_only_ledger import LedgerInputError

EFFECTIVE = datetime(2024, 12, 12, 14, 30, tzinfo=UTC)
PAYMENT = datetime(2025, 1, 9, 14, 30, tzinfo=UTC)


def entitlement() -> Entitlement:
    return Entitlement(
        action_id="tsm-2024-12",
        security_id="US8740391003",
        unit_basis="legal-ADR",
        eligible_quantity=Decimal(1),
        legal_units_per_held_unit=Decimal(1),
        rates=CashRates(
            gross=Decimal("0.608106"),
            net=Decimal("0.480404"),
            withholding=None,
            depositary_fee=None,
        ),
        currency="USD",
        reporting_basis="net",
        effective_at=EFFECTIVE,
        payment_at=PAYMENT,
        available_at=EFFECTIVE,
        provenance=(
            "https://investor.tsmc.com/english/dividends/2q24",
            "Software mechanics; clocks not verified research evidence",
        ),
    )


def test_settlement_transfers_receivable_to_cash_without_nav_change() -> None:
    portfolio = EntitlementPortfolio(initial_cash=Decimal(0), reporting_basis="net")
    portfolio.accrue(entitlement(), at=EFFECTIVE, unit_basis="legal-ADR")
    assert portfolio.ledger.cash == 0
    before = portfolio.mark({}, at=EFFECTIVE)
    portfolio.settle(entitlement().action_id, at=PAYMENT)
    assert portfolio.ledger.cash == before == Decimal("0.480404")
    assert portfolio.mark({}, at=PAYMENT) == before
    with pytest.raises(EntitlementError):
        portfolio.settle(entitlement().action_id, at=PAYMENT)
    assert portfolio.mark({}, at=PAYMENT) == before


def test_cash_rejection_preserves_original_receivable_for_retry() -> None:
    portfolio = EntitlementPortfolio(initial_cash=Decimal(0), reporting_basis="net")
    portfolio.accrue(entitlement(), at=EFFECTIVE, unit_basis="legal-ADR")
    later = PAYMENT + timedelta(minutes=1)
    portfolio.mark({}, at=later)
    with pytest.raises(LedgerInputError):
        portfolio.settle(entitlement().action_id, at=PAYMENT)
    assert portfolio.ledger.cash == 0
    assert portfolio.mark({}, at=later) == Decimal("0.480404")
    portfolio.settle(entitlement().action_id, at=later)
    assert portfolio.ledger.cash == Decimal("0.480404")
    assert portfolio.mark({}, at=later) == Decimal("0.480404")


def test_unavailable_receivable_aborts_nav_before_advancing_ledger_clock() -> None:
    portfolio = EntitlementPortfolio(initial_cash=Decimal(0), reporting_basis="net")
    portfolio.accrue(
        replace(entitlement(), available_at=None), at=EFFECTIVE, unit_basis="legal-ADR"
    )
    with pytest.raises(IncompleteEntitlements):
        portfolio.mark({}, at=PAYMENT)
    assert portfolio.ledger.mark({}, EFFECTIVE) == 0
    with pytest.raises(IncompleteEntitlements):
        portfolio.settle(entitlement().action_id, at=PAYMENT)
    assert portfolio.ledger.cash_movements == ()


def test_accrual_cannot_revise_nav_after_execution_clock_advanced() -> None:
    portfolio = EntitlementPortfolio(initial_cash=Decimal(0), reporting_basis="net")
    later = EFFECTIVE + timedelta(minutes=1)
    assert portfolio.mark({}, at=later) == 0
    with pytest.raises(LedgerInputError):
        portfolio.accrue(entitlement(), at=EFFECTIVE, unit_basis="legal-ADR")
    assert portfolio.mark({}, at=later) == 0


def test_accrual_advances_execution_clock_without_price_or_cash_events() -> None:
    portfolio = EntitlementPortfolio(initial_cash=Decimal(0), reporting_basis="net")
    portfolio.accrue(entitlement(), at=EFFECTIVE, unit_basis="legal-ADR")
    with pytest.raises(LedgerInputError):
        portfolio.ledger.mark({}, EFFECTIVE - timedelta(minutes=1))
    assert portfolio.ledger.cash_movements == ()
    assert portfolio.ledger.fills == ()
