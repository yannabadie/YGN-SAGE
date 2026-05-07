"""CostTracker.tighten_remaining_budget — Stage B v0 contract tests.

Per ``docs/contracts/SAGE_CLI_PROTOCOL.md`` invariant 7 (timeout/budget
enforcement, attacker-can't-loosen): the CLI ``set_budget`` command
TIGHTENS ONLY. The root guard lives on ``CostTracker``, not in the
CLI parsing layer (cgpro Stage B lock 2026-05-07: "loosen prevention
belongs there, not only in CLI parsing").
"""
from __future__ import annotations

import math

import pytest

from sage.contracts.cost_tracker import BudgetUpdateResult, CostTracker


class TestTightenRemainingBudgetFromFinite:
    """When the tracker starts with a finite budget cap."""

    def test_tighten_to_lower_remaining_accepted(self) -> None:
        tracker = CostTracker(budget_usd=10.0)
        tracker.record_spend(2.0)  # remaining = 8.0
        result = tracker.tighten_remaining_budget(3.0)
        assert result.accepted is True
        assert result.reason == "budget_tightened"
        assert result.remaining == pytest.approx(3.0)
        assert result.total_spent == pytest.approx(2.0)
        assert tracker.budget_usd == pytest.approx(5.0)  # spent + new_remaining

    def test_tighten_to_zero_remaining_REJECTED(self) -> None:
        """Zero is rejected because ``budget_usd <= 0`` is the unlimited
        sentinel — accepting zero would corrupt the cap value (cgpro Stage
        B VERIFY round-2 trap, 2026-05-07)."""
        tracker = CostTracker(budget_usd=10.0)
        tracker.record_spend(4.0)
        original_cap = tracker.budget_usd
        result = tracker.tighten_remaining_budget(0.0)
        assert result.accepted is False
        assert result.reason == "budget_invalid_value"
        assert tracker.budget_usd == original_cap

    def test_loosen_rejected_keeps_old_cap(self) -> None:
        tracker = CostTracker(budget_usd=10.0)
        tracker.record_spend(2.0)  # remaining = 8.0
        original_cap = tracker.budget_usd
        result = tracker.tighten_remaining_budget(20.0)  # would loosen
        assert result.accepted is False
        assert result.reason == "budget_loosen_rejected"
        assert tracker.budget_usd == original_cap
        # Reported state matches pre-call snapshot.
        assert result.budget_usd == original_cap
        assert result.remaining == pytest.approx(8.0)

    def test_equal_remaining_accepted_as_no_op_tightening(self) -> None:
        tracker = CostTracker(budget_usd=10.0)
        tracker.record_spend(2.0)  # remaining = 8.0
        result = tracker.tighten_remaining_budget(8.0)
        assert result.accepted is True
        # Cap is rebased to spent + 8 = 10 (unchanged).
        assert tracker.budget_usd == pytest.approx(10.0)

    def test_smallest_positive_tighten_accepted(self) -> None:
        """Tighten remaining to a tiny positive value still rebases."""
        tracker = CostTracker(budget_usd=10.0)
        tracker.record_spend(2.0)
        result = tracker.tighten_remaining_budget(0.001)
        assert result.accepted is True
        # Cap = 2 + 0.001 = 2.001.
        assert tracker.budget_usd == pytest.approx(2.001)
        assert tracker.remaining == pytest.approx(0.001)


class TestTightenRemainingBudgetFromUnlimited:
    """When the tracker starts unlimited (budget_usd <= 0)."""

    def test_finite_value_accepted_as_tightening_from_infinity(self) -> None:
        tracker = CostTracker(budget_usd=0.0)  # unlimited
        tracker.record_spend(3.0)
        assert tracker.remaining == math.inf
        result = tracker.tighten_remaining_budget(5.0)
        assert result.accepted is True
        assert result.reason == "budget_tightened"
        # New cap = total_spent + new_remaining = 3 + 5 = 8.
        assert tracker.budget_usd == pytest.approx(8.0)
        assert tracker.remaining == pytest.approx(5.0)

    def test_zero_from_unlimited_REJECTED(self) -> None:
        """Zero from unlimited is rejected: ``budget_usd = total_spent + 0
        = total_spent`` would still leave the tracker effectively unlimited
        when total_spent is zero (cgpro Stage B VERIFY round-2 trap,
        2026-05-07)."""
        tracker = CostTracker(budget_usd=0.0)
        tracker.record_spend(2.5)
        result = tracker.tighten_remaining_budget(0.0)
        assert result.accepted is False
        assert result.reason == "budget_invalid_value"
        assert tracker.budget_usd == 0.0  # still unlimited
        assert tracker.remaining == math.inf

    def test_smallest_positive_finite_from_unlimited_accepted(self) -> None:
        """Any positive finite value (incl. very small) is accepted as a
        tightening from infinity. This is the proper way to "freeze" spend
        — the frontend MUST send a positive finite value, not zero."""
        tracker = CostTracker(budget_usd=0.0)
        tracker.record_spend(2.5)
        result = tracker.tighten_remaining_budget(0.001)
        assert result.accepted is True
        # Cap = 2.5 + 0.001 = 2.501.
        assert tracker.budget_usd == pytest.approx(2.501)


class TestTightenRemainingBudgetInvalidValues:
    """Negative / NaN / inf are REJECTED (budget_invalid_value)."""

    @pytest.mark.parametrize(
        "bad_value",
        [0.0, -0.01, -1.0, float("nan"), float("inf"), -float("inf")],
    )
    def test_invalid_value_rejected(self, bad_value: float) -> None:
        tracker = CostTracker(budget_usd=10.0)
        original_cap = tracker.budget_usd
        result = tracker.tighten_remaining_budget(bad_value)
        assert result.accepted is False
        assert result.reason == "budget_invalid_value"
        assert tracker.budget_usd == original_cap

    def test_invalid_from_unlimited_keeps_unlimited(self) -> None:
        tracker = CostTracker(budget_usd=0.0)
        result = tracker.tighten_remaining_budget(float("nan"))
        assert result.accepted is False
        assert tracker.budget_usd == 0.0
        assert tracker.remaining == math.inf


class TestBudgetUpdateResultIsFrozen:
    """The result dataclass is immutable so callers can't mutate the
    reported snapshot to forge a different verdict downstream."""

    def test_frozen_dataclass(self) -> None:
        tracker = CostTracker(budget_usd=10.0)
        result = tracker.tighten_remaining_budget(5.0)
        with pytest.raises((AttributeError, Exception)):
            result.accepted = False  # type: ignore[misc]


class TestRejectedUpdateLeavesBudgetExactlyUnchanged:
    """Defense-in-depth: a rejected update leaves budget_usd byte-identical
    to the pre-call value (cgpro Stage B lock 2026-05-07: 'rejected update
    leaves budget_usd unchanged')."""

    def test_rejected_loosen_byte_identical(self) -> None:
        tracker = CostTracker(budget_usd=7.5)
        tracker.record_spend(2.5)
        before = tracker.budget_usd
        spent_before = tracker.total_spent
        tracker.tighten_remaining_budget(100.0)  # loosen attempt
        assert tracker.budget_usd == before
        assert tracker.total_spent == spent_before

    def test_rejected_invalid_byte_identical(self) -> None:
        tracker = CostTracker(budget_usd=3.0)
        tracker.record_spend(1.0)
        before = tracker.budget_usd
        tracker.tighten_remaining_budget(float("nan"))
        assert tracker.budget_usd == before


class TestIsOverBudgetReflectsTightenedCap:
    """After tightening, ``is_over_budget`` MUST reflect the new cap
    (not the original)."""

    def test_tighten_to_below_current_spent_marks_over_budget(self) -> None:
        tracker = CostTracker(budget_usd=10.0)
        tracker.record_spend(8.0)  # remaining = 2.0
        # Cannot tighten below current spend — but can tighten remaining
        # to a smaller value. After remaining=1, cap=9, spent=8 → still
        # under budget. After spending 2 more, over budget vs new cap.
        result = tracker.tighten_remaining_budget(1.0)
        assert result.accepted is True
        assert tracker.budget_usd == pytest.approx(9.0)
        assert tracker.is_over_budget is False
        tracker.record_spend(2.0)  # spent = 10, cap = 9
        assert tracker.is_over_budget is True
