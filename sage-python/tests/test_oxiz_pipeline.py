"""Tests for PolicyVerifier.verify_node — node-scoped checks."""
from __future__ import annotations
from unittest.mock import MagicMock


class MockNode:
    def __init__(self, security_label=0, max_cost_usd=1.0):
        self.security_label = security_label
        self.max_cost_usd = max_cost_usd


def test_verify_node_passes_valid():
    from sage.contracts.policy import PolicyVerifier
    node = MockNode(security_label=1, max_cost_usd=2.0)
    preds = [MockNode(security_label=0)]
    assert PolicyVerifier.verify_node(node, preds, budget_remaining=5.0) is True


def test_verify_node_fails_info_flow():
    from sage.contracts.policy import PolicyVerifier
    node = MockNode(security_label=0)  # public node
    preds = [MockNode(security_label=2)]  # confidential predecessor
    assert PolicyVerifier.verify_node(node, preds, budget_remaining=5.0) is False


def test_verify_node_fails_budget():
    from sage.contracts.policy import PolicyVerifier
    node = MockNode(security_label=0, max_cost_usd=10.0)
    assert PolicyVerifier.verify_node(node, [], budget_remaining=1.0) is False


def test_verify_node_fails_fan_in():
    from sage.contracts.policy import PolicyVerifier
    node = MockNode()
    preds = [MockNode() for _ in range(6)]  # 6 > max_fan_in=5
    assert PolicyVerifier.verify_node(node, preds, budget_remaining=100.0) is False


def test_verify_node_no_predecessors():
    from sage.contracts.policy import PolicyVerifier
    node = MockNode(security_label=2, max_cost_usd=1.0)
    assert PolicyVerifier.verify_node(node, [], budget_remaining=5.0) is True
