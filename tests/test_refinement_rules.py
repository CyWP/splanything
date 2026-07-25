"""Per-primitive tracking for refinement rules.

Regression: a single rule instance shared across multiple primitives
must maintain an independent call counter per primitive. Previously
``RefinementRule`` had a single integer ``calls`` incremented on every
``__call__`` invocation, which (a) double-counted any rule attached to
more than one primitive and (b) ticked on every invocation regardless
of whether ``apply`` actually ran.
"""

from __future__ import annotations

import pytest
import torch

from splanything.primitives import RadialFreqPrimitive
from splanything.training.refinement.base import FilterRule
from splanything.training.refinement.rules import AlphaFilter


class CountingFilter(FilterRule):
    """Filter rule that records every ``apply`` execution for testing."""

    def __init__(self, interval: int = 1):
        super().__init__(interval=interval)
        self.apply_count = 0

    def criterion(self, primitive, **kwargs):
        return torch.ones(len(primitive), device=primitive.device)

    def judge(self, criterion):
        return torch.ones_like(criterion, dtype=torch.bool)

    def apply(self, primitive, **kwargs):
        self.apply_count += 1
        return super().apply(primitive, **kwargs)


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_rule_registers_via_add_filter_rule(device):
    prim = RadialFreqPrimitive(size=10).to(device)
    rule = AlphaFilter(threshold=10.0, interval=1)
    prim.add_filter_rule(rule)
    assert rule.calls(prim) == 0
    assert prim in rule._calls


def test_calls_only_increments_on_execution(device):
    prim = RadialFreqPrimitive(size=10).to(device)
    rule = CountingFilter(interval=2)
    prim.add_filter_rule(rule)
    for _ in range(5):
        prim.check_filter()
    assert rule.apply_count == 2  # ticks 2 and 4 fired
    assert rule.calls(prim) == 2


def test_calls_independent_per_primitive(device):
    a = RadialFreqPrimitive(size=10).to(device)
    b = RadialFreqPrimitive(size=10).to(device)
    rule = CountingFilter(interval=1)
    a.add_filter_rule(rule)
    b.add_filter_rule(rule)

    a.check_filter()
    assert rule.calls(a) == 1 and rule.calls(b) == 0

    b.check_filter()
    b.check_filter()
    assert rule.calls(a) == 1 and rule.calls(b) == 2


def test_shared_rule_independent_interval_gating(device):
    """A rule at interval=N must gate each primitive by its own ticks."""
    a = RadialFreqPrimitive(size=10).to(device)
    b = RadialFreqPrimitive(size=10).to(device)
    rule = CountingFilter(interval=3)
    a.add_filter_rule(rule)
    b.add_filter_rule(rule)

    for _ in range(6):
        a.check_filter()
    for _ in range(3):
        b.check_filter()

    assert rule.calls(a) == 2  # ticks 3 and 6
    assert rule.calls(b) == 1  # tick 3


def test_unregister_resets_counter(device):
    prim = RadialFreqPrimitive(size=10).to(device)
    rule = CountingFilter(interval=1)
    prim.add_filter_rule(rule)
    prim.check_filter()
    assert rule.calls(prim) == 1

    rule.unregister(prim)
    assert rule.calls(prim) == 0
    prim.check_filter()
    assert rule.calls(prim) == 1  # lazy re-register on next __call__


def test_lazy_register_on_call(device):
    """Calling a rule without ``register`` lazily registers the primitive."""
    prim = RadialFreqPrimitive(size=10).to(device)
    rule = CountingFilter(interval=1)
    # Avoid add_filter_rule which would auto-register; force lazy path.
    rule(torch.nn.Identity() if False else prim)
    assert prim in rule._calls
    assert rule.calls(prim) == 1


def test_unregister_removes_primitive(device):
    prim = RadialFreqPrimitive(size=10).to(device)
    rule = CountingFilter(interval=1)
    prim.add_filter_rule(rule)
    assert prim in rule._calls
    rule.unregister(prim)
    assert prim not in rule._calls


def test_base_rule_register_is_idempotent(device):
    prim = RadialFreqPrimitive(size=10).to(device)
    rule = CountingFilter(interval=1)
    rule.register(prim)
    rule._calls[prim] = 5
    rule._ticks[prim] = 9
    rule.register(prim)
    assert rule.calls(prim) == 5
    assert rule._ticks[prim] == 9
