"""Unit tests for the HeartScale gate — no model download required.

These exercise the masking math, fail-open safety, determinism, and validation
directly on synthetic logits, so they run in <1s on any machine (and in Colab
without pulling transformers weights)."""

import math

import pytest
import torch

from heartscale_gate import HeartScaleLogitsProcessor


def _logits(*vals):
    return torch.tensor([list(vals)], dtype=torch.float32)


def test_high_surprisal_token_is_masked():
    # Two tokens: one near-certain (low surprisal), one rare (high surprisal).
    # A tight budget should mask the rare one but keep the likely one.
    scores = _logits(5.0, -5.0)  # p ≈ [0.99995, 0.00005]
    gate = HeartScaleLogitsProcessor(ri=0.2, bc=0.2)  # small budget
    out = gate(torch.zeros(1, 1, dtype=torch.long), scores.clone())
    assert torch.isinf(out[0, 1]) and out[0, 1] < 0  # rare token masked
    assert not torch.isinf(out[0, 0])                # likely token kept
    assert gate.rejected_count == 1


def test_generous_budget_masks_nothing():
    scores = _logits(2.0, 1.0, 0.0, -1.0)
    gate = HeartScaleLogitsProcessor(ri=1.0, bc=1.0)  # max budget = 64
    out = gate(torch.zeros(1, 1, dtype=torch.long), scores.clone())
    assert not torch.isinf(out).any()
    assert gate.rejected_count == 0


def test_fail_open_restores_when_all_masked():
    scores = _logits(0.0, 0.0, 0.0)  # uniform → every token equally surprising
    # Zero capacity ⇒ everything would be masked; fail-open must restore.
    gate = HeartScaleLogitsProcessor(ri=0.0, bc=0.0, fail_open=True)
    out = gate(torch.zeros(1, 1, dtype=torch.long), scores.clone())
    assert torch.equal(out, scores)        # restored exactly
    assert gate.fail_open_count == 1


def test_fail_closed_masks_everything():
    scores = _logits(0.0, 0.0, 0.0)
    gate = HeartScaleLogitsProcessor(ri=0.0, bc=0.0, fail_open=False)
    out = gate(torch.zeros(1, 1, dtype=torch.long), scores.clone())
    assert torch.isinf(out).all()


def test_determinism_same_input_same_mask():
    scores = _logits(3.0, 0.5, -2.0, -4.0, 1.0)
    g1 = HeartScaleLogitsProcessor(ri=0.3, bc=0.4)
    g2 = HeartScaleLogitsProcessor(ri=0.3, bc=0.4)
    o1 = g1(torch.zeros(1, 1, dtype=torch.long), scores.clone())
    o2 = g2(torch.zeros(1, 1, dtype=torch.long), scores.clone())
    assert torch.equal(o1, o2)


def test_from_capacity_scales_budget_linearly():
    g = HeartScaleLogitsProcessor.from_capacity(0.25)
    # ri=bc=sqrt(0.25)=0.5 ⇒ agent_frequency = 0.25 * 64 = 16
    assert math.isclose(g.agent_frequency, 16.0, rel_tol=1e-6)


def test_stricter_capacity_rejects_at_least_as_much():
    scores = _logits(*[float(x) for x in range(10, -10, -1)])  # 20-token spread
    strict = HeartScaleLogitsProcessor.from_capacity(0.2)
    loose = HeartScaleLogitsProcessor.from_capacity(0.8)
    strict(torch.zeros(1, 1, dtype=torch.long), scores.clone())
    loose(torch.zeros(1, 1, dtype=torch.long), scores.clone())
    assert strict.rejected_count >= loose.rejected_count


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_validation_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        HeartScaleLogitsProcessor(ri=bad, bc=0.5)
    with pytest.raises(ValueError):
        HeartScaleLogitsProcessor.from_capacity(bad)
