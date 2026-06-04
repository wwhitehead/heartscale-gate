"""HeartScale — a Ma'at coherence gate for language-model decoding.

Drop one object into ``model.generate(logits_processor=[...])`` and every token
the model proposes is checked against an *agent-capacity budget* before it can
be sampled. Tokens that cost more "effort" than the agent can bear are masked.

The rule (WP-02 / HeartScale HCRS, AsAManThinks Research)
--------------------------------------------------------
For each candidate token at a decoding step:

    action_weight   = -log p(token) * semantic_load(token)   # surprisal × cost
    agent_frequency = RI * BC * frequency_gain               # the capacity budget

A token is **"Evenly Yoked"** (per the Ma'at feather-weight principle) iff

    action_weight / agent_frequency <= 1.0

Tokens that exceed the budget are masked (logit → -inf). If *every* candidate is
masked the gate **fails open** — it restores the original logits so generation
never stalls. ``RI`` (resonance index) and ``BC`` (breath coherence) come from
the model's current TERA/Vortex state in the full system; here they are simply
two knobs in ``[0, 1]`` that set how permissive the gate is.

This module has **no required dependencies beyond torch**. ``transformers`` is
used only for the ``LogitsProcessor`` base class when available; without it the
class is still a plain callable on ``(input_ids, scores)`` tensors.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

# Soft optional import of the transformers base class.
try:
    from transformers.generation.logits_process import LogitsProcessor as _LP
except ImportError:  # pragma: no cover - exercised only without transformers

    class _LP:  # type: ignore[no-redef]
        """Minimal stand-in matching the transformers LogitsProcessor shape."""

        def __call__(
            self, input_ids: torch.Tensor, scores: torch.Tensor
        ) -> torch.Tensor:  # pragma: no cover
            raise NotImplementedError


# Default agent-frequency multiplier from the HeartScale spec (RI * BC * 64).
_DEFAULT_FREQUENCY_GAIN: float = 64.0


def _default_semantic_load(token_ids: torch.Tensor) -> torch.Tensor:
    """Default: every token carries unit semantic load."""
    return torch.ones_like(token_ids, dtype=torch.float32)


class HeartScaleLogitsProcessor(_LP):
    """Mask candidate tokens that exceed the Ma'at capacity gate.

    Parameters
    ----------
    ri:
        Resonance index ∈ [0, 1] — "how focused is the agent right now".
    bc:
        Breath coherence ∈ [0, 1] — "how steady is the agent right now".
    frequency_gain:
        Scalar converting ``RI*BC`` into an agent-frequency budget.
        Default ``64`` matches the HeartScale spec. Larger ⇒ more permissive.
    semantic_load_fn:
        Optional callable mapping a 1-D long tensor of token ids to a per-token
        semantic-load tensor of the same shape. Default = 1.0 for every token.
    fail_open:
        If True (default) and a row has *all* tokens masked, restore that row's
        original scores so generation never stalls.

    Examples
    --------
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> tok = AutoTokenizer.from_pretrained("distilgpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> gate = HeartScaleLogitsProcessor(ri=0.6, bc=0.6)
    >>> ids = tok("The meaning of life is", return_tensors="pt").input_ids
    >>> out = model.generate(ids, max_new_tokens=20, do_sample=False,
    ...                      logits_processor=[gate])
    >>> gate.rejected_count > 0
    True
    """

    def __init__(
        self,
        ri: float,
        bc: float,
        *,
        frequency_gain: float = _DEFAULT_FREQUENCY_GAIN,
        semantic_load_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        fail_open: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= ri <= 1.0:
            raise ValueError(f"ri must be in [0,1]; got {ri}")
        if not 0.0 <= bc <= 1.0:
            raise ValueError(f"bc must be in [0,1]; got {bc}")
        if frequency_gain <= 0.0:
            raise ValueError(f"frequency_gain must be positive; got {frequency_gain}")
        self.ri = float(ri)
        self.bc = float(bc)
        self.frequency_gain = float(frequency_gain)
        self.semantic_load_fn = semantic_load_fn or _default_semantic_load
        self.fail_open = bool(fail_open)
        self.agent_frequency = self.ri * self.bc * self.frequency_gain
        self._rejected_count: int = 0
        self._fail_open_count: int = 0
        self._steps: int = 0

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------
    @classmethod
    def from_capacity(cls, capacity: float, **kw) -> "HeartScaleLogitsProcessor":
        """Build a gate from a single ``capacity`` ∈ [0, 1] knob.

        ``capacity`` maps to ``ri = bc = sqrt(capacity)`` so that the resulting
        ``agent_frequency = capacity * frequency_gain`` scales linearly. 0 is the
        strictest possible gate; 1 is the most permissive.
        """
        if not 0.0 <= capacity <= 1.0:
            raise ValueError(f"capacity must be in [0,1]; got {capacity}")
        s = math.sqrt(capacity)
        return cls(ri=s, bc=s, **kw)

    # ------------------------------------------------------------------
    # LogitsProcessor.__call__
    # ------------------------------------------------------------------
    def __call__(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """Return masked logits. ``scores`` has shape ``(batch, vocab)``."""
        self._steps += 1

        if self.agent_frequency <= 0.0:
            # Zero-capacity agent — every action exceeds capacity.
            if self.fail_open:
                self._fail_open_count += scores.shape[0]
                return scores
            return torch.full_like(scores, float("-inf"))

        # -log p(token), numerically stable.
        log_probs = torch.log_softmax(scores.float(), dim=-1)  # (B, V)
        vocab_size = scores.shape[-1]
        token_ids = torch.arange(vocab_size, device=scores.device, dtype=torch.long)
        sem_load = self.semantic_load_fn(token_ids).to(scores.device).to(scores.dtype)

        action_weight = (-log_probs) * sem_load.unsqueeze(0)  # (B, V)
        ratio = action_weight / max(self.agent_frequency, 1e-9)
        mask = ratio > 1.0  # not "evenly yoked"

        rejected = scores.masked_fill(mask, float("-inf"))
        self._rejected_count += int(mask.sum().item())

        if self.fail_open:
            row_all_masked = mask.all(dim=-1)  # (B,)
            if bool(row_all_masked.any().item()):
                self._fail_open_count += int(row_all_masked.sum().item())
                rejected = torch.where(
                    row_all_masked.unsqueeze(-1).expand_as(scores),
                    scores,
                    rejected,
                )
        return rejected

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def rejected_count(self) -> int:
        """Cumulative count of token candidates masked across all steps."""
        return self._rejected_count

    @property
    def fail_open_count(self) -> int:
        """Cumulative count of (row, step) pairs that hit fail-open."""
        return self._fail_open_count

    @property
    def steps(self) -> int:
        """Number of decoding steps this gate has processed."""
        return self._steps

    def stats(self) -> dict:
        return {
            "ri": self.ri,
            "bc": self.bc,
            "agent_frequency": self.agent_frequency,
            "steps": self._steps,
            "rejected_count": self._rejected_count,
            "fail_open_count": self._fail_open_count,
        }

    def reset_counters(self) -> None:
        self._rejected_count = 0
        self._fail_open_count = 0
        self._steps = 0


__all__ = ["HeartScaleLogitsProcessor"]
