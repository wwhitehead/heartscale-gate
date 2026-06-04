"""heartscale-gate — a drop-in Ma'at coherence gate for LLM decoding.

    from heartscale_gate import HeartScaleLogitsProcessor
    gate = HeartScaleLogitsProcessor(ri=0.6, bc=0.6)
    model.generate(..., logits_processor=[gate])

The gate masks tokens whose surprisal exceeds an interpretable agent-capacity
budget, deterministically, on any 🤗 transformers causal LM. The closed-form
HeartScale (WP-02) mathematics is published; the production cognitive-state
estimator that supplies RI/BC in the full system is patent-pending and not
included here.

© 2026 Weslyn Whitehead Jr. / AsAManThinks Research. Business Source License 1.1.
Patent pending (USPTO #64/040,504, #64/040,509, #64/040,513).
"""

from .processor import HeartScaleLogitsProcessor

__all__ = ["HeartScaleLogitsProcessor", "compare"]
__version__ = "0.1.0"


def __getattr__(name: str):
    # Lazy: importing `compare` pulls in transformers, so only load on demand.
    if name == "compare":
        from . import compare as _c

        return _c
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
