# HuggingFace Forums post

**Category:** 🤗Transformers / Show and Tell

**Title:** HeartScale Gate — a drop-in LogitsProcessor that gates decoding with a capacity budget (+ Space)

**Body:**

Sharing a small thing that slots into the `transformers` generation API: a
`LogitsProcessor` that masks high-surprisal tokens at decode time against an
interpretable "agent capacity" budget, with a fail-open fallback so generation
never deadlocks.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from heartscale_gate import HeartScaleLogitsProcessor

tok   = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

gate = HeartScaleLogitsProcessor.from_capacity(0.1)
out  = model.generate(**tok("My honest advice is", return_tensors="pt"),
                      max_new_tokens=35, do_sample=True,
                      logits_processor=[gate])
print(tok.decode(out[0]), gate.stats())
```

It implements cleanly against `LogitsProcessor` (soft-optional import so the core
works with just torch), exposes diagnostics (`rejected_count`, `fail_open_count`,
`steps`), validates its inputs, and is numerically stable (`log_softmax`). One
knob `capacity ∈ [0,1]` sets how permissive the gate is.

- **Space (interactive):** https://huggingface.co/spaces/docwes1/heartscale-gate
- **Repo:** https://github.com/wwhitehead/heartscale-gate (9 unit tests, Colab notebook)
- **The math, reproduced from scratch:** https://github.com/wwhitehead/aamt-reproduce

Honest scope: this demonstrates a *mechanism* (a deterministic surprisal bound),
not a benchmark win. Feedback on the API shape and on where a decode-time gate is
actually useful would be very welcome — especially from folks doing constrained /
guided generation.
