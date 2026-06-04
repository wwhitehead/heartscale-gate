# HeartScale Gate

**Add 5 lines, and any 🤗 transformers language model decodes under a Ma'at
coherence budget — high-surprisal tokens get masked at sampling time, with a
fail-open safety net so generation never stalls. One `LogitsProcessor`, no
fine-tuning, no retraining, runs on a laptop.**

[![Tests](https://img.shields.io/badge/unit_tests-9%2F9_passing-22c55e)](./tests/test_processor.py)
[![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-C49A26)](./LICENSE)
[![Verify the math](https://img.shields.io/badge/verify_the_math-aamt--reproduce-6366f1)](https://github.com/wwhitehead/aamt-reproduce)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from heartscale_gate import HeartScaleLogitsProcessor

tok   = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

gate = HeartScaleLogitsProcessor.from_capacity(0.1)          # ← the whole trick
out  = model.generate(**tok("My honest advice is", return_tensors="pt"),
                      max_new_tokens=35, do_sample=True,
                      logits_processor=[gate])                # ← drop it in
print(tok.decode(out[0]), gate.stats())
```

---

## What it does — a real before/after (distilgpt2, same prompt, same seed)

The only knob is **`capacity` ∈ [0, 1]** — the agent's coherence budget. As it
tightens, the gate masks more of the sampling distribution and steers the model;
push it too far and the **fail-open safety** takes over so output never stops.
This is *actual* captured output, not an illustration:

| capacity | budget | what comes out |
|---:|---:|---|
| **none** (baseline) | ∞ | *…get more **plants muslim** in cultivating a floral specifically for their family market…* |
| `0.20` | 12.8 | *…get more plants muslim in cultivating a floral…* (budget too loose → ~no effect) |
| `0.12` | 7.7 | *…get more **meaningful and real in life**, but without that kind of uncertainty — my decision was made upon **conscious observation**.* |
| `0.08` | 5.1 | *…take the steps to **build trust in your own life**. You don't always have that kind of trust…* |
| `0.05` | 3.2 | *…start off with the basics of the game. The game itself is a very simple game…* (very conservative) |
| `0.03` | 1.9 | *…release the bathroom appliance… Please enable Javascript…* — **fail-open fired 14×** (budget so strict the gate mass-masks; safety keeps it alive) |

The baseline drifts into *"plants muslim"*; at `capacity≈0.1` the same model,
same seed, instead produces *"meaningful and real in life… conscious
observation."* That swing is the gate — nothing else changed.

Reproduce it yourself:

```bash
git clone https://github.com/wwhitehead/heartscale-gate && cd heartscale-gate
pip install -e ".[demo,dev]"
pytest -q                                   # 9 passed — pure-math, no model needed
python -m heartscale_gate.compare --sweep   # the table above, on your machine
```

Or open the Colab notebook — zero setup:
[`notebooks/heartscale_demo.ipynb`](./notebooks/heartscale_demo.ipynb).

---

## How the gate decides (WP-02 / HeartScale HCRS)

For every candidate token at each decoding step:

```
action_weight   = -log p(token) · semantic_load(token)     # surprisal × cost
agent_frequency = RI · BC · 64                             # the capacity budget
```

A token is **"Evenly Yoked"** (the Ma'at feather-weight principle) iff
`action_weight / agent_frequency ≤ 1`. Tokens that cost more than the agent can
bear are masked (logit → −∞). If *every* candidate is masked, the gate restores
the original logits — **fail-open** — so decoding never deadlocks.

`RI` (resonance index) and `BC` (breath coherence) are the agent's live cognitive
state in the full MaiiaM Alchemist system. Here they're just two `[0,1]` knobs,
or one via `from_capacity()` (`ri = bc = √capacity`).

---

## Honesty (same discipline as the rest of AAMT)

- **Proven, runs anywhere:** the masking math, fail-open safety, determinism, and
  validation are covered by 9 unit tests on synthetic logits — `pytest -q`, no
  model download, <2 s.
- **What this demo is:** a deterministic, interpretable bound on per-token
  surprisal. It is **not** a benchmark claim that gated text scores higher on any
  alignment metric — that would be a separate study. What you *can* see directly
  is the mechanism: rejection counts, fail-open events, and the knob steering
  real output.
- **Verify the underlying math:** [`aamt-reproduce`](https://github.com/wwhitehead/aamt-reproduce)
  re-derives the HeartScale safety geometry and termination bound from scratch.

---

## Scope & IP

The closed-form HeartScale math here is published WP-series math (it also runs in
the public demo). The production cognitive-state estimator that supplies RI/BC
from a live TERA/Vortex projection is patent-pending and **not** included.

Patent pending: USPTO #64/040,504, #64/040,509, #64/040,513 · Code: BUSL-1.1 ·
Papers: CC BY 4.0 · © 2026 Weslyn Whitehead Jr. / AsAManThinks Research.
