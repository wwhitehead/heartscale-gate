#!/usr/bin/env python3
"""Generate the Colab-ready demo notebook from plain Python cell lists.

Authoring as code keeps the .ipynb valid and reviewable.
Run:  python tools/build_notebook.py
"""

import json
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "notebooks")
REPO = "wwhitehead/heartscale-gate"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": "\n".join(lines)}


CELLS = [
    md(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
       f"(https://colab.research.google.com/github/{REPO}/blob/main/notebooks/heartscale_demo.ipynb)"),
    md("# HeartScale Gate — drop-in coherence gating for any LLM",
       "",
       "Add one `LogitsProcessor` and watch a small model decode under a Ma'at",
       "*coherence budget*: high-surprisal tokens get masked at sampling time, with",
       "a **fail-open** safety net so generation never stalls. No fine-tuning.",
       "",
       "> **Honesty:** the masking math is deterministic and unit-tested (cell 2).",
       "> The before/after is the *mechanism* on a real model — not a benchmark",
       "> claim that gated text scores higher on any metric."),
    code("# Colab setup",
         "try:",
         "    import heartscale_gate  # noqa",
         "except ImportError:",
         "    !pip -q install torch transformers",
         "    !git clone -q https://github.com/" + REPO + ".git /content/heartscale-gate || true",
         "    import sys; sys.path.insert(0, '/content/heartscale-gate')",
         "    import heartscale_gate  # noqa",
         "print('heartscale_gate', heartscale_gate.__version__)"),
    md("## 1 · The whole trick — 5 lines"),
    code("from transformers import AutoModelForCausalLM, AutoTokenizer",
         "from heartscale_gate import HeartScaleLogitsProcessor",
         "import torch",
         "",
         "tok = AutoTokenizer.from_pretrained('distilgpt2')",
         "model = AutoModelForCausalLM.from_pretrained('distilgpt2').eval()",
         "if tok.pad_token_id is None: tok.pad_token = tok.eos_token",
         "",
         "gate = HeartScaleLogitsProcessor.from_capacity(0.1)   # the coherence budget",
         "enc = tok('My honest advice to a young person starting out is', return_tensors='pt')",
         "torch.manual_seed(7)",
         "out = model.generate(**enc, max_new_tokens=35, do_sample=True, temperature=1.1,",
         "                     top_k=0, top_p=1.0, pad_token_id=tok.pad_token_id,",
         "                     logits_processor=[gate])",
         "print(tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True))",
         "print(gate.stats())"),
    md("## 2 · Proven, no model needed — the gate's math is unit-tested",
       "These pass on synthetic logits in <2s, anywhere."),
    code("!cd /content/heartscale-gate 2>/dev/null && python -m pytest -q tests/ || "
         "python -m pytest -q tests/"),
    md("## 3 · The capacity knob — a real before/after",
       "Same prompt, same seed; only the budget changes. Watch the text steer, and",
       "the **fail-open** counter climb once the budget gets too strict."),
    code("from heartscale_gate.compare import capacity_sweep, _print",
         "_print(capacity_sweep('My honest advice to a young person starting out is',",
         "                      max_new_tokens=35))"),
    md("## 4 · Try your own prompt & model",
       "Swap in `gpt2`, `EleutherAI/pythia-160m`, or any causal LM."),
    code("from heartscale_gate.compare import generate_compare, _print",
         "_print(generate_compare('The most important thing I learned was',",
         "                        model_id='distilgpt2', capacity=0.1, max_new_tokens=35))"),
    md("---",
       "Verify the underlying HeartScale math from scratch: "
       "[`aamt-reproduce`](https://github.com/wwhitehead/aamt-reproduce). "
       "Code BUSL-1.1 · © 2026 Weslyn Whitehead Jr. / AsAManThinks Research. "
       "Patent pending USPTO #64/040,504, #64/040,509, #64/040,513."),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    nb = {
        "cells": [{**c, "source": c["source"].splitlines(keepends=True)} for c in CELLS],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = os.path.join(OUT, "heartscale_demo.ipynb")
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()
