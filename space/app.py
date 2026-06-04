"""HeartScale Gate — interactive Space.

Move the capacity slider and watch the same model, same prompt, same seed decode
differently: high-surprisal tokens get masked, with a fail-open safety net.

The gate itself is the published `heartscale-gate` package (installed from GitHub
via requirements.txt). Runs on CPU on the HF free tier.
"""
from __future__ import annotations

import functools

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from heartscale_gate import HeartScaleLogitsProcessor

MODELS = ["distilgpt2", "gpt2", "EleutherAI/pythia-160m"]

EXAMPLES = [
    "My honest advice to a young person starting out is",
    "The most important thing I learned was",
    "Here is the truth about discipline:",
]


@functools.lru_cache(maxsize=3)
def _load(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model


@torch.no_grad()
def _generate(model_id, prompt, processors, max_new_tokens, seed, temperature):
    tok, model = _load(model_id)
    enc = tok(prompt, return_tensors="pt")
    torch.manual_seed(seed)
    out = model.generate(
        **enc, max_new_tokens=int(max_new_tokens), do_sample=True,
        temperature=float(temperature), top_k=0, top_p=1.0, num_beams=1,
        pad_token_id=tok.pad_token_id, logits_processor=processors,
    )
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def run(prompt, model_id, capacity, max_new_tokens, seed, temperature):
    if not prompt.strip():
        return "Enter a prompt.", "", ""
    baseline = _generate(model_id, prompt, None, max_new_tokens, seed, temperature)
    gate = HeartScaleLogitsProcessor.from_capacity(float(capacity))
    gated = _generate(model_id, prompt, [gate], max_new_tokens, seed, temperature)
    s = gate.stats()
    rej_per_step = s["rejected_count"] // max(s["steps"], 1)
    stats = (
        f"capacity {capacity}  →  agent budget {s['agent_frequency']:.1f}    |    "
        f"masked ~{rej_per_step:,}/step    |    "
        f"fail-open fired {s['fail_open_count']}×"
        + ("   ⚠️ budget so tight the safety net is carrying generation"
           if s["fail_open_count"] else "")
    )
    return baseline, gated, stats


CSS = """
.gradio-container {max-width: 1000px !important}
#title h1 {margin-bottom: 0}
footer {visibility: hidden}
"""

with gr.Blocks(css=CSS, title="HeartScale Gate") as demo:
    gr.Markdown(
        "# 🜂 HeartScale Gate\n"
        "**One knob gates a language model's decoding.** High-surprisal tokens get "
        "masked at sampling time; if everything is masked, the gate *fails open* so "
        "generation never stalls. Same prompt, same seed — only the **coherence "
        "budget** changes. No fine-tuning.\n\n"
        "Lower capacity ⇒ stricter budget ⇒ more masking ⇒ more conservative text. "
        "Too low, and you'll watch the fail-open safety carry it.",
        elem_id="title",
    )
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value=EXAMPLES[0], lines=2, scale=3)
        model_id = gr.Dropdown(MODELS, value="distilgpt2", label="Model", scale=1)
    with gr.Row():
        capacity = gr.Slider(0.02, 1.0, value=0.1, step=0.01,
                             label="capacity  (the coherence budget)")
        max_new = gr.Slider(15, 60, value=35, step=5, label="max new tokens")
        seed = gr.Slider(0, 100, value=7, step=1, label="seed")
        temp = gr.Slider(0.7, 1.5, value=1.1, step=0.05, label="temperature")
    btn = gr.Button("Generate before / after", variant="primary")
    stats = gr.Markdown()
    with gr.Row():
        out_base = gr.Textbox(label="Baseline — no gate", lines=6)
        out_gated = gr.Textbox(label="HeartScale-gated", lines=6)
    gr.Examples(EXAMPLES, inputs=prompt)
    gr.Markdown(
        "Sweet spot for distilgpt2 is around **capacity 0.08–0.12**. "
        "[Repo](https://github.com/wwhitehead/heartscale-gate) · "
        "[Verify the math](https://github.com/wwhitehead/aamt-reproduce) · "
        "BUSL-1.1 · © 2026 AsAManThinks Research"
    )

    btn.click(run, [prompt, model_id, capacity, max_new, seed, temp],
              [out_base, out_gated, stats])

if __name__ == "__main__":
    demo.launch()
