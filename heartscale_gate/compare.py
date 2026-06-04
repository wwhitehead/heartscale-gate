"""Before/after comparison: the same model, with and without the HeartScale gate.

Loads any 🤗 causal LM (default: distilgpt2 — CPU-friendly, ~350 MB), generates
greedily from a prompt with the gate off and on, and reports what the gate did.

This is deliberately honest about what the gate *is*: a deterministic bound on
per-token surprisal relative to an agent-capacity budget. It is **not** a claim
that the gated text is "more aligned" on any benchmark — that would be a
NEEDS-MODEL-RUN study, not a demo. What you can see directly is the mechanism:
how many candidates the gate rejects, when it fails open, and how the capacity
knob trades permissiveness for conservatism.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .processor import HeartScaleLogitsProcessor


@dataclass
class GenResult:
    label: str
    text: str
    stats: dict = field(default_factory=dict)


def _pick_device():
    # Prefer Metal (MPS) on Apple Silicon — it bypasses the CPU Accelerate
    # BLAS path. Fall back to CUDA, then CPU.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load(model_id: str, device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or _pick_device()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model, device


@torch.no_grad()
def generate_compare(
    prompt: str,
    *,
    model_id: str = "distilgpt2",
    capacity: float = 0.5,
    max_new_tokens: int = 40,
    seed: int = 0,
    do_sample: bool = True,
    temperature: float = 1.1,
) -> list[GenResult]:
    """Return [baseline, gated] generations for the same prompt.

    Sampling is on by default: the HeartScale gate constrains the *sampling
    distribution* by masking high-surprisal tokens, so the sampled text
    genuinely changes. (Under pure greedy decoding the gate is a no-op on the
    chosen token — it only ever masks the low-probability tail the argmax
    never selects — so a greedy before/after would look identical by design.)
    The RNG is reseeded identically before each run so any divergence is the
    gate's effect alone, not sampling noise.
    """
    tok, model, device = _load(model_id)
    enc = tok(prompt, return_tensors="pt").to(device)
    ids = enc.input_ids

    def _gen(processors):
        torch.manual_seed(seed)  # identical RNG start ⇒ fair comparison
        out = model.generate(
            ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            num_beams=1,
            pad_token_id=tok.pad_token_id,
            logits_processor=processors,
        )
        return tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)

    baseline = GenResult("baseline (no gate)", _gen(None))

    gate = HeartScaleLogitsProcessor.from_capacity(capacity)
    gated = GenResult(f"HeartScale gate (capacity={capacity})", _gen([gate]), gate.stats())
    return [baseline, gated]


@torch.no_grad()
def capacity_sweep(
    prompt: str,
    *,
    model_id: str = "distilgpt2",
    capacities=(0.03, 0.05, 0.08, 0.12, 0.2),
    max_new_tokens: int = 40,
    seed: int = 7,
    temperature: float = 1.1,
) -> list[GenResult]:
    """Show the capacity knob with sampling: a tighter budget masks more of the
    distribution and steers the text, until — at the strict extreme — the gate
    mass-masks and the fail-open safety keeps generation alive.

    The default capacities bracket the regime where the budget (capacity * 64)
    enters the real per-token surprisal range (~2-13 nats). Above it the gate is
    effectively a no-op; below it the fail-open count climbs. RNG is reseeded
    identically per run, so differences are the gate's doing, not noise.
    """
    tok, model, device = _load(model_id)
    enc = tok(prompt, return_tensors="pt").to(device)
    ids = enc.input_ids

    # Baseline (no gate) for reference, same RNG start.
    torch.manual_seed(seed)
    base = model.generate(
        ids, attention_mask=enc.attention_mask, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
        num_beams=1, pad_token_id=tok.pad_token_id,
    )
    results = [GenResult("baseline (no gate)",
                         tok.decode(base[0][ids.shape[1]:], skip_special_tokens=True))]

    for c in capacities:
        gate = HeartScaleLogitsProcessor.from_capacity(c)
        torch.manual_seed(seed)
        out = model.generate(
            ids, attention_mask=enc.attention_mask, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
            num_beams=1, pad_token_id=tok.pad_token_id, logits_processor=[gate],
        )
        text = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
        results.append(GenResult(f"capacity={c}", text, gate.stats()))
    return results


def _print(results: list[GenResult]) -> None:
    for r in results:
        print("\n" + "=" * 70)
        print(r.label)
        if r.stats:
            s = r.stats
            rej_per_step = s["rejected_count"] / max(s["steps"], 1)
            print(
                f"  agent_frequency={s['agent_frequency']:.2f}  "
                f"steps={s['steps']}  rejected={s['rejected_count']} "
                f"(~{rej_per_step:.0f}/step)  fail_open={s['fail_open_count']}"
            )
        print("-" * 70)
        print(r.text.strip())


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="HeartScale before/after demo")
    ap.add_argument("--prompt", default="Here is the honest truth about money:")
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--capacity", type=float, default=0.5)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--sweep", action="store_true", help="run a capacity sweep instead")
    args = ap.parse_args()

    if args.sweep:
        print(f"Capacity sweep · model={args.model}\nprompt: {args.prompt!r}")
        _print(capacity_sweep(args.prompt, model_id=args.model,
                              max_new_tokens=args.max_new_tokens))
    else:
        print(f"Before/after · model={args.model}\nprompt: {args.prompt!r}")
        _print(generate_compare(args.prompt, model_id=args.model,
                                capacity=args.capacity,
                                max_new_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()
