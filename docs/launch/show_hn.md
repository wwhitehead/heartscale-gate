# Show HN post

**Title:** Show HN: HeartScale Gate – gate an LLM's decoding with one LogitsProcessor

**URL:** https://github.com/wwhitehead/heartscale-gate

**First comment (post this yourself immediately after submitting):**

Author here. This came out of a larger inference system I'm building, but this
piece stands alone and I thought it was worth sharing on its own.

It's a drop-in `LogitsProcessor` for HuggingFace `transformers`. The premise:
instead of fine-tuning a model to change its behavior, gate it at decode time.
For every candidate token you compute surprisal (`-log p`); if a token costs more
than a configurable "capacity" budget, you mask it. If everything gets masked, it
fails open so generation never stalls.

What I found interesting: it's one knob, and on a fixed prompt + seed it visibly
steers the output. A tiny model that rambles at one setting produces coherent
text at another — same weights, same seed, only the budget changed. Past a
threshold you can watch the fail-open safety take over in the diagnostics.

I've tried hard to be honest about what it is and isn't. It's a deterministic,
interpretable bound on per-token surprisal — not a claim that gated text scores
higher on any alignment benchmark (that'd be a separate study). The unit tests
cover the masking math, fail-open, determinism, and validation with no model
download; the underlying math is reproducible from scratch in a companion repo
(linked in the README).

Runs on CPU on a laptop. There's a live Space if you want to drag the slider
before reading any code. I'd love to hear where people think a decode-time gate
is genuinely useful vs. where it falls down.
