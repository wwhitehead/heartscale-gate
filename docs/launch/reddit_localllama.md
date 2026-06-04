# r/LocalLLaMA post

**Title:** I made a drop-in LogitsProcessor that gates decoding with a "coherence budget" — here's what surprised me

**Body:**

I've been experimenting with controlling models at *decode* time instead of fine-tuning, and ended up with something small enough to share. It's one `LogitsProcessor` you drop into `model.generate(...)`.

The idea: give the decoder a budget. For every candidate token, compute its surprisal (`-log p`). If a token costs more than the agent's "capacity" can bear, mask it (logit → -inf). If *every* candidate gets masked, it fails open and restores the original logits so generation never stalls.

```python
from heartscale_gate import HeartScaleLogitsProcessor
gate = HeartScaleLogitsProcessor.from_capacity(0.1)   # one knob
model.generate(**inputs, do_sample=True, logits_processor=[gate])
```

**What surprised me:** there's a single knob (`capacity`), and on the same prompt + same seed it visibly steers the model. distilgpt2 baseline drifts into nonsense ("...get more plants muslim in cultivating a floral..."). At capacity ≈ 0.1 the *same* model, *same* seed instead produces "...get more meaningful and real in life, but without that kind of uncertainty — my decision was made upon conscious observation." Push the budget too tight and you watch the fail-open counter climb in the diagnostics — the safety net is literally carrying generation.

The honest caveat (because this sub will rightly ask): this is **not** a benchmark claim that gated text is "more aligned" on some metric. It's a deterministic, interpretable bound on per-token surprisal. What you can see directly is the mechanism — rejection counts, fail-open events, and the knob steering output. Under pure greedy decoding it's a no-op on the chosen token (it only masks the tail argmax never picks), so it only does interesting things under sampling. I'd genuinely love for people to break it / find where it's useful or useless.

- Repo (9 unit tests, no model download needed to run them): https://github.com/wwhitehead/heartscale-gate
- Live demo (drag the slider): https://huggingface.co/spaces/docwes1/heartscale-gate
- Colab notebook in the repo if you want to run your own model/prompt

Works on CPU, runs on a laptop. Curious what you all find — especially failure modes.
