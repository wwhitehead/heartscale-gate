---
title: HeartScale Gate
emoji: 🪶
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
python_version: "3.11"
app_file: app.py
pinned: true
license: other
short_description: One knob gates any LLM's decoding.
---

# HeartScale Gate — interactive demo

Move the **capacity** slider and watch the same model, same prompt, same seed
decode differently. High-surprisal tokens get masked at sampling time; if every
candidate is masked, the gate **fails open** so generation never stalls.

The gate is one `LogitsProcessor` from the
[`heartscale-gate`](https://github.com/wwhitehead/heartscale-gate) package — no
fine-tuning, no retraining. The underlying HeartScale (WP-02) math is published
and independently reproducible at
[`aamt-reproduce`](https://github.com/wwhitehead/aamt-reproduce).

© 2026 Weslyn Whitehead Jr. / AsAManThinks Research · Code BUSL-1.1 ·
Patent pending USPTO #64/040,504, #64/040,509, #64/040,513.
