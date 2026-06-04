#!/usr/bin/env python3
"""Render the capacity-sweep before/after as a shareable PNG card.

The text below is REAL captured output from `python -m heartscale_gate.compare
--sweep` on distilgpt2 (seed 7, temperature 1.1, 35 new tokens). Re-run the
sweep and paste fresh rows here if you change the defaults.

Usage:  python tools/render_sweep.py            # writes docs/heartscale_sweep.png
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROMPT = "My honest advice to a young person starting out is"

# (label, budget, rejected/step, fail_open, text, accent)
ROWS = [
    ("baseline", "—", "—", "—",
     "a process that will allow you to get more plants muslim in cultivating a "
     "floral specifically for their family market. You should aim to get as many "
     "plants as possible to achieve levels", "#6b7280"),
    ("capacity 0.20", "12.8", "47k", "0",
     "a process that will allow you to get more plants muslim in cultivating a "
     "floral specifically for their family market… (budget too loose → ~no effect)",
     "#9ca3af"),
    ("capacity 0.12", "7.7", "50.1k", "0",
     "a process that will allow you to get more MEANINGFUL AND REAL in life, but "
     "without that kind of uncertainty — my decision was made upon CONSCIOUS "
     "OBSERVATION.", "#22c55e"),
    ("capacity 0.08", "5.1", "50.2k", "0",
     "to take the steps to BUILD TRUST in your own life. You don't always have "
     "that kind of trust. You always have to give them that kind of trust.",
     "#eab308"),
    ("capacity 0.05", "3.2", "50.3k", "0",
     "to start off with the basics of the game. The game itself is a very simple "
     "game. It has a very simple and simple game. (very conservative)", "#f97316"),
    ("capacity 0.03", "1.9", "50.3k", "14",
     "to release the bathroom appliance and foremost, to keep the door open and "
     "shut. Please enable Javascript… (gate mass-masks → FAIL-OPEN safety fires "
     "14×, generation survives)", "#ef4444"),
]

BG = "#0b0f17"
FG = "#e5e7eb"
SUB = "#9ca3af"

fig_h = 1.7 + 0.95 * len(ROWS)
fig, ax = plt.subplots(figsize=(11, fig_h), dpi=160)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, fig_h)
ax.axis("off")

y = fig_h - 0.35
ax.text(2, y, "HeartScale Gate", color=FG, fontsize=22, fontweight="bold", va="top")
ax.text(2, y - 0.5, "One knob steers a model's decoding. Same prompt, same seed — "
        "only the coherence budget changes.",
        color=SUB, fontsize=11, va="top")
ax.text(2, y - 0.92, f"prompt:  “{PROMPT}”   ·   model: distilgpt2",
        color="#60a5fa", fontsize=10, va="top", style="italic")

row_y = y - 1.5
row_h = 0.9
for label, budget, rej, fo, text, accent in ROWS:
    box = FancyBboxPatch((2, row_y - row_h + 0.12), 96, row_h - 0.18,
                         boxstyle="round,pad=0.02,rounding_size=0.06",
                         linewidth=0, facecolor="#111827")
    ax.add_patch(box)
    ax.add_patch(plt.Rectangle((2, row_y - row_h + 0.12), 0.5, row_h - 0.18,
                               facecolor=accent, edgecolor="none"))
    ax.text(3.2, row_y - 0.05, label, color=accent, fontsize=11.5,
            fontweight="bold", va="top")
    meta = f"budget {budget}" + (f"   rej {rej}/step" if rej != "—" else "")
    if fo not in ("—", "0"):
        meta += f"   fail-open {fo}×"
    ax.text(3.2, row_y - 0.42, meta, color=SUB, fontsize=8.5, va="top")
    wrapped = textwrap.fill(text, width=78)
    ax.text(20, row_y - 0.05, wrapped, color=FG, fontsize=9.6, va="top",
            family="monospace")
    row_y -= row_h

ax.text(2, 0.35, "github.com/wwhitehead/heartscale-gate   ·   one LogitsProcessor, "
        "no fine-tuning   ·   math: github.com/wwhitehead/aamt-reproduce",
        color="#6b7280", fontsize=8.5, va="bottom")

out = os.path.join(os.path.dirname(__file__), "..", "docs", "heartscale_sweep.png")
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
print("wrote", os.path.relpath(out))
