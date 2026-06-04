# Cold-DM-5 Playbook — decode/sampling/inference people

The Busemeyer move worked because it was specific, humble, and gave the recipient
something concrete. Same shape here — but now you attach a **runnable repo**, not
a paper. Goal of the first message is *one reply*, not a meeting.

## Who to target (archetypes, not just names)

Pick 5 to start. Bias toward people who work on **decode-time control** —
they'll get why a `LogitsProcessor` matters in 10 seconds.

1. **Constrained / guided / structured-generation authors** — people behind
   grammar-constrained decoding, JSON/regex-constrained generation, logit
   biasing libraries. A coherence-budget gate is the same family; they'll have
   sharp opinions.
2. **HF `transformers` generation maintainers / contributors** — anyone whose
   name shows up on `LogitsProcessor` / `generate()` PRs. The API fluency is the
   hook.
3. **Sampling-method researchers** — typical/eta/min-p/contrastive-search-style
   decoding folks. Your gate is "another lens on the sampling distribution."
4. **Local-inference toolmakers** — llama.cpp / Ollama / LM Studio ecosystem
   contributors who add samplers. A portable, dependency-light gate is their
   language.
5. **Interpretability-leaning practitioners** — people who like that this is
   *legible* (you can read why each token was masked), not a black box.

How to find them fast: GitHub PR authors on `logits_process.py`, recent arXiv on
"constrained decoding"/"sampling", and whoever the above retweet. Read one real
thing they made before writing.

## The template (adapt every line — never send it generic)

> Hi [name] — I saw your work on [specific thing: their constrained-decoding lib /
> their min-p PR / their sampler]. I built a small `LogitsProcessor` that gates
> decoding against a per-token "capacity budget" (mask if `-log p` exceeds it,
> fail-open if all masked) and your work is exactly the lens I'd want on it.
>
> One knob steers the output on a fixed prompt+seed — there's a 30-sec slider
> demo [Space link] and the repo's 9 tests run with no model download [repo link].
>
> Honest ask: if you have 5 minutes, I'd love to know where you think a
> decode-time gate like this is actually useful vs. where it falls down. No
> agenda — your read would just be really valuable.

Why it works: names their real work, gives them something *touchable*, makes a
small humble ask, and signals epistemic honesty ("where it falls down") that
technical people respect.

## Rules

- **5 at a time, hand-written.** Quality > volume. A batch blast reads as spam and
  burns the names.
- **Lead with their work, not yours.** One sentence on you, max.
- **The ask is a critique, not a pitch.** "Where does this fall down?" gets
  replies; "would you invest/collaborate?" gets silence.
- **Respond within the hour if they bite.** Speed is the conversion. Have the repo
  open, answer their exact question, show the mind behind it.
- **Track replies, not sends.** A reply that says "neat, but X" is a warm lead —
  fix X, tell them you did. That loop is how a stranger becomes an advocate.
- **Don't open with the cosmology.** TERA, Ma'at, the 15 whitepapers — all of it
  comes *after* they're hooked by the gate. Give them the wedge first.

## After a reply lands

1. Answer their technical question precisely (no deflection).
2. If they ran it: ask what surprised them / what broke. Fix the smallest real
   thing they hit, same day, and tell them.
3. Only then, if it's flowing: "this is one piece of a larger inference system —
   happy to show you the rest if you're curious." Let *them* pull.
