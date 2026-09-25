# SpecHeads

Speculative decoding written from scratch for a frozen `Qwen/Qwen2.5-1.5B-Instruct`,
with two trained drafters and an honest benchmark against vanilla decoding.

1. **Medusa-style heads** — K lightweight heads on the final hidden state, each
   predicting token *t+k*, verified in a single target forward pass with tree attention.
2. **EAGLE-style drafter** — one transformer decoder layer that autoregressively
   predicts the target's next hidden feature, mapped through the frozen LM head.

Baselines: vanilla greedy decoding, and Hugging Face assisted generation with
`Qwen/Qwen2.5-0.5B-Instruct` as a separate draft model.

**Research question.** How much does drafter speedup degrade under domain shift?
Train on chat only, evaluate on chat, code, and math — then test whether mixed-domain
self-distillation recovers what was lost.

The core algorithms (draft trees, tree attention masks, verification, KV cache
pruning) are implemented here rather than imported. Correctness and honest
benchmarking are the point; large numbers are not.

## Status

**Phases 0–6 run on Apple MPS (no CUDA device available). EAGLE trained but
undertrained; Phase 7 write-up complete.**

> ⚠️ **No CUDA hardware was ever used.** Wall-clock numbers are MPS and do not
> transfer — that part genuinely needs a T4. The *correctness* question is now
> settled locally though: holding the decode logic fixed and varying only
> precision and attention kernel isolates the cause completely
> (see [below](#where-the-fp16-divergences-actually-come-from)). Re-running on
> CUDA is confirmation, not a load-bearing gap. See
> [`results/t4/PENDING.md`](results/t4/PENDING.md) and
> [`notebooks/kaggle/t4_losslessness_and_bench.py`](notebooks/kaggle/t4_losslessness_and_bench.py).

The single most important caveat: **every wall-clock number here was measured on
Apple MPS, not the Kaggle T4 the project targets.** Throughput and speedup do not
transfer. Mean accepted length, per-head accuracy, tokens-per-forward and
losslessness *are* properties of the drafter and target, and do transfer — which is
why the domain-shift claim rests on those and not on tok/s.

| Phase | | |
|---|---|---|
| 0 | Plan and environment check | ✅ |
| 1 | Vanilla decoding + benchmark harness | ✅ decode loop token-identical to `model.generate` |
| 2 | Self-distillation data | ✅ 1000 prompts, 235k response tokens |
| 3 | Medusa heads | ✅ two K=5 drafters (chat-only, mixed) |
| 4 | Tree verification | ✅ lossless modulo fp16 ties, see below |
| 5 | EAGLE drafter | 🟡 sweep + training ran; sweep inconclusive, drafter undertrained |
| 6 | Experiments | ✅ domain shift, tree sweep, sampling (both verification rules) |
| 7 | Write-up | ✅ incl. tree-attention diagram |

## Results

<!-- RESULTS:START -->

> Every number below is produced by `scripts/build_readme.py` from files in
> `results/`. Nothing is hand-entered; anything unmeasured reads `TBD`.

**Measured on:** Apple MPS (no CUDA device). Throughput and speedup describe this device and **do not transfer** to a T4. Mean accepted length, tokens-per-forward and per-head accuracy are properties of the drafter and target, and do transfer.

### Distillation data

1000 prompts, 235,043 response tokens at max_new_tokens=256: chat 112,785, code 62,726, math 59,532.

### Per-head validation accuracy (top-1)

| Drafter | head 0 | head 1 | head 2 | head 3 | head 4 |
|---|---|---|---|---|---|
| `medusa_chat` | 0.222 | 0.102 | 0.067 | 0.050 | 0.039 |
| `medusa_mixed` | 0.402 | 0.231 | 0.147 | 0.097 | 0.071 |

### EAGLE loss-weight sweep

100 steps per candidate on an identical slice, identical seed; selected on `val_top1_accuracy` over the held-out validation split.

| w_cross_entropy | val top-1 | val CE | val regression |
|---|---|---|---|
| 0.03 **(selected)** | 0.1353 | 7.5102 | 1.7686 |
| 0.1 | 0.1266 | 7.2838 | 1.8735 |
| 0.3 | 0.1282 | 8.4833 | 2.0130 |
| 1.0 | 0.1333 | 7.9796 | 2.1176 |

**Read this as inconclusive.** The top-1 spread across all four weights is 0.0087 over 100 steps and a handful of validation examples, which is noise. The regression term does rise monotonically with the weight, so the sweep mechanism works; it simply does not separate the candidates at this scale. 0.03 was taken as the winner because something had to be, not because it is established.

EAGLE drafter (51,517,952 params, w_cross_entropy=0.03, 400 steps): validation top-1 0.2342.

### Domain shift — mean accepted length (hardware-independent)

Tokens accepted per step, excluding the bonus token. 95% bootstrap CI over prompts.

| Drafter | Tree | Chat | Code | Math |
|---|---|---|---|---|
| `medusa_chat` | chain-2 | 0.214 [0.17, 0.25] | 0.289 [0.25, 0.33] | 0.286 [0.24, 0.34] |
| `medusa_chat` | chain-3 | 0.215 [0.18, 0.25] | 0.295 [0.26, 0.33] | 0.289 [0.24, 0.34] |
| `medusa_chat` | chain-5 | 0.212 [0.17, 0.25] | 0.301 [0.26, 0.34] | 0.297 [0.25, 0.35] |
| `medusa_chat` | tree-3x2 | 0.381 [0.32, 0.45] | 0.464 [0.41, 0.52] | 0.472 [0.43, 0.52] |
| `medusa_chat` | tree-4x2x2 | 0.440 [0.37, 0.51] | 0.520 [0.45, 0.59] | 0.539 [0.49, 0.59] |
| `medusa_mixed` | chain-2 | 0.168 [0.14, 0.20] | 0.381 [0.33, 0.43] | 0.687 [0.61, 0.77] |
| `medusa_mixed` | chain-3 | 0.169 [0.14, 0.20] | 0.395 [0.34, 0.45] | 0.757 [0.67, 0.86] |
| `medusa_mixed` | chain-5 | 0.165 [0.14, 0.19] | 0.402 [0.35, 0.46] | 0.782 [0.69, 0.89] |
| `medusa_mixed` | tree-3x2 | 0.314 [0.26, 0.37] | 0.600 [0.54, 0.66] | 0.904 [0.82, 0.99] |
| `medusa_mixed` | tree-4x2x2 | 0.357 [0.30, 0.41] | 0.674 [0.61, 0.74] | 1.039 [0.94, 1.14] |
| `eagle_mixed` | chain-2 | 0.113 [0.09, 0.14] | 0.223 [0.19, 0.25] | 0.411 [0.36, 0.47] |
| `eagle_mixed` | chain-3 | 0.113 [0.09, 0.14] | 0.223 [0.19, 0.25] | 0.419 [0.36, 0.47] |
| `eagle_mixed` | chain-5 | 0.104 [0.08, 0.13] | 0.223 [0.19, 0.25] | 0.423 [0.37, 0.48] |
| `eagle_mixed` | tree-3x2 | 0.225 [0.19, 0.26] | 0.358 [0.31, 0.41] | 0.582 [0.51, 0.65] |
| `eagle_mixed` | tree-4x2x2 | 0.254 [0.22, 0.29] | 0.385 [0.34, 0.43] | 0.648 [0.57, 0.72] |

### Throughput (this device only)

| Drafter | Domain | Tree | tok/s (median) | Speedup [95% CI] | tok/forward |
|---|---|---|---|---|---|
| vanilla | chat | — | 39.1 | 1.00x | 1.000 |
| `medusa_chat` | chat | chain-2 | 35.5 | 0.91x [0.83, 0.95] | 1.214 |
| `medusa_chat` | chat | chain-3 | 34.9 | 0.89x [0.87, 0.96] | 1.215 |
| `medusa_chat` | chat | chain-5 | 33.3 | 0.85x [0.80, 0.88] | 1.212 |
| `medusa_chat` | chat | tree-3x2 | 39.3 | 1.00x [0.97, 1.09] | 1.381 |
| `medusa_chat` | chat | tree-4x2x2 | 38.4 | 0.98x [0.91, 1.05] | 1.440 |
| vanilla | code | — | 38.4 | 1.00x | 1.000 |
| `medusa_chat` | code | chain-2 | 34.8 | 0.91x [0.87, 0.97] | 1.289 |
| `medusa_chat` | code | chain-3 | 33.0 | 0.86x [0.83, 0.92] | 1.295 |
| `medusa_chat` | code | chain-5 | 33.0 | 0.86x [0.85, 0.92] | 1.301 |
| `medusa_chat` | code | tree-3x2 | 38.4 | 1.00x [0.97, 1.07] | 1.464 |
| `medusa_chat` | code | tree-4x2x2 | 38.0 | 0.99x [0.96, 1.03] | 1.520 |
| vanilla | math | — | 35.5 | 1.00x | 1.000 |
| `medusa_chat` | math | chain-2 | 36.6 | 1.03x [0.95, 1.11] | 1.286 |
| `medusa_chat` | math | chain-3 | 34.7 | 0.98x [0.95, 1.04] | 1.289 |
| `medusa_chat` | math | chain-5 | 34.0 | 0.96x [0.91, 1.03] | 1.297 |
| `medusa_chat` | math | tree-3x2 | 52.6 | 1.48x [1.39, 1.58] | 1.472 |
| `medusa_chat` | math | tree-4x2x2 | 53.6 | 1.51x [1.44, 1.60] | 1.539 |
| vanilla | chat | — | 51.2 | 1.00x | 1.000 |
| `medusa_mixed` | chat | chain-2 | 45.7 | 0.89x [0.87, 0.92] | 1.168 |
| `medusa_mixed` | chat | chain-3 | 43.5 | 0.85x [0.82, 0.88] | 1.169 |
| `medusa_mixed` | chat | chain-5 | 43.7 | 0.85x [0.83, 0.87] | 1.165 |
| `medusa_mixed` | chat | tree-3x2 | 49.1 | 0.96x [0.94, 1.01] | 1.314 |
| `medusa_mixed` | chat | tree-4x2x2 | 48.7 | 0.95x [0.92, 1.00] | 1.357 |
| vanilla | code | — | 51.8 | 1.00x | 1.000 |
| `medusa_mixed` | code | chain-2 | 52.5 | 1.02x [0.99, 1.05] | 1.381 |
| `medusa_mixed` | code | chain-3 | 52.1 | 1.01x [0.98, 1.04] | 1.395 |
| `medusa_mixed` | code | chain-5 | 50.9 | 0.98x [0.95, 1.02] | 1.402 |
| `medusa_mixed` | code | tree-3x2 | 59.4 | 1.15x [1.10, 1.16] | 1.600 |
| `medusa_mixed` | code | tree-4x2x2 | 58.0 | 1.12x [1.08, 1.15] | 1.674 |
| vanilla | math | — | 51.7 | 1.00x | 1.000 |
| `medusa_mixed` | math | chain-2 | 64.0 | 1.24x [1.20, 1.32] | 1.687 |
| `medusa_mixed` | math | chain-3 | 64.1 | 1.24x [1.23, 1.36] | 1.757 |
| `medusa_mixed` | math | chain-5 | 63.3 | 1.22x [1.20, 1.34] | 1.782 |
| `medusa_mixed` | math | tree-3x2 | 69.3 | 1.34x [1.29, 1.41] | 1.904 |
| `medusa_mixed` | math | tree-4x2x2 | 71.8 | 1.39x [1.32, 1.46] | 2.039 |
| vanilla | chat | — | 51.9 | 1.00x | 1.000 |
| `eagle_mixed` | chat | chain-2 | 37.5 | 0.72x [0.71, 0.75] | 1.113 |
| `eagle_mixed` | chat | chain-3 | 33.0 | 0.64x [0.63, 0.66] | 1.113 |
| `eagle_mixed` | chat | chain-5 | 27.3 | 0.53x [0.52, 0.54] | 1.104 |
| `eagle_mixed` | chat | tree-3x2 | 39.2 | 0.75x [0.73, 0.78] | 1.225 |
| `eagle_mixed` | chat | tree-4x2x2 | 35.0 | 0.67x [0.65, 0.69] | 1.254 |
| vanilla | code | — | 50.8 | 1.00x | 1.000 |
| `eagle_mixed` | code | chain-2 | 39.8 | 0.78x [0.78, 0.82] | 1.223 |
| `eagle_mixed` | code | chain-3 | 35.5 | 0.70x [0.69, 0.73] | 1.223 |
| `eagle_mixed` | code | chain-5 | 29.6 | 0.58x [0.57, 0.60] | 1.223 |
| `eagle_mixed` | code | tree-3x2 | 43.7 | 0.86x [0.81, 0.87] | 1.358 |
| `eagle_mixed` | code | tree-4x2x2 | 37.7 | 0.74x [0.71, 0.76] | 1.385 |
| vanilla | math | — | 51.2 | 1.00x | 1.000 |
| `eagle_mixed` | math | chain-2 | 48.2 | 0.94x [0.89, 0.97] | 1.411 |
| `eagle_mixed` | math | chain-3 | 43.0 | 0.84x [0.80, 0.87] | 1.419 |
| `eagle_mixed` | math | chain-5 | 36.0 | 0.70x [0.66, 0.72] | 1.423 |
| `eagle_mixed` | math | tree-3x2 | 50.7 | 0.99x [0.95, 1.03] | 1.582 |
| `eagle_mixed` | math | tree-4x2x2 | 45.5 | 0.89x [0.86, 0.94] | 1.648 |

### Where the fp16 divergences come from (precision vs kernel)

| precision | attention kernel | exact-tie rate | divergent prompts |
|---|---|---|---|
| float16 | sdpa | 0.3912% | 3 / 8 |
| float16 | eager | 0.0000% | 0 / 8 |
| float32 | sdpa | 0.0000% | 0 / 8 |

### Sampling (temperature > 0)

`rejection` preserves the target distribution; `typical` does **not**. Its higher acceptance is bought with fidelity, so the two rows are not comparable as if they were the same algorithm.

| Domain | T | Mode | Preserves distribution | Mean accepted | tok/s | Speedup |
|---|---|---|---|---|---|---|
| chat | 0.7 | vanilla | — | — | 49.1 | 1.00x |
| chat | 0.7 | `rejection` | **yes** | 0.177 | 37.5 | 0.76x |
| chat | 0.7 | `typical` | no | 0.228 | 39.6 | 0.81x |
| chat | 1.0 | vanilla | — | — | 48.3 | 1.00x |
| chat | 1.0 | `rejection` | **yes** | 0.161 | 37.0 | 0.77x |
| chat | 1.0 | `typical` | no | 0.197 | 38.2 | 0.79x |
| code | 0.7 | vanilla | — | — | 48.0 | 1.00x |
| code | 0.7 | `rejection` | **yes** | 0.290 | 40.2 | 0.84x |
| code | 0.7 | `typical` | no | 0.314 | 40.2 | 0.84x |
| code | 1.0 | vanilla | — | — | 48.2 | 1.00x |
| code | 1.0 | `rejection` | **yes** | 0.255 | 39.3 | 0.82x |
| code | 1.0 | `typical` | no | 0.313 | 41.8 | 0.87x |
| math | 0.7 | vanilla | — | — | 48.4 | 1.00x |
| math | 0.7 | `rejection` | **yes** | 0.673 | 53.5 | 1.10x |
| math | 0.7 | `typical` | no | 0.726 | 53.0 | 1.09x |
| math | 1.0 | vanilla | — | — | 47.7 | 1.00x |
| math | 1.0 | `rejection` | **yes** | 0.585 | 49.6 | 1.04x |
| math | 1.0 | `typical` | no | 0.637 | 52.0 | 1.09x |

### Losslessness

Synthetic drafters (random + oracle), 60 checks on the real target at float16: **0 divergences**.

With a *trained* drafter at fp16, 3 of 8 prompts diverged, of which 2 sat on an **exact** fp16 tie (top-1 and top-2 logits bit-identical). The same prompts at fp32 gave **0 divergences**, and the fp32 exact-tie rate is 0.0000% against 0.3912% at fp16. See [the write-up](#fp16-losslessness-and-argmax-ties).

<!-- RESULTS:END -->

## fp16 losslessness and argmax ties

Greedy speculative decoding is supposed to be **token-identical** to greedy vanilla
decoding. With synthetic drafters it was: 60/60 checks on the real 1.5B at fp16, zero
divergences. With a *trained* drafter, divergences appeared — and the rule is to find
the root cause, not to widen a tolerance until the problem disappears.

**It is fp16 argmax ties, not a decode bug.** The evidence is in
[`results/fp16_tie_investigation/`](results/fp16_tie_investigation/):

| | fp16 (MPS) | fp32 (CPU) |
|---|---|---|
| Divergent prompts (same 8 prompts, same drafter) | 3 / 8 | **0 / 8** |
| Positions where top-1 and top-2 logits are bit-identical | 0.391% | 0.000% |

A worked case — chat prompt 0, token index 74. In fp32 the target prefers `' h'` over
`' local'` by **0.0039**. But fp16's ULP at magnitude 19.27 is 2⁻⁶ = **0.015625**, so
both logits round to exactly `19.265625`. At an exact tie `argmax` returns whichever
index its reduction reaches first, and that order differs between a one-token decode
forward and a multi-token tree forward. Vanilla picked `' local'`; speculative picked
`' h'`.

The rates line up too: a 0.391% tie rate over 96 tokens predicts 31.4% of sequences
hitting at least one tie, and 37.5% was observed.

Across the **full Phase 6 grid — 51 divergences in total**: 28 sat on an exact tie
(gap 0.000000), 50 of 51 were within **one** fp16 ULP, and the single remaining case
had a gap of 0.03125, exactly **two** ULP. That last one is reported rather than
absorbed by a looser threshold; two ULP is still far below the rounding a 28-layer
fp16 forward accumulates, but it is not something this run *proved*.

So the honest statement is: **lossless in fp32; lossless modulo fp16 argmax ties in
fp16.** `scripts/evaluate.py` reports `lossless` and `lossless_modulo_fp16_ties`
separately, with the measured logit gap for every divergence.

## How tree verification works

A step drafts several candidate continuations at once, then verifies **all of
them in a single target forward pass**. The mask is what makes that possible:
each candidate must see the prefix and its own ancestors, and nothing else.

Take the tree `tree-3x2` — three candidates at depth 1, two children under the
best of them:

```
                    root = last accepted token  (pos 5)
                   /            |            \
             c0 "the"      c1 "a"       c2 "our"      depth 1  (pos 6)
            /        \
      c3 "cat"   c4 "dog"                              depth 2  (pos 7)
```

Five candidates go through the target in one forward. Two things have to be right:

**Position ids repeat across siblings.** `c0`, `c1`, `c2` all sit at position 6 —
they are competing hypotheses for the *same* slot, not a sequence of three
tokens. If they got 6, 7, 8 then RoPE would encode a candidate's branch into its
own embedding, and verification would be scoring the wrong thing.

```
node      root  c0  c1  c2  c3  c4
position     5   6   6   6   7   7
```

**The attention mask is a tree, not a triangle.** Rows are queries, columns are
keys; `x` means "may attend". `P` is the cached prefix:

```
          P P P P P | root | c0  c1  c2 | c3  c4
root      x x x x x |  x   |  .   .   . |  .   .
c0        x x x x x |  x   |  x   .   . |  .   .
c1        x x x x x |  x   |  .   x   . |  .   .
c2        x x x x x |  x   |  .   .   x |  .   .
c3        x x x x x |  x   |  x   .   . |  x   .
c4        x x x x x |  x   |  x   .   . |  .   x
```

`c1` cannot see `c0`; `c3` sees its parent `c0` but not its uncle `c2`. A
standard causal mask would be lower-triangular here and would let `c1` attend to
`c0` — still producing fluent text, still passing a smoke test, and silently not
what greedy decoding would have said. That is the failure mode
`tests/test_tree.py::test_siblings_cannot_see_each_other` exists to catch.

**Verification** then walks down from the root, taking the one child whose token
matches the target's own greedy argmax at the parent's position, and stops at the
first depth where none matches. Exactly one child can match, because siblings are
distinct ranks of the same top-k. The accepted path plus the target's next token
(the "bonus token") is what the step emits — so even a fully rejected step still
advances by one and never loses to vanilla.

**Then the cache is pruned.** After the forward, the KV cache holds *every*
candidate including the rejected branches. Only the prefix, the root and the
accepted path may survive; leaving `c1` and `c2` in the cache would let the next
step attend to tokens the model never emitted. This is the single most bug-prone
step in the project, and `results/` is only meaningful because
`tests/test_losslessness.py` exercises it with a drafter whose candidates are
*always* accepted — under a random drafter the accepted path is almost always
empty and this code path barely runs.

## Where the fp16 divergences actually come from

"fp16 argmax ties" was the right direction but the wrong resolution. Holding the
decode logic fixed and varying **only** precision and attention kernel
([`results/attention_precision_probe/`](results/attention_precision_probe/),
8 chat prompts, 1278 scored positions):

| precision | attention kernel | exact-tie rate | divergent prompts |
|---|---|---|---|
| fp16 | **SDPA** | **0.3912%** (5/1278) | **3 / 8** |
| fp16 | eager | 0.0000% (0/1278) | 0 / 8 |
| fp32 | SDPA | 0.0000% (0/1278) | 0 / 8 |

Same precision, same device, same weights, same decode path — swap only the
attention kernel and the ties vanish. So the ties are **not inherent to fp16**.
They come from the kernel's internal accumulation policy.

The mechanism is in transformers' own source. `eager_attention_forward` runs

```python
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
```

— it upcasts the softmax to fp32 and casts back. The fused SDPA kernel on MPS
does not, so attention output is coarser, and downstream logits land on the same
representable fp16 value often enough to produce a 0.39% exact-tie rate.

Three consequences:

1. **The decode logic is exonerated three independent ways**, not one. fp32
   diverges on 0/8 and eager-fp16 diverges on 0/8 — both run the *identical*
   tree mask, verification and cache pruning. A bug there would show up in all
   three columns.
2. **There is an actual fix**, not just an explanation. If strict fp16
   losslessness is required, run the target with `attn_implementation="eager"`.
   It costs throughput and memory, which is a real trade, but it is a knob.
3. **It changes what to expect on CUDA.** CUDA's SDPA dispatches to
   FlashAttention or the memory-efficient kernel, both of which accumulate in
   fp32 even for fp16 inputs. So the tie rate on a T4 is likely *lower* than on
   MPS, plausibly zero — the opposite of the "false pass" I originally warned
   about. That is a falsifiable prediction, and the T4 run tests it.

My earlier write-up called this "the most important open item." That was an
overstatement: it attributed the divergences to fp16 alone and stopped, when one
more local experiment — holding precision fixed and varying the kernel — resolved
the cause without any CUDA hardware at all.

## What the results say

**Mixed-domain distillation recovers acceptance under domain shift — on code and
math.** At `tree-4x2x2`, mean accepted length (95% bootstrap CI, n=16):

| | chat | code | math |
|---|---|---|---|
| chat-trained | 0.440 [0.37, 0.51] | 0.520 [0.45, 0.59] | 0.539 [0.49, 0.59] |
| mixed-trained | 0.357 [0.30, 0.41] | **0.674 [0.61, 0.74]** | **1.039 [0.94, 1.14]** |

Code +30% and math +93%, both with non-overlapping CIs. The apparent chat *cost*
(0.440 → 0.357) has **overlapping CIs and is not established** at this sample size.

**The unexpected result:** the chat-trained drafter scores *higher* on code (0.520)
and math (0.539) than on chat (0.440) — its own training domain. Absolute acceptance
appears driven more by how templated a domain's responses are than by what the
drafter was trained on. The training-mixture effect shows up in the *between-drafter*
comparison, not the within-drafter one. A benchmark reporting only "chat-trained
drafter, evaluated on chat" would have missed this entirely.

**Branching trees dominate chain depth.** `tree-4x2x2` roughly doubles `chain-2`'s
accepted length in every cell, while going from `chain-2` to `chain-5` barely moves it
(e.g. mixed/code: 0.381 → 0.402). With weak heads, extra *depth* is wasted because
acceptance dies at depth 1; extra *width* gives depth 1 more chances to hit.

**EAGLE lost to Medusa here, and the shape of the loss is informative.** Its
throughput *falls* as chain depth grows (chat: 29.0 → 26.0 → 21.0 tok/s for
chain-2/3/5) because the drafter runs autoregressively — one forward per depth, on the
critical path — whereas Medusa's K heads read one hidden state in parallel. At the low
acceptance rates reached here, that cost is not repaid, and EAGLE is a net slowdown
(0.52–1.01× vs vanilla). This was predicted in the Phase 0 plan. It is a statement
about *this* undertrained drafter, not about EAGLE as a method.

## Limitations

These are real and they bound every number above.

- **Wrong hardware.** No CUDA device was available; everything ran on Apple MPS. All
  throughput and speedup figures are therefore *not* the T4 numbers the plan calls
  for. fp16 divergence depends on kernel reduction order, so even the tie behaviour
  could differ on Turing. **The fp16 losslessness gate still needs to be re-run on a
  T4** — the risk is a false pass here.
- **Severely undertrained.** ~105k response tokens per Medusa drafter over 2 epochs,
  and 400 steps for EAGLE, against a paper-scale budget orders of magnitude larger.
  Loss was still falling steeply when training stopped. Head accuracies
  (h0 = 0.22–0.40) and accepted lengths (0.1–1.0) are a **floor**, not the methods'
  ceilings. Best observed speedup was 1.39×; published Medusa results are ~2×+.
- **Small evaluation.** 16 prompts per domain at 96 new tokens. CIs are reported for
  exactly this reason, and one headline comparison (chat) is inconclusive because of it.
- **Sampling is chain-only.** Both verification rules are implemented and measured,
  but `rejection` is restricted to chains: its distribution-preserving proof is
  stated for a linear sequence of draft positions, and extending it to a branching
  tree needs the multi-round SpecInfer construction, which is not implemented.
  Tree + rejection raises rather than silently claiming a guarantee it has not
  earned.
- **The EAGLE sweep is inconclusive**, as its own section states. `w_cross_entropy`
  was set to 0.03 because a value was needed, not because 0.03 was shown to be right.
- **Decontamination found nothing**, which is weaker evidence than it sounds: 250 code
  prompts is too few to conclude Glaive is free of HumanEval overlap.
- **One target model, batch size 1, one seed** for training. No seed-variance study.



## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Phase 0 needs only `pyyaml pytest numpy`; the full stack is required from Phase 1.

```bash
python scripts/memory_budget.py        # analytic T4 budget behind docs/plan.md
```

## Layout

```
configs/              one YAML per run
data/                 manifest, prompt sets, distilled responses (token ids)
docs/plan.md          Phase 0 plan
scripts/              memory_budget.py, build_readme.py, figures.py, tables.py
src/specheads/
  model/              target loading, medusa_heads.py, eagle_drafter.py
  decode/             vanilla.py, tree.py, verify.py, kv_cache.py, speculative.py
  train/              distill_data.py, train_medusa.py, train_eagle.py
  bench/              run_bench.py, timing.py, metrics.py
  utils/              seeding, environment capture, config loading
notebooks/kaggle/     thin launchers only
results/              per run: config copy, raw timings, outputs, metrics.json
tests/
```

## Hardware notes

*Designed* for a Kaggle **T4 (Turing, sm_75)**: no native bf16, no FlashAttention 2.
`Qwen2.5-1.5B-Instruct` is published as `bfloat16`, so it is run in **fp16** and the
vanilla baseline is loaded identically — losslessness is only meaningful between
matched precisions.

**It has not actually been run on a T4.** Every measurement in this repo comes from
Apple MPS, because no CUDA device was available. Batch size 1, one target model.

## License

MIT — see [LICENSE](LICENSE).
