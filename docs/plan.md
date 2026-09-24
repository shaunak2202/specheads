# Phase 0 — Plan and environment check

Status: **awaiting gate approval**. Nothing in Phases 1–7 has been started.

---

## 1. Environment check

### Local (development, CPU only)

| | |
|---|---|
| Machine | Apple M5 Pro, 24 GB unified memory |
| Python | 3.11.15 (`python3.11`) |
| Installed now | `pyyaml`, `pytest`, `numpy` only |
| Not yet installed | torch, transformers, datasets, accelerate, pandas, scipy, matplotlib, wandb |

The heavy stack is deliberately **not** installed yet. Phase 0 needs no tensors, and
the 19 tests in this repo run without torch. It gets installed at Phase 1, when the
CPU tests against a tiny model actually need it.

### Target model — checked against the published config, not from memory

Fetched from `Qwen/Qwen2.5-1.5B-Instruct/config.json`:

| Field | Value |
|---|---|
| `hidden_size` | 1536 |
| `intermediate_size` | 8960 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` / `num_key_value_heads` | 12 / 2 (GQA, `head_dim` 128) |
| `vocab_size` | 151936 |
| `tie_word_embeddings` | **`True`** |
| `torch_dtype` | **`bfloat16`** |

Three consequences worth settling before any code is written:

**`tie_word_embeddings: True` — the LM head *is* the input embedding matrix.**
So "the frozen shared LM head" is not just a memory optimisation, it is a
correctness requirement: any gradient reaching it would corrupt the target's input
embeddings and silently change the model we are supposed to be matching losslessly.
Both drafters must hold it under `requires_grad_(False)` and assert that at train
time. It also means EAGLE's `embed(token_{t+1})` input reads the same tensor as the
output projection — fine, but the same freeze applies.

Scale, from `scripts/memory_budget.py`: one copy is 233,373,696 params, **24.7× the
entire K=4 Medusa head stack**. Four copies would be 933M — 60% of the 1.54B target.

**The config says `bfloat16`, and a T4 (Turing, sm_75) has no native bf16.**
We run the target in **fp16**. This is a real numerics change from the model's
native dtype, and it is exactly the situation Hard Rule 2 anticipates. Two
requirements follow: the vanilla baseline must be loaded in fp16 too, so
losslessness compares like with like; and any divergence gets the logit-gap
evidence the rule demands, not a shrug.

**GQA makes the KV cache almost free.** 2 KV heads × 128 × 28 layers × 2 (K,V) × 2
bytes = **28 KB/token**; 55 MB at 2048 tokens. Tree verification over 64 candidates
costs ~1.8 MB of cache. Memory will not be what limits tree size on a T4 — compute
and kernel-launch overhead will.

---

## 2. Datasets

All licenses below were read from the Hugging Face dataset API, not recalled.

### Training prompts

**Chat — recommend `HuggingFaceH4/ultrachat_200k` (MIT).**

| Option | License | Notes |
|---|---|---|
| **UltraChat 200k** | **MIT** | 200k multi-turn dialogues, already filtered/deduped. Clean license, ample size. |
| `databricks-dolly-15k` | CC BY-SA 3.0 | Human-written, but share-alike is stickier and 15k is thin. |
| `allenai/tulu-3-sft-mixture` | ODC-BY | Mixed provenance; more license surface to audit for no real gain. |

We only need the *prompts* — responses come from self-distillation (Phase 2) — which
makes the license question narrow and UltraChat the obvious pick.

**Code — recommend `glaiveai/glaive-code-assistant` (Apache 2.0).**

| Option | License | Notes |
|---|---|---|
| `Magicoder-OSS-Instruct-75K` | MIT | Higher quality, but synthesised with GPT-3.5, and seeded from The Stack — the more likely of the two to carry HumanEval contamination. |
| **Glaive Code Assistant** | **Apache 2.0** | Clean license, ~140k code Q&A, no OpenAI-derivation question. |

Recommending Glaive on provenance grounds. Happy to switch to Magicoder if you'd
rather have the quality — flagging it as a call for you, not a silent choice.

**Math — recommend `openai/gsm8k`, `train` split (MIT).** Eval is the `test` split,
so disjointness is structural rather than something we have to enforce.
`AI-MO/NuminaMath-CoT` (Apache 2.0) is the alternative if we want harder problems,
but it is partly GSM8K-derived and would need a contamination check against our own
eval set — GSM8K train/test avoids that entirely.

### Eval sets — fixed, disjoint, 80 prompts each

| Domain | Source | License | Selection |
|---|---|---|---|
| Chat | `HuggingFaceH4/mt_bench_prompts` | Apache 2.0 | all 80, turn 1 only |
| Code | `openai/openai_humaneval` | MIT | 80 of 164, seeded sample |
| Math | `openai/gsm8k` (`test`) | MIT | 80 of 1319, seeded sample |

MT-Bench is exactly 80 prompts, which conveniently sets the size for the other two.

### Contamination

Train/eval disjointness is not free here. Code instruction datasets are known to
carry HumanEval problems, and a drafter that has memorised an eval problem would
inflate acceptance for reasons that have nothing to do with speculative decoding.
Phase 2 therefore runs a decontamination pass — normalised 13-gram overlap between
every training prompt and every eval prompt, dropping training hits — and records
how many were dropped in `data/manifest.json`. That count is itself a result worth
reporting: if it is large for code, that is a finding about the dataset.

`data/manifest.json` records name, revision (pinned commit, not `main`), license,
split, count, and SHA-256 of the materialised prompt list for every dataset.

---

## 3. Drafter hyperparameters

### Medusa heads

| Choice | Recommendation | Why |
|---|---|---|
| K (number of heads) | **Train K=5, evaluate depth 1–5** | Heads are independent and cost 2.36M each, so training a 5th is ~2.4M params of insurance. It lets the Phase 6 tree sweep explore depth without retraining — you can always use fewer heads at inference than you trained, never more. |
| Head architecture | **1 ResBlock**: `Linear(1536,1536) + SiLU`, residual, then frozen tied LM head | Medusa's default. 2 ResBlocks doubles head params for what the paper reports as marginal gain; hold it as an ablation if per-head top-1 accuracy turns out to be the bottleneck. |
| Per-head loss weight | `λ_k = 0.8^k` | Later heads are both harder and less often reached; without decay they dominate the gradient. |
| Loss | CE for head k predicting token t+k+1 | Per spec. Report top-1 and top-5 per head on val. |

K=5 → **11,804,160 added params (0.76% of target)**.

### EAGLE drafter

| Choice | Recommendation | Why |
|---|---|---|
| Fusion | `Linear(3072 → 1536)` on `concat(feature_t, embed(token_{t+1}))` | Per spec. 4.72M params. |
| Layer | **One full-width Qwen2 decoder layer** (`intermediate_size` 8960) | Faithful to the paper; establishes the honest number first. |
| Alternative | Narrow the MLP to 4096 | The MLP is 41.3M of the layer's 46.8M, so this is the only real size lever: 51.5M → ~29M. |

Full width = **51,517,952 params (3.34% of target)**.

The tradeoff is sharper than it looks for EAGLE, and worth stating plainly: the
drafter runs **autoregressively, once per draft token**, so its latency is on the
critical path in a way Medusa's parallel heads are not. A drafter that is 5× the
size of a Medusa head stack has to earn that back in acceptance rate. If Phase 6
shows drafting latency eating the speedup, the narrow-MLP variant is the first
thing to sweep — which is why it is worth *measuring* drafter-only latency in Phase
5 rather than inferring it from the end-to-end number.

### Training (both drafters)

Optimizer AdamW; cosine schedule with 100 warmup steps; seq len 1024; batch 1 with
grad accumulation 16; AMP fp16 autocast over fp32 master weights; grad clip 1.0.
LR **1e-3 for Medusa** (small, randomly initialised heads) and **3e-4 for EAGLE**
(larger, and it has a regression term that is easier to destabilise).

**Open question:** EAGLE's loss is `w_reg · SmoothL1(feature) + w_ce · CE`. I do not
want to assert a weighting from memory — published values vary and the right
balance depends on feature scale, which we will not know until Phase 2 produces
real hidden states. Proposal: measure both term magnitudes on the first val batch
and set `w_reg` so they start within ~1 order of magnitude, then tune on val only.

---

## 4. Memory budget (T4, 16 GB)

From `scripts/memory_budget.py` — **analytic, not measured**, and superseded by
`torch.cuda.max_memory_allocated()` in Phases 3 and 5.

| | Medusa (K=4) | EAGLE |
|---|---|---|
| Frozen target, fp16 | 2.88 GB | 2.88 GB |
| Added params | 9.44 M (0.61%) | 51.5 M (3.34%) |
| Optimizer state (fp32 master + grads + Adam m,v) | 0.14 GB | 0.77 GB |
| Logits, fp32, unchunked @ seq 1024 | 2.32 GB | 0.58 GB |
| Logits, fp32, chunked @ 256 | 0.58 GB | 0.15 GB |
| Activation slack (flat allowance) | 1.00 GB | 1.00 GB |
| **Estimated peak** | **5.18 GB** | **4.93 GB** |

**The binding constraint is logits, not weights.** At vocab 151936, one 1024-token
sequence of fp32 logits is 622 MB, and Medusa needs K of them. Unchunked at K=5 and
seq 2048 that alone is 5.8 GB before gradients. So: **cross-entropy is computed in
chunks over the sequence** (256 positions at a time, accumulating loss), which cuts
peak logit memory by the chunk ratio at a small throughput cost.

Both drafters land near 5 GB against 16 GB, so there is real headroom — batch 2–4,
or seq 2048, if Phase 3 shows we want it. Inference is far cheaper: 2.88 GB of
weights plus a 55 MB KV cache at 2048 tokens, plus ~0.99 GB if the HF assisted
baseline's 0.5B draft model is resident.

**GPU budget.** ~30 h/week. Rough allocation: Phase 2 self-distillation is the
single largest line item (two training sets, greedy generation over ~15–20k prompts)
and is checkpointed for resume. Every long run gets a smoke test at ~1% scale first,
per the spec.

---

## 5. Benchmark protocol (frozen at the end of Phase 1)

Recording it here so the gate can approve it before it is locked: batch size 1,
fixed `max_new_tokens` across all domains, 3 repeats per prompt, warmup runs
discarded, `torch.cuda.synchronize()` around every timed region, median and p90
tokens/sec over prompts, bootstrap 95% CIs over prompts. **A vanilla baseline is
re-run in the same Kaggle session as every method**, and speedup is always reported
against that session's baseline — shared GPUs drift, and a baseline from yesterday
is not a baseline.

---

## 6. Open questions for the gate

1. **Code dataset** — Glaive (Apache 2.0, cleaner provenance) or Magicoder (MIT,
   better quality, higher contamination risk)? I recommend Glaive.
2. **K=5 vs K=4** — I recommend training 5 and sweeping depth at inference, since
   the marginal head is ~2.4M params and buys sweep range.
3. **EAGLE loss weighting** — OK to fix empirically from Phase 2 feature statistics
   rather than pinning a number now?
4. **`max_new_tokens`** — proposing 256 for all three domains. Math and code want
   longer generations than chat, but the spec requires it be equal across domains,
   and 256 keeps the Phase 6 sweep affordable.
5. **wandb** — listed in the stack. Use it, or keep runs to local JSON only? Offline
   JSON in `results/` is the source of truth for the README either way, per Rule 1.

---

## Addendum — transformers 5.x findings (from the decode-core work)

The installed transformers is **5.17.0**, not 4.x. Four things were read out of the
installed source before any cache or mask code was written, and all four would have
been wrong from memory:

**Tree attention has a clean injection point.** `Qwen2Model.forward` begins
`if not isinstance(causal_mask_mapping := attention_mask, dict):` — passing
`attention_mask={"full_attention": <4D mask>}` bypasses `create_causal_mask`
entirely and uses ours verbatim. Qwen2.5-1.5B is all `full_attention` layers
(`use_sliding_window: False`), so one key covers every layer.

**An explicit mask disables SDPA's causal flag automatically.** In
`sdpa_attention_forward`: `is_causal = q_length > 1 and attention_mask is None and
is_causal`. So supplying a tree mask turns `is_causal` off with no extra plumbing —
and conversely, under SDPA with no padding `create_causal_mask` returns `None`, not a
tensor, because it leans on that flag.

**`Cache.crop` is mid-deprecation and its sign is load-bearing.** A *negative*
argument removes that many tokens; a *positive* one means "final absolute size",
warns, and is removed in 5.18. The natural-looking `crop(current - length)` truncates
to the wrong position *and* breaks on upgrade. `rollback_to` passes the negative form.

**`crop` cannot express what tree verification needs** anyway — it only truncates
from the end, whereas acceptance must keep the prefix and a scattered subset of
candidate positions. `DynamicLayer` stores `.keys`/`.values` as
``[batch, num_kv_heads, seq, head_dim]`` concatenated on ``dim=-2``, so
`prune_to_indices` does an `index_select` per layer.

One trap outside transformers' own API: **`layer_types` does not shrink when
`num_hidden_layers` is overridden** on a pretrained config. `DynamicCache` reads it,
so a 2-layer test model got 28 cache slots with 26 permanently uninitialised.
`tiny_target` now rewrites `layer_types` explicitly.

`pyproject.toml` is pinned to `transformers>=5.0,<6` accordingly.
