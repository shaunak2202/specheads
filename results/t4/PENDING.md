# T4 / CUDA verification — NOT YET RUN

This directory is empty on purpose. **Every measurement in this repository was
taken on Apple MPS.** Nothing here has run on the Kaggle T4 the project targets.

## Update: the correctness question was resolved locally

The original framing here — that the losslessness diagnosis needed CUDA to be
trusted — was wrong. Holding the decode logic fixed and varying only precision
and attention kernel (`results/attention_precision_probe/`) isolated the cause
without any CUDA hardware:

| precision | kernel | tie rate | divergent |
|---|---|---|---|
| fp16 | SDPA | 0.3912% | 3/8 |
| fp16 | eager | 0.0000% | 0/8 |
| fp32 | SDPA | 0.0000% | 0/8 |

The divergences come from SDPA's fp16 softmax accumulation, not from fp16
storage and not from a decode bug — eager fp16 and fp32 both run the identical
tree mask and cache pruning and both diverge zero times.

**What remains genuinely T4-only is the throughput numbers.** Those cannot be
obtained on any other hardware, full stop.

**Prediction for the CUDA run:** CUDA SDPA dispatches to FlashAttention or the
memory-efficient kernel, which accumulate in fp32 even for fp16 inputs. Expect
the tie rate to be *lower* than MPS, plausibly zero. If CUDA instead shows a
*higher* tie rate, or any fp32 divergence, that contradicts the mechanism above
and needs investigating.

## Original framing (kept for the record)

The repo's central correctness claim is that all 51 losslessness divergences are
**fp16 argmax ties**, not a bug in the tree mask or cache pruning. The evidence
(`results/fp16_tie_investigation/`) is strong — fp32 diverges on 0/8 prompts
where fp16 diverges on 3/8, and 50 of 51 divergences sit within one fp16 ULP.

But that evidence was gathered on one backend. fp16 *rounding* is fixed by
IEEE-754 and is identical on any conforming hardware, so the **tie rate** should
reproduce exactly on CUDA. What is *not* fixed is the reduction **order** inside
a kernel, and that is precisely what decides which index `argmax` returns for an
exact tie. So:

- If CUDA shows the **same tie rate** but a **different set** of diverging
  prompts, that confirms the diagnosis and is the expected outcome.
- If CUDA shows **fp32 divergences**, the diagnosis is wrong and there is a real
  bug that MPS happened to hide.
- If CUDA shows a **different tie rate**, something deeper differs and it needs
  investigating before any speedup number is trusted.

The failure mode being guarded against is a **false pass**: green on MPS,
divergent on Turing.

## What to run

`notebooks/kaggle/t4_losslessness_and_bench.py` holds the exact cells, in order.
Notebook settings: **Accelerator = GPU T4 x2**, **Internet = On** (both need a
phone-verified Kaggle account).

The scripts are backend-agnostic and take `--device` / `--fp16-device` /
`--fp32-device`. `investigate_fp16_ties.py` previously hardcoded `"mps"` and has
been fixed; the fix was verified to reproduce the committed MPS numbers
byte-for-byte on the original device pair, so the only variable when it runs on
CUDA is the hardware.

## Known gap

`results/eagle_mixed/drafter.pt` is 197 MB and is not in git, so EAGLE
throughput cannot be re-measured from a clean clone without re-training it or
hosting the checkpoint elsewhere. The two Medusa head checkpoints (45 MB each)
*are* committed, so the critical path — the fp16 tie investigation and Medusa
throughput — works from a clean clone.
