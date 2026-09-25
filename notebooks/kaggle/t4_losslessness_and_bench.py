"""Kaggle T4 launcher: re-run the fp16 losslessness gate and the benchmark on CUDA.

Paste this into a Kaggle notebook cell. Notebook settings must have:
  Accelerator = GPU T4 x2   (one T4 is used; x2 is fine)
  Internet    = On          (needed to clone and to pull the model weights)

Both require a phone-verified Kaggle account.

Why this has to run on real CUDA: every number currently in the repo was measured
on Apple MPS. fp16 *rounding* is fixed by IEEE-754 and is identical everywhere,
but which index `argmax` returns for an exact tie depends on reduction order,
which is a kernel implementation detail. The repo's central claim -- that all 51
divergences are fp16 argmax ties rather than a decode bug -- is therefore only
established for MPS until this is run.

Runtime: roughly 25-40 minutes on one T4, dominated by the benchmark sweep.
"""

SETUP = r"""
set -euxo pipefail
cd /kaggle/working
rm -rf specheads
git clone --depth 1 https://github.com/shaunak2202/specheads.git
cd specheads
# Kaggle images ship torch already; installing the package without deps avoids
# dragging in a different torch build and breaking the CUDA wheel.
pip install -q -e . --no-deps
pip install -q "transformers>=5.0,<6" datasets accelerate
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
"""

# --- Step 0: environment, recorded before anything else -----------------------
STEP_0 = r"""
cd /kaggle/working/specheads
mkdir -p results/t4
python - <<'PY'
import json, torch, platform
from specheads.utils.env import capture_env
env = capture_env().as_dict()
env["cuda"] = {
    "device_name": torch.cuda.get_device_name(0),
    "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
    "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "driver_cuda": torch.version.cuda,
    "supports_bf16": torch.cuda.get_device_capability(0)[0] >= 8,
}
json.dump(env, open("results/t4/env.json", "w"), indent=2)
print(json.dumps(env["cuda"], indent=2))
PY
"""

# --- Step 1: the losslessness gate on the real target, fp16, CUDA -------------
STEP_1 = r"""
cd /kaggle/working/specheads
python scripts/check_losslessness.py \
  --device cuda --dtype float16 --max-new-tokens 96 \
  --out results/t4/losslessness_cuda_fp16
"""

# --- Step 2: the fp16-vs-fp32 tie investigation, BOTH on CUDA ----------------
# Both precisions on one backend, so the only variable is precision. Compare the
# resulting file against results/fp16_tie_investigation/investigation.json (MPS)
# to answer the separate question of whether the backend changes anything.
STEP_2 = r"""
cd /kaggle/working/specheads
python scripts/investigate_fp16_ties.py \
  --n-prompts 8 --max-new-tokens 96 \
  --fp16-device cuda --fp32-device cuda \
  --heads results/medusa_chat/heads.pt \
  --out results/t4/fp16_tie_investigation_cuda
"""

# --- Step 3: real T4 throughput, replacing the provisional MPS numbers --------
# NOTE: heads.pt / drafter.pt are gitignored, so a fresh clone has no trained
# weights. Phase 2+3 must be re-run first, or the checkpoints uploaded as a
# Kaggle Dataset and pointed at with --heads.
STEP_3 = r"""
cd /kaggle/working/specheads
for LABEL in medusa_chat medusa_mixed; do
  python scripts/evaluate.py \
    --heads results/$LABEL/heads.pt --drafter-type medusa --label $LABEL \
    --n-prompts 16 --max-new-tokens 96 --device cuda --dtype float16 \
    --out results/t4/eval_$LABEL
done
python scripts/evaluate.py \
  --heads results/eagle_mixed/drafter.pt --drafter-type eagle --label eagle_mixed \
  --n-prompts 16 --max-new-tokens 96 --device cuda --dtype float16 \
  --out results/t4/eval_eagle_mixed
"""

# --- Step 4: print the comparison that actually answers the question ----------
STEP_4 = r"""
cd /kaggle/working/specheads
python - <<'PY'
import json, glob, os

def load(p):
    return json.load(open(p)) if os.path.exists(p) else None

mps = load("results/fp16_tie_investigation/investigation.json")
cuda = load("results/t4/fp16_tie_investigation_cuda/investigation.json")

print("=" * 64)
print("fp16 TIE INVESTIGATION: MPS (existing) vs CUDA/T4 (this run)")
print("=" * 64)
for name, d in (("MPS", mps), ("CUDA", cuda)):
    if not d:
        print(f"{name}: MISSING"); continue
    print(f"{name:5} fp16 divergent prompts: {d['fp16']['n_divergent_prompts']}/{d['fp16']['n_prompts']}"
          f" | fp32 divergent: {d['fp32']['n_divergent_prompts']}/{d['fp32']['n_prompts']}"
          f" | fp16 exact-tie rate {d['fp16']['exact_tie_rate']:.4%}")
    gaps = sorted(r.get("gap", 0.0) for r in d["fp16"]["divergences"])
    print(f"      divergence gaps: {[round(g,6) for g in gaps]}")

if mps and cuda:
    same_fp32 = mps["fp32"]["n_divergent_prompts"] == cuda["fp32"]["n_divergent_prompts"] == 0
    print()
    print("fp32 lossless on BOTH backends:", same_fp32)
    print("fp16 divergence count identical:",
          mps["fp16"]["n_divergent_prompts"] == cuda["fp16"]["n_divergent_prompts"])
    print("exact-tie RATE identical:",
          abs(mps["fp16"]["exact_tie_rate"] - cuda["fp16"]["exact_tie_rate"]) < 1e-9)
    print()
    print("If the tie RATE matches but WHICH prompts diverge differs, that is the")
    print("expected result: identical IEEE-754 rounding, different reduction order.")

print()
print("=" * 64)
print("T4 THROUGHPUT (real numbers, replacing the provisional MPS ones)")
print("=" * 64)
for f in sorted(glob.glob("results/t4/eval_*/metrics.json")):
    d = json.load(open(f))
    label = d["settings"]["label"]
    best = max((r for r in d["rows"] if r.get("tree")), key=lambda r: r["speedup_vs_vanilla"])
    van = [r for r in d["rows"] if r["drafter"] == "vanilla" and r["domain"] == best["domain"]][0]
    print(f"{label:14} best {best['speedup_vs_vanilla']:.2f}x on {best['domain']}/{best['tree']}"
          f"  ({best['median_tokens_per_second']:.1f} vs vanilla {van['median_tokens_per_second']:.1f} tok/s)"
          f"  accepted={best['mean_accepted_length']:.3f}")
PY
"""

# --- Step 5: get the results back out ----------------------------------------
# Kaggle notebooks cannot push to GitHub without a PAT, and a PAT must never be
# pasted into a notebook -- Kaggle stores notebook source in plaintext and
# public notebooks expose it. Zip the results and download them instead, then
# commit locally.
STEP_5 = r"""
cd /kaggle/working/specheads
zip -r /kaggle/working/t4_results.zip results/t4
echo "Download /kaggle/working/t4_results.zip from the notebook's Output tab,"
echo "unzip into the repo, then commit locally."
"""

CELLS = [SETUP, STEP_0, STEP_1, STEP_2, STEP_3, STEP_4, STEP_5]

if __name__ == "__main__":
    for index, cell in enumerate(CELLS):
        print(f"\n{'#' * 70}\n# CELL {index}\n{'#' * 70}")
        print("%%bash" if cell.strip().startswith(("set -", "cd ")) else "")
        print(cell.strip())
