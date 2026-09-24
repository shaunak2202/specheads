#!/usr/bin/env python3
"""Build training prompt pools and the fixed eval sets, then decontaminate.

Writes `data/prompts/*.json` and populates `data/manifest.json` with pinned
dataset ids, licenses, counts, content hashes and the decontamination report.

Eval sets are sampled once with a fixed seed and never touched again -- they are
not used for training, nor for any hyperparameter choice (that is what the
validation split is for).
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from specheads.train.prompts import (
    EVAL_SOURCES,
    SOURCES,
    PromptSet,
    decontaminate,
    extract_prompt,
)
from specheads.utils.env import git_commit


def take_training_prompts(domain: str, spec: dict, limit: int) -> PromptSet:
    kwargs = {"split": spec["split"], "streaming": True}
    if "config" in spec:
        kwargs["name"] = spec["config"]
    dataset = load_dataset(spec["id"], **kwargs)

    prompts: list[str] = []
    seen: set[str] = set()
    for row in dataset:
        prompt = extract_prompt(domain, row)
        if not prompt or not prompt.strip():
            continue
        prompt = prompt.strip()
        # Very long prompts blow up the distillation context for no benefit.
        if len(prompt) > 2000 or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
        if len(prompts) >= limit:
            break

    return PromptSet(domain, spec["id"], spec["license"], prompts)


def take_eval_prompts(domain: str, spec: dict, n: int, seed: int) -> PromptSet:
    kwargs = {"split": spec["split"]}
    if "config" in spec:
        kwargs["name"] = spec["config"]
    dataset = load_dataset(spec["id"], **kwargs)

    prompts = []
    for row in dataset:
        prompt = extract_prompt(domain, row)
        if prompt and prompt.strip():
            prompts.append(prompt.strip())

    if len(prompts) > n:
        rng = random.Random(seed)
        prompts = rng.sample(prompts, n)
    return PromptSet(domain, spec["id"], spec["license"], prompts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat", type=int, default=500, help="chat training prompts")
    parser.add_argument("--code", type=int, default=250)
    parser.add_argument("--math", type=int, default=250)
    parser.add_argument("--eval-n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    limits = {"chat": args.chat, "code": args.code, "math": args.math}
    prompts_dir = args.out / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    eval_sets = {
        domain: take_eval_prompts(domain, spec, args.eval_n, args.seed)
        for domain, spec in EVAL_SOURCES.items()
    }
    all_eval = [p for s in eval_sets.values() for p in s.prompts]
    print(f"eval sets: {[(d, len(s)) for d, s in eval_sets.items()]}")

    manifest: dict = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "seed": args.seed,
        "datasets": {},
        "eval_sets": {},
        "decontamination": {
            "method": "normalised 13-gram overlap, training prompts vs all eval prompts",
            "training_prompts_dropped": 0,
            "per_domain": {},
        },
    }

    for domain, prompt_set in eval_sets.items():
        (prompts_dir / f"eval_{domain}.json").write_text(json.dumps(prompt_set.prompts, indent=2))
        manifest["eval_sets"][domain] = {
            "source": prompt_set.source,
            "license": prompt_set.license,
            "split": EVAL_SOURCES[domain]["split"],
            "count": len(prompt_set),
            "sha256": prompt_set.content_hash,
        }

    total_dropped = 0
    for domain, spec in SOURCES.items():
        raw = take_training_prompts(domain, spec, limits[domain])
        kept, dropped = decontaminate(raw.prompts, all_eval)
        total_dropped += dropped
        clean = PromptSet(domain, raw.source, raw.license, kept)

        (prompts_dir / f"train_{domain}.json").write_text(json.dumps(clean.prompts, indent=2))
        manifest["datasets"][domain] = {
            "source": clean.source,
            "license": clean.license,
            "split": spec["split"],
            "requested": limits[domain],
            "retrieved": len(raw),
            "count": len(clean),
            "sha256": clean.content_hash,
        }
        manifest["decontamination"]["per_domain"][domain] = {
            "checked": len(raw),
            "dropped": dropped,
        }
        print(f"train {domain:5} retrieved={len(raw):4d} dropped={dropped:3d} kept={len(clean):4d}")

    manifest["decontamination"]["training_prompts_dropped"] = total_dropped
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out / 'manifest.json'} (dropped {total_dropped} contaminated prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
