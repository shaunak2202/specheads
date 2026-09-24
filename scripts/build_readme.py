#!/usr/bin/env python3
"""Rebuild the README results section from `results/` only (Hard Rule 1).

Nothing here is hand-written. Anything not found on disk is emitted as `TBD`,
and every table states the device it was measured on, because throughput numbers
do not transfer between machines while acceptance numbers do.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

TREE_ORDER = ["chain-2", "chain-3", "chain-5", "tree-3x2", "tree-4x2x2"]
DOMAINS = ["chat", "code", "math"]


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def fmt(value, spec=".3f", missing="TBD"):
    if value is None:
        return missing
    if isinstance(value, float) and value != value:  # NaN
        return missing
    return format(value, spec) if isinstance(value, (int, float)) else str(value)


def rows_for(metrics, label):
    return [r for r in metrics["rows"] if r["drafter"] == label] if metrics else []


def acceptance_table(evals: dict[str, dict]) -> list[str]:
    """Mean accepted length per drafter per domain -- the domain-shift result."""
    lines = [
        "| Drafter | Tree | Chat | Code | Math |",
        "|---|---|---|---|---|",
    ]
    any_row = False
    for label, metrics in evals.items():
        if not metrics:
            continue
        for tree in TREE_ORDER:
            cells = []
            for domain in DOMAINS:
                match = [
                    r for r in metrics["rows"]
                    if r.get("tree") == tree and r["domain"] == domain and r["drafter"] == label
                ]
                if match:
                    row = match[0]
                    cells.append(
                        f"{row['mean_accepted_length']:.3f} "
                        f"[{row['accepted_ci_low']:.2f}, {row['accepted_ci_high']:.2f}]"
                    )
                else:
                    cells.append("TBD")
            if any(c != "TBD" for c in cells):
                any_row = True
                lines.append(f"| `{label}` | {tree} | " + " | ".join(cells) + " |")
    if not any_row:
        lines.append("| TBD | TBD | TBD | TBD | TBD |")
    return lines


def throughput_table(evals: dict[str, dict]) -> list[str]:
    lines = [
        "| Drafter | Domain | Tree | tok/s (median) | Speedup | tok/forward |",
        "|---|---|---|---|---|---|",
    ]
    for label, metrics in evals.items():
        if not metrics:
            continue
        for domain in DOMAINS:
            baseline = [
                r for r in metrics["rows"] if r["drafter"] == "vanilla" and r["domain"] == domain
            ]
            if baseline:
                b = baseline[0]
                lines.append(
                    f"| vanilla | {domain} | — | {b['median_tokens_per_second']:.1f} | 1.00x | 1.000 |"
                )
            for tree in TREE_ORDER:
                match = [
                    r for r in metrics["rows"]
                    if r.get("tree") == tree and r["domain"] == domain and r["drafter"] == label
                ]
                if match:
                    r = match[0]
                    lines.append(
                        f"| `{label}` | {domain} | {tree} | "
                        f"{r['median_tokens_per_second']:.1f} | "
                        f"{r['speedup_vs_vanilla']:.2f}x | "
                        f"{r['mean_tokens_per_forward']:.3f} |"
                    )
    if len(lines) == 2:
        lines.append("| TBD | TBD | TBD | TBD | TBD | TBD |")
    return lines


def head_accuracy_table(trainings: dict[str, dict]) -> list[str]:
    lines = ["| Drafter | " + " | ".join(f"head {k}" for k in range(5)) + " |",
             "|---" * 6 + "|"]
    any_row = False
    for label, summary in trainings.items():
        if not summary or not summary.get("history"):
            continue
        top1 = summary["history"][-1]["val"]["top1_accuracy"]
        lines.append(f"| `{label}` | " + " | ".join(f"{a:.3f}" for a in top1) + " |")
        any_row = True
    if not any_row:
        lines.append("| TBD |" + " TBD |" * 5)
    return lines


def build_section(results: Path) -> str:
    evals = {
        "medusa_chat": load(results / "eval_medusa_chat" / "metrics.json"),
        "medusa_mixed": load(results / "eval_medusa_mixed" / "metrics.json"),
    }
    trainings = {
        "medusa_chat": load(results / "medusa_chat" / "summary.json"),
        "medusa_mixed": load(results / "medusa_mixed" / "summary.json"),
    }
    lossless = load(results / "losslessness_mps_fp16" / "losslessness.json")
    ties = load(results / "fp16_tie_investigation" / "investigation.json")
    distill = load(results / "distill" / "summary.json")

    any_eval = next((m for m in evals.values() if m), None)
    device = any_eval["env"]["gpu"] if any_eval else None
    device_name = (
        device.get("name") if device and device.get("available") else "Apple MPS (no CUDA device)"
    )

    out: list[str] = [START, ""]
    out += [
        "> Every number below is produced by `scripts/build_readme.py` from files in",
        "> `results/`. Nothing is hand-entered; anything unmeasured reads `TBD`.",
        "",
        f"**Measured on:** {device_name}. "
        "Throughput and speedup describe this device and **do not transfer** to a T4. "
        "Mean accepted length, tokens-per-forward and per-head accuracy are properties "
        "of the drafter and target, and do transfer.",
        "",
    ]

    if distill:
        out += [
            "### Distillation data",
            "",
            f"{distill['n_records']} prompts, {distill['total_response_tokens']:,} response tokens "
            f"at max_new_tokens={distill['settings']['max_new_tokens']}: "
            + ", ".join(f"{k} {v:,}" for k, v in distill["response_tokens_by_domain"].items())
            + ".",
            "",
        ]

    out += ["### Per-head validation accuracy (top-1)", ""] + head_accuracy_table(trainings) + [""]
    out += [
        "### Domain shift — mean accepted length (hardware-independent)",
        "",
        "Tokens accepted per step, excluding the bonus token. 95% bootstrap CI over prompts.",
        "",
    ] + acceptance_table(evals) + [""]
    out += [
        "### Throughput (this device only)",
        "",
    ] + throughput_table(evals) + [""]

    if lossless:
        out += [
            "### Losslessness",
            "",
            f"Synthetic drafters (random + oracle), {lossless['n_checks']} checks on the real "
            f"target at {lossless['settings']['dtype']}: "
            f"**{lossless['n_divergences']} divergences**.",
            "",
        ]
    if ties:
        c = ties["conclusion"]
        out += [
            f"With a *trained* drafter at fp16, {c['fp16_divergences_total']} of "
            f"{ties['fp16']['n_prompts']} prompts diverged, of which "
            f"{c['fp16_divergences_on_exact_ties']} sat on an **exact** fp16 tie "
            f"(top-1 and top-2 logits bit-identical). The same prompts at fp32 gave "
            f"**{c['fp32_divergences_total']} divergences**, and the fp32 exact-tie rate is "
            f"{ties['fp32']['exact_tie_rate']:.4%} against {ties['fp16']['exact_tie_rate']:.4%} "
            f"at fp16. See [the write-up](#fp16-losslessness-and-argmax-ties).",
            "",
        ]

    out += [END]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args()

    section = build_section(args.results)
    text = args.readme.read_text()

    if START in text and END in text:
        head = text.split(START)[0]
        tail = text.split(END, 1)[1]
        args.readme.write_text(head + section + tail)
    else:
        raise SystemExit(
            f"README is missing the {START} / {END} markers; add them where results belong."
        )
    print(f"rebuilt results section in {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
