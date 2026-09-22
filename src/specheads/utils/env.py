"""Capture the exact environment a result was produced in.

Kaggle hands out shared GPUs and rebuilds its image regularly, so a timing is
only interpretable next to the machine and library versions that produced it.
Every run directory gets one of these records.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_commit() -> str | None:
    """Current commit hash, or None outside a repo."""
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool | None:
    """Whether the working tree has uncommitted changes.

    Recorded because a timing produced from a dirty tree cannot be tied back to
    a specific commit, which makes it unreproducible no matter what else is
    logged.
    """
    status = _git("status", "--porcelain")
    return None if status is None else bool(status)


def library_versions() -> dict[str, str]:
    """Versions of the libraries that can change results between images."""
    versions: dict[str, str] = {}
    for name in ("torch", "transformers", "datasets", "accelerate", "numpy", "scipy"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = "not installed"
    return versions


def gpu_info() -> dict[str, Any]:
    """GPU name, capability and driver, or a clear marker that there is none."""
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}

    if not torch.cuda.is_available():
        return {"available": False, "reason": "no CUDA device"}

    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    return {
        "available": True,
        "name": props.name,
        "total_memory_gb": round(props.total_memory / 1024**3, 2),
        "capability": f"{capability[0]}.{capability[1]}",
        # Turing (7.5) has no native bf16 and no FlashAttention 2. The target is
        # published as bfloat16, so on a T4 it must be run in fp16 -- a real
        # numerics change that the losslessness tests have to account for.
        "supports_bf16": capability[0] >= 8,
        "driver": torch.version.cuda,
    }


@dataclass(frozen=True)
class EnvRecord:
    """Everything needed to interpret or reproduce a result."""

    python: str
    platform: str
    commit: str | None
    dirty: bool | None
    libraries: dict[str, str] = field(default_factory=dict)
    gpu: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_env() -> EnvRecord:
    """Snapshot the current environment."""
    return EnvRecord(
        python=sys.version.split()[0],
        platform=platform.platform(),
        commit=git_commit(),
        dirty=git_dirty(),
        libraries=library_versions(),
        gpu=gpu_info(),
    )
