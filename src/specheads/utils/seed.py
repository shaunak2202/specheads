"""Deterministic seeding.

Every result folder records the seed it ran under. torch is imported lazily so
the local, CPU-only, torch-free environment can still run the config and
manifest tests.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedState:
    """What was actually seeded, for the record written alongside results."""

    seed: int
    torch_seeded: bool
    cuda_seeded: bool
    deterministic_algorithms: bool


def seed_everything(seed: int, deterministic: bool = False) -> SeedState:
    """Seed python, numpy and torch.

    Args:
        seed: the master seed.
        deterministic: also request deterministic cuDNN kernels. This is off by
            default because it measurably slows matmuls, and every number this
            project reports is a wall-clock timing -- a determinism flag that
            changes throughput would corrupt the thing being measured. Turn it
            on for correctness tests, not for benchmarks.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    torch_seeded = cuda_seeded = False
    try:
        import torch

        torch.manual_seed(seed)
        torch_seeded = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            cuda_seeded = True
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        deterministic = False

    return SeedState(
        seed=seed,
        torch_seeded=torch_seeded,
        cuda_seeded=cuda_seeded,
        deterministic_algorithms=deterministic,
    )
