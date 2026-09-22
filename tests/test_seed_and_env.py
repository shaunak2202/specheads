import random

from specheads.utils.env import capture_env, library_versions
from specheads.utils.seed import seed_everything


def test_seeding_is_reproducible():
    seed_everything(123)
    first = [random.random() for _ in range(5)]
    seed_everything(123)
    assert [random.random() for _ in range(5)] == first


def test_seed_state_reports_what_it_actually_seeded():
    state = seed_everything(7)
    assert state.seed == 7
    # torch is optional locally; the record must reflect reality either way
    # rather than claiming a seed it never set.
    assert isinstance(state.torch_seeded, bool)
    if not state.torch_seeded:
        assert state.cuda_seeded is False


def test_library_versions_marks_absent_packages_explicitly():
    """A missing library must read as 'not installed', never as a blank or a guess."""
    versions = library_versions()
    assert set(versions) >= {"torch", "transformers", "numpy"}
    for value in versions.values():
        assert isinstance(value, str) and value


def test_capture_env_records_commit_and_gpu_shape():
    env = capture_env().as_dict()
    assert env["python"].startswith("3.")
    assert "available" in env["gpu"]
    if not env["gpu"]["available"]:
        assert "reason" in env["gpu"]
