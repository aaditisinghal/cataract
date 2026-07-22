import random

from patchguard.repro import is_paper_ready, run_fingerprint, seed_everything


def test_seed_everything_is_reproducible():
    seed_everything(123)
    a = [random.random() for _ in range(5)]
    seed_everything(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_everything_seeds_numpy():
    import numpy as np

    seed_everything(7)
    a = np.random.rand(4)
    seed_everything(7)
    b = np.random.rand(4)
    assert (a == b).all()


def test_fingerprint_has_required_keys():
    fp = run_fingerprint()
    for key in ("git_sha", "git_dirty", "python", "platform", "numpy", "torch"):
        assert key in fp
    assert isinstance(fp["git_dirty"], bool)


def test_paper_ready_requires_clean_known_commit():
    assert is_paper_ready({"git_dirty": False, "git_sha": "abc123"}) is True
    assert is_paper_ready({"git_dirty": True, "git_sha": "abc123"}) is False
    assert is_paper_ready({"git_dirty": False, "git_sha": "unknown"}) is False
