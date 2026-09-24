import pytest
import torch

from specheads.decode.tree import TreeSpec
from specheads.decode.verify import accept_path, cache_indices_after_acceptance


def logits_for(vocab: int, argmaxes: list[int]) -> torch.Tensor:
    """Logits whose argmax at row i is argmaxes[i]."""
    out = torch.zeros(len(argmaxes), vocab)
    for row, token in enumerate(argmaxes):
        out[row, token] = 10.0
    return out


def test_accepts_the_full_chain_when_every_candidate_matches():
    spec = TreeSpec.chain(3)
    candidates = torch.tensor([11, 22, 33])
    # root predicts 11; node0 predicts 22; node1 predicts 33; node2 predicts 44.
    node_logits = logits_for(100, [22, 33, 44])
    root_logits = logits_for(100, [11])[0]

    acc = accept_path(spec, candidates, node_logits, root_logits)
    assert acc.path == (0, 1, 2)
    assert acc.accepted_length == 3
    assert acc.bonus_token == 44
    assert acc.tokens == (11, 22, 33, 44)
    assert acc.emitted == 4


def test_rejection_at_the_root_still_emits_the_bonus_token():
    """A fully-rejected step must still advance by one, or speculation could lose."""
    spec = TreeSpec.chain(3)
    candidates = torch.tensor([11, 22, 33])
    acc = accept_path(spec, candidates, logits_for(100, [22, 33, 44]), logits_for(100, [99])[0])
    assert acc.path == ()
    assert acc.accepted_length == 0
    assert acc.tokens == (99,)
    assert acc.emitted == 1


def test_partial_acceptance_stops_at_the_first_mismatch():
    spec = TreeSpec.chain(3)
    candidates = torch.tensor([11, 22, 33])
    # node0 predicts 77, so candidate 22 at depth 2 is wrong.
    acc = accept_path(spec, candidates, logits_for(100, [77, 33, 44]), logits_for(100, [11])[0])
    assert acc.path == (0,)
    assert acc.tokens == (11, 77)
    assert acc.bonus_token == 77


def test_walks_into_the_correct_branch_of_a_tree():
    spec = TreeSpec.from_widths((3, 2))
    # ordered: (0,) (1,) (2,) (0,0) (0,1)
    candidates = torch.tensor([10, 20, 30, 40, 50])
    # root predicts 20 -> the rank-1 sibling, which is a leaf.
    acc = accept_path(spec, candidates, logits_for(100, [0, 88, 0, 0, 0]), logits_for(100, [20])[0])
    assert acc.path == (1,)
    assert acc.bonus_token == 88


def test_ambiguous_siblings_are_rejected_loudly():
    """Two siblings with the same token would make the accepted path undefined."""
    spec = TreeSpec.from_widths((2,))
    candidates = torch.tensor([7, 7])
    with pytest.raises(ValueError, match="malformed"):
        accept_path(spec, candidates, logits_for(100, [1, 1]), logits_for(100, [7])[0])


def test_shape_mismatches_are_caught():
    spec = TreeSpec.chain(3)
    with pytest.raises(ValueError, match="candidate tokens"):
        accept_path(spec, torch.tensor([1, 2]), logits_for(10, [1, 2, 3]), logits_for(10, [1])[0])
    with pytest.raises(ValueError, match="rows of logits"):
        accept_path(spec, torch.tensor([1, 2, 3]), logits_for(10, [1, 2]), logits_for(10, [1])[0])


def test_cache_indices_keep_prefix_and_path_only():
    spec = TreeSpec.from_widths((3, 2))
    keep = cache_indices_after_acceptance(prefix_len=5, spec=spec, path=(0, 3))
    assert keep == [0, 1, 2, 3, 4, 5, 8]


def test_cache_indices_for_empty_path_keep_only_prefix():
    spec = TreeSpec.chain(2)
    assert cache_indices_after_acceptance(4, spec, ()) == [0, 1, 2, 3]
