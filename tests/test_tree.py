import pytest
import torch

from specheads.decode.tree import TreeSpec


def test_chain_shape():
    chain = TreeSpec.chain(3)
    assert chain.ordered == ((0,), (0, 0), (0, 0, 0))
    assert chain.parents == (-1, 0, 1)
    assert chain.depth == 3
    assert chain.root_children == (0,)


def test_from_widths_branches_only_the_top_path():
    tree = TreeSpec.from_widths((3, 2))
    assert tree.ordered == ((0,), (1,), (2,), (0, 0), (0, 1))
    # Only rank-0 gets children; ranks 1 and 2 are leaves.
    assert tree.children[1] == () and tree.children[2] == ()
    assert tree.children[0] == (3, 4)


def test_parents_precede_children_in_order():
    """The mask builder relies on this instead of a topological sort."""
    tree = TreeSpec.from_widths((3, 2, 2))
    for node, parent in enumerate(tree.parents):
        assert parent < node


def test_prefix_closure_is_enforced():
    with pytest.raises(ValueError, match="missing its parent"):
        TreeSpec((((0,)), (0, 0, 0)))


def test_rejects_duplicate_and_empty_paths():
    with pytest.raises(ValueError, match="duplicate"):
        TreeSpec(((0,), (0,)))
    with pytest.raises(ValueError, match="implicit root"):
        TreeSpec(((),))


def test_ancestors_are_transitive():
    tree = TreeSpec.from_widths((1, 1, 1))
    assert tree.ancestors == ((), (0,), (0, 1))
    assert tree.path_to(2) == (0, 1, 2)


def test_mask_is_causal_for_a_chain():
    """A chain's candidate mask must be exactly lower-triangular over candidates."""
    chain = TreeSpec.chain(4)
    mask = chain.attention_mask(prefix_len=2)[0, 0]
    assert mask.shape == (4, 6)
    assert mask[:, :2].all()  # prefix fully visible
    candidates = mask[:, 2:]
    assert torch.equal(candidates, torch.tril(torch.ones(4, 4, dtype=torch.bool)))


def test_siblings_cannot_see_each_other():
    """The failure that silently breaks losslessness: cross-branch attention."""
    tree = TreeSpec.from_widths((2, 2))
    mask = tree.attention_mask(prefix_len=1)[0, 0]
    a, b = 0, 1  # two depth-1 siblings
    assert not mask[a, 1 + b]
    assert not mask[b, 1 + a]
    # A child sees its own parent but not its uncle.
    child_of_a = tree.ordered.index((0, 0))
    assert mask[child_of_a, 1 + a]
    assert not mask[child_of_a, 1 + b]


def test_every_node_sees_exactly_its_ancestors_and_itself():
    tree = TreeSpec.from_widths((3, 2, 2))
    prefix = 5
    mask = tree.attention_mask(prefix_len=prefix)[0, 0]
    for node in range(tree.size):
        expected = set(tree.ancestors[node]) | {node}
        seen = {i for i in range(tree.size) if mask[node, prefix + i]}
        assert seen == expected, f"node {node} ({tree.ordered[node]})"
        assert mask[node, :prefix].all()


def test_siblings_share_a_position_id():
    """Competing hypotheses for one slot must get identical RoPE positions."""
    tree = TreeSpec.from_widths((3, 2))
    positions = tree.position_ids(prefix_len=7)[0]
    depth_one = [i for i, p in enumerate(tree.ordered) if len(p) == 1]
    assert len({int(positions[i]) for i in depth_one}) == 1
    assert int(positions[depth_one[0]]) == 7


def test_float_mask_matches_bool_mask():
    tree = TreeSpec.from_widths((2, 2))
    boolean = tree.attention_mask(prefix_len=3)
    additive = tree.attention_mask(prefix_len=3, dtype=torch.float32)
    assert torch.equal(additive == 0.0, boolean)
    assert (additive[~boolean] == torch.finfo(torch.float32).min).all()
