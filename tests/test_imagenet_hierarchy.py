"""Unit tests for ImageNet/WordNet hierarchy navigation."""

from repsim.imagenet_hierarchy import (
    build_nodes,
    leaf_class_indices,
    random_grouping_nodes,
    resolve_node,
    sample_target_nodes,
    wnid_to_synset,
)


def test_tabby_cat_index_maps_to_cat_synset():
    # Class index 281 is "tabby, tabby cat"; its wnid must resolve to a cat synset.
    synset = wnid_to_synset("n02123045")
    assert synset.name().startswith("tabby")
    cat_indices = leaf_class_indices(resolve_node("domestic_cat"))
    assert 281 in cat_indices


def test_feline_contains_known_cat_classes():
    feline_indices = set(leaf_class_indices(resolve_node("feline")))
    # tabby (281), Persian cat (283), lion (291), tiger (292) are all felines.
    assert {281, 283, 291, 292}.issubset(feline_indices)


def test_build_nodes_spans_descendants_self_and_ancestors():
    nodes = build_nodes("feline", max_ancestor_levels=2, max_descendant_levels=2)
    relations = {n.relation for n in nodes}
    assert relations == {"descendant", "self", "ancestor"}

    self_node = next(n for n in nodes if n.relation == "self")
    assert self_node.label == "feline"
    assert self_node.depth == 0

    # Descendants have positive depth and are narrower; ancestors are broader.
    descendants = [n for n in nodes if n.relation == "descendant"]
    ancestors = [n for n in nodes if n.relation == "ancestor"]
    assert all(n.depth > 0 for n in descendants)
    assert all(n.depth < 0 for n in ancestors)
    assert all(len(n.class_indices) < len(self_node.class_indices) for n in descendants)
    assert all(len(n.class_indices) > len(self_node.class_indices) for n in ancestors)


def test_build_nodes_descendants_are_subtree_subsets():
    nodes = build_nodes("feline", max_ancestor_levels=1, max_descendant_levels=2)
    self_node = next(n for n in nodes if n.relation == "self")
    feline_classes = set(self_node.class_indices)
    for node in nodes:
        if node.relation == "descendant":
            assert set(node.class_indices).issubset(feline_classes)


def test_random_grouping_nodes_match_sizes_and_are_seeded():
    nodes = random_grouping_nodes(sizes=[3, 3, 10], n_replicates=2, seed=0)
    # Two distinct sizes x two replicates each.
    assert len(nodes) == 4
    assert {len(n.class_indices) for n in nodes} == {3, 10}
    # Labelled "random" (not "null", which pandas read_csv coerces to NaN).
    assert all(n.grouping == "random" and n.relation == "random" for n in nodes)
    # Reproducible for a fixed seed.
    again = random_grouping_nodes(sizes=[3, 10], n_replicates=2, seed=0)
    assert [n.class_indices for n in nodes] == [n.class_indices for n in again]


def test_sample_target_nodes_is_deterministic_and_bounded():
    kwargs = dict(n=6, exclude=["feline"], min_classes=5, max_classes=40, seed=0)
    first = sample_target_nodes(**kwargs)
    second = sample_target_nodes(**kwargs)
    assert first == second  # reproducible for a fixed seed
    assert len(first) == 6
    # "feline" is excluded; every sample respects the class-count bounds.
    assert "feline.n.01" not in first
    for name in first:
        assert 5 <= len(leaf_class_indices(resolve_node(name))) <= 40


def test_sample_target_nodes_excludes_explicit_targets():
    sampled = sample_target_nodes(
        n=10, exclude=["feline", "snake"], min_classes=3, max_classes=60, seed=1
    )
    assert "feline.n.01" not in sampled
    assert "snake.n.01" not in sampled
