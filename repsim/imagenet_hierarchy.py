"""WordNet-based navigation of the ImageNet-1k label hierarchy.

ImageNet-1k classes are leaves of the WordNet noun hierarchy. This module maps
class indices to WordNet synsets and, given a target node (e.g. ``"feline"``),
builds a connected chain of hierarchy nodes at which we fit cross-model
transforms: a configurable number of descendant levels below the target, the
target itself, and a configurable number of ancestors above it. The chain spans
a range of granularities (subtree breadths) per target. The inference pool is the
full set of 1000 ImageNet classes (embedded once), so every node simply selects
the leaf classes within its own subtree.

For a confound control, :func:`random_grouping_nodes` builds size-matched
``"random"`` nodes whose classes are drawn at random, ignoring the hierarchy:
comparing a coherent k-class subtree against a random k-class set isolates the
effect of semantic locality from that of breadth (class count) alone.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from timm.data import ImageNetInfo

_N_CLASSES = 1000


@lru_cache(maxsize=1)
def _index_to_wnid() -> tuple[str, ...]:
    """Return the 1000 ImageNet WordNet ids ordered by class index."""
    info = ImageNetInfo()
    return tuple(info.index_to_label_name(i) for i in range(_N_CLASSES))


def wnid_to_synset(wnid: str) -> Synset:
    """Resolve an ImageNet WordNet id (e.g. ``"n02123045"``) to its synset."""
    return wn.synset_from_pos_and_offset("n", int(wnid[1:]))


@lru_cache(maxsize=1)
def _synset_to_index() -> dict[Synset, int]:
    """Map each ImageNet leaf synset to its class index."""
    return {wnid_to_synset(w): i for i, w in enumerate(_index_to_wnid())}


def resolve_node(name: str) -> Synset:
    """Resolve a node name to a noun synset.

    Accepts a fully qualified synset name (``"feline.n.01"``) or a bare lemma
    (``"feline"``), in which case the first noun sense is used.
    """
    if "." in name:
        return wn.synset(name)
    senses = wn.synsets(name, pos=wn.NOUN)
    if not senses:
        raise ValueError(f"No noun synset found for {name!r}")
    return senses[0]


def _descendant_synsets(node: Synset) -> set[Synset]:
    """Return the hyponym closure of ``node`` (including ``node`` itself)."""
    return {node, *node.closure(lambda s: s.hyponyms())}


def leaf_class_indices(node: Synset) -> tuple[int, ...]:
    """Return the sorted ImageNet class indices that are descendants of ``node``."""
    descendants = _descendant_synsets(node)
    s2i = _synset_to_index()
    return tuple(sorted(s2i[s] for s in descendants if s in s2i))


def all_class_indices() -> tuple[int, ...]:
    """Return all 1000 ImageNet class indices (the full inference pool)."""
    return tuple(range(_N_CLASSES))


@lru_cache(maxsize=1)
def _internal_nodes() -> tuple[Synset, ...]:
    """Return every WordNet synset that is a strict ancestor of an ImageNet class."""
    internal: set[Synset] = set()
    for leaf in _synset_to_index():
        internal.update(leaf.closure(lambda s: s.hypernyms()))
    return tuple(internal)


def sample_target_nodes(
    n: int,
    exclude: Sequence[str],
    min_classes: int,
    max_classes: int,
    seed: int,
) -> list[str]:
    """Randomly sample internal hierarchy nodes to use as experiment targets.

    Candidates are internal nodes whose subtree contains between ``min_classes``
    and ``max_classes`` ImageNet leaf classes (bounding their central granularity)
    and that have at least two immediate children carrying ImageNet classes (so
    the descendant/self/ancestor chain is non-trivial).

    Args:
        n: Number of nodes to sample.
        exclude: Synset names to exclude (e.g. explicitly configured targets).
        min_classes: Minimum leaf classes in a candidate's subtree.
        max_classes: Maximum leaf classes in a candidate's subtree.
        seed: Seed for reproducible sampling.

    Returns:
        A list of up to ``n`` synset names (fewer if too few candidates exist).
    """
    excluded = {resolve_node(name).name() for name in exclude}
    candidates = []
    for node in _internal_nodes():
        if node.name() in excluded:
            continue
        if not min_classes <= len(leaf_class_indices(node)) <= max_classes:
            continue
        children_with_classes = sum(
            bool(leaf_class_indices(child)) for child in node.hyponyms()
        )
        if children_with_classes >= 2:
            candidates.append(node.name())
    candidates.sort()
    return random.Random(seed).sample(candidates, min(n, len(candidates)))


def _ancestor_chain(node: Synset, levels: int) -> list[Synset]:
    """Return up to ``levels`` ancestors along the primary hypernym path."""
    chain: list[Synset] = []
    current = node
    for _ in range(levels):
        hypernyms = current.hypernyms()
        if not hypernyms:
            break
        current = hypernyms[0]
        chain.append(current)
    return chain


def _descendant_levels(node: Synset, levels: int) -> list[tuple[Synset, int]]:
    """Return ``(synset, depth)`` for hyponyms up to ``levels`` below ``node``.

    A breadth-first descent that assigns each synset the depth at which it is
    first reached (the hierarchy is a DAG, so a synset may be reachable by
    several paths). Only synsets carrying at least one ImageNet leaf class are
    returned; ``node`` itself is excluded.
    """
    found: list[tuple[Synset, int]] = []
    seen = {node}
    queue: deque[tuple[Synset, int]] = deque([(node, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth == levels:
            continue
        for child in current.hyponyms():
            if child in seen:
                continue
            seen.add(child)
            if leaf_class_indices(child):
                found.append((child, depth + 1))
            queue.append((child, depth + 1))
    return found


@dataclass(frozen=True)
class HierarchyNode:
    """A node at which cross-model transforms are evaluated.

    Attributes:
        key: Unique identifier (synset name for real nodes, e.g.
            ``"feline.n.01"``; a synthetic id for null nodes).
        label: Human-readable lemma, e.g. ``"feline"``.
        relation: One of ``"descendant"``, ``"self"``, ``"ancestor"``, ``"random"``.
        depth: Signed distance from the target node (negative for ancestors,
            ``0`` for the target, positive for descendants). Always ``0`` for
            random control nodes, which have no hierarchical position.
        class_indices: ImageNet class indices belonging to this node.
        grouping: ``"hierarchical"`` for real subtree nodes, ``"random"`` for the
            size-matched random-class control nodes.
    """

    key: str
    label: str
    relation: str
    depth: int
    class_indices: tuple[int, ...] = field()
    grouping: str = "hierarchical"


def build_nodes(
    target_name: str, max_ancestor_levels: int, max_descendant_levels: int
) -> list[HierarchyNode]:
    """Build the chain of evaluation nodes around a target.

    Args:
        target_name: Name of the target hierarchy node (e.g. ``"feline"``).
        max_ancestor_levels: Number of ancestor levels above the target.
        max_descendant_levels: Number of descendant levels below the target.

    Returns:
        A list of :class:`HierarchyNode`, one per distinct synset in the
        descendant -> self -> ancestor chain, each carrying its full set of
        ImageNet leaf-class indices (the inference pool is all 1000 classes).
    """
    target = resolve_node(target_name)

    def node(syn: Synset, relation: str, depth: int) -> HierarchyNode:
        return HierarchyNode(
            key=syn.name(),
            label=syn.lemmas()[0].name(),
            relation=relation,
            depth=depth,
            class_indices=leaf_class_indices(syn),
        )

    nodes = [node(syn, "descendant", depth)
             for syn, depth in _descendant_levels(target, max_descendant_levels)]
    nodes.append(node(target, "self", 0))
    nodes += [node(ancestor, "ancestor", -level)
              for level, ancestor in enumerate(_ancestor_chain(target, max_ancestor_levels), 1)]
    return nodes


def random_grouping_nodes(
    sizes: Sequence[int], n_replicates: int, seed: int
) -> list[HierarchyNode]:
    """Build size-matched random-class control nodes.

    For each distinct breadth ``k`` in ``sizes``, draw ``n_replicates`` random
    ``k``-class subsets of the full 1000-class pool. These nodes match the real
    hierarchy nodes on class count but carry no semantic coherence, so comparing
    them isolates locality from breadth.

    The grouping/relation label is ``"random"`` (not ``"null"``) because pandas'
    ``read_csv`` coerces the literal string ``"null"`` to ``NaN``.

    Args:
        sizes: Class counts to match (duplicates are collapsed).
        n_replicates: Random subsets to draw per distinct size.
        seed: Seed for reproducible sampling.

    Returns:
        A list of ``"random"`` :class:`HierarchyNode` objects.
    """
    pool = all_class_indices()
    rng = random.Random(seed)
    nodes: list[HierarchyNode] = []
    for k in sorted({s for s in sizes if 1 <= s <= len(pool)}):
        for rep in range(n_replicates):
            indices = tuple(sorted(rng.sample(pool, k)))
            nodes.append(HierarchyNode(
                key=f"random_k{k}_r{rep}",
                label=f"random(k={k})",
                relation="random",
                depth=0,
                class_indices=indices,
                grouping="random",
            ))
    return nodes
