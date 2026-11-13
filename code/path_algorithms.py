"""Exact path algorithms used by the analysis.

The main routine supports multiple declared sources, source nodes that may have
incoming edges, deterministic tie breaking, and nonnegative or signed transition
costs.  It optimizes the mean of node utility minus edge costs on a finite DAG.
"""
from __future__ import annotations

from collections import defaultdict, deque
from math import inf
from typing import Callable, Dict, Hashable, List, Mapping, MutableMapping, Sequence, Tuple, TypeVar

T = TypeVar("T", bound=Hashable)


def topological_order(nodes: Sequence[T], succ: Mapping[T, Sequence[T]], pred: Mapping[T, Sequence[T]]) -> List[T]:
    xs = list(dict.fromkeys(nodes))
    indegree = {x: len(pred.get(x, ())) for x in xs}
    queue = deque(sorted((x for x in xs if indegree[x] == 0), key=repr))
    order: List[T] = []
    while queue:
        x = queue.popleft()
        order.append(x)
        for y in sorted(succ.get(x, ()), key=repr):
            indegree[y] -= 1
            if indegree[y] == 0:
                queue.append(y)
    if len(order) != len(xs):
        raise ValueError("The directed graph is not acyclic")
    return order


def max_mean_path_multi_source(
    nodes: Sequence[T],
    succ: Mapping[T, Sequence[T]],
    pred: Mapping[T, Sequence[T]],
    utility: Callable[[T], float],
    sources: Sequence[T],
    targets: Sequence[T] | None = None,
    transition_cost: Callable[[T, T], float] | None = None,
    atol: float = 1e-12,
) -> Tuple[List[T], float, Dict[Tuple[T, int], float]]:
    """Return an exact maximum-mean source-to-target path.

    For a path ``p=(v_1,...,v_k)`` the objective is

    ``[sum_i u(v_i) - sum_i c(v_i,v_{i+1})] / k``.

    Every declared source is initialized at length one.  If a source is also
    reachable from another source, both the direct-start and predecessor
    recurrences remain admissible, which corrects the common multiple-source
    initialization error.
    """
    xs = list(dict.fromkeys(nodes))
    xset = set(xs)
    source_set = set(sources)
    if not source_set:
        raise ValueError("At least one source is required")
    if not source_set.issubset(xset):
        raise ValueError("Every source must be a graph node")
    edge_cost = transition_cost or (lambda _a, _b: 0.0)
    order = topological_order(xs, succ, pred)
    target_set = set(targets) if targets is not None else {x for x in xs if not succ.get(x)}
    if not target_set:
        raise ValueError("At least one target is required")

    dp: Dict[Tuple[T, int], float] = {}
    parent: Dict[Tuple[T, int], Tuple[T, int] | None] = {}
    lengths: MutableMapping[T, set[int]] = defaultdict(set)
    for s in sorted(source_set, key=repr):
        dp[(s, 1)] = float(utility(s))
        parent[(s, 1)] = None
        lengths[s].add(1)

    for x in order:
        for ell in sorted(lengths.get(x, ())):
            base = dp[(x, ell)]
            for y in sorted(succ.get(x, ()), key=repr):
                key = (y, ell + 1)
                value = base + float(utility(y)) - float(edge_cost(x, y))
                old = dp.get(key, -inf)
                if value > old + atol:
                    dp[key] = value
                    parent[key] = (x, ell)
                    lengths[y].add(ell + 1)
                elif abs(value - old) <= atol:
                    candidate = (x, ell)
                    current = parent.get(key)
                    if current is None or repr(candidate) < repr(current):
                        parent[key] = candidate

    best_key: Tuple[T, int] | None = None
    best_mean = -inf
    for t in sorted(target_set, key=repr):
        for ell in sorted(lengths.get(t, ())):
            mean = dp[(t, ell)] / ell
            key = (t, ell)
            if mean > best_mean + atol or (
                abs(mean - best_mean) <= atol and (best_key is None or repr(key) < repr(best_key))
            ):
                best_mean = mean
                best_key = key
    if best_key is None:
        raise RuntimeError("No source-to-target path exists")

    path: List[T] = []
    key: Tuple[T, int] | None = best_key
    while key is not None:
        path.append(key[0])
        key = parent[key]
    path.reverse()
    return path, float(best_mean), dp


def enumerate_source_target_paths(
    succ: Mapping[T, Sequence[T]], sources: Sequence[T], targets: Sequence[T]
) -> List[List[T]]:
    target_set = set(targets)
    paths: List[List[T]] = []

    def visit(x: T, path: List[T]) -> None:
        if x in target_set:
            paths.append(path.copy())
        for y in sorted(succ.get(x, ()), key=repr):
            if y in path:
                raise ValueError("Cycle found during path enumeration")
            visit(y, path + [y])

    for source in sorted(set(sources), key=repr):
        visit(source, [source])
    return paths


def path_mean(
    path: Sequence[T], utility: Callable[[T], float], transition_cost: Callable[[T, T], float] | None = None
) -> float:
    if not path:
        raise ValueError("A path cannot be empty")
    cost = transition_cost or (lambda _a, _b: 0.0)
    total = sum(float(utility(x)) for x in path)
    total -= sum(float(cost(a, b)) for a, b in zip(path[:-1], path[1:]))
    return total / len(path)


def top_k_paths(
    succ: Mapping[T, Sequence[T]],
    sources: Sequence[T],
    targets: Sequence[T],
    utility: Callable[[T], float],
    transition_cost: Callable[[T, T], float] | None = None,
    k: int = 10,
) -> List[Tuple[List[T], float]]:
    if k <= 0:
        return []
    scored = [
        (path, path_mean(path, utility, transition_cost))
        for path in enumerate_source_target_paths(succ, sources, targets)
    ]
    scored.sort(key=lambda item: (-item[1], repr(item[0])))
    return scored[:k]


def recurrence_tables(paths: Sequence[Sequence[T]]) -> Tuple[Dict[T, float], Dict[Tuple[T, T], float]]:
    if not paths:
        return {}, {}
    node_counts: Dict[T, int] = defaultdict(int)
    edge_counts: Dict[Tuple[T, T], int] = defaultdict(int)
    for path in paths:
        for node in set(path):
            node_counts[node] += 1
        for edge in set(zip(path[:-1], path[1:])):
            edge_counts[edge] += 1
    n = len(paths)
    return ({x: c / n for x, c in node_counts.items()}, {e: c / n for e, c in edge_counts.items()})


def consensus_path(
    nodes: Sequence[T],
    succ: Mapping[T, Sequence[T]],
    pred: Mapping[T, Sequence[T]],
    sources: Sequence[T],
    targets: Sequence[T],
    node_frequency: Mapping[T, float],
    edge_frequency: Mapping[Tuple[T, T], float],
    edge_weight: float = 0.5,
) -> Tuple[List[T], float]:
    utility = lambda x: float(node_frequency.get(x, 0.0))
    # Negative cost is a recurrence bonus; this is an audit consensus, not a policy objective.
    cost = lambda a, b: -edge_weight * float(edge_frequency.get((a, b), 0.0))
    path, score, _ = max_mean_path_multi_source(nodes, succ, pred, utility, sources, targets, cost)
    return path, score


def count_source_target_paths(
    nodes: Sequence[T],
    succ: Mapping[T, Sequence[T]],
    pred: Mapping[T, Sequence[T]],
    sources: Sequence[T],
    targets: Sequence[T],
) -> int:
    """Count source-to-target paths in a DAG without enumerating them."""
    order = topological_order(nodes, succ, pred)
    source_set = set(sources)
    counts: Dict[T, int] = {x: int(x in source_set) for x in nodes}
    for x in order:
        for y in succ.get(x, ()):
            counts[y] = counts.get(y, 0) + counts.get(x, 0)
    return int(sum(counts.get(t, 0) for t in set(targets)))


def top_k_mean_paths_dag(
    nodes: Sequence[T],
    succ: Mapping[T, Sequence[T]],
    pred: Mapping[T, Sequence[T]],
    utility: Callable[[T], float],
    sources: Sequence[T],
    targets: Sequence[T],
    transition_cost: Callable[[T, T], float] | None = None,
    k: int = 10,
) -> List[Tuple[List[T], float]]:
    """Exact top-k mean paths on a DAG using length-indexed k-best sums.

    For a fixed endpoint and length, only the k largest additive scores can
    contribute to the global top k. Keeping those states is therefore exact
    while avoiding exhaustive path enumeration.
    """
    if k <= 0:
        return []
    xs = list(dict.fromkeys(nodes))
    edge_cost = transition_cost or (lambda _a, _b: 0.0)
    order = topological_order(xs, succ, pred)
    states: Dict[Tuple[T, int], List[Tuple[float, Tuple[T, ...]]]] = {}
    for s in sorted(set(sources), key=repr):
        states[(s, 1)] = [(float(utility(s)), (s,))]
    for x in order:
        current_keys = sorted((key for key in states if key[0] == x), key=lambda z: z[1])
        for key in current_keys:
            ell = key[1]
            for additive, path in states[key]:
                for y in sorted(succ.get(x, ()), key=repr):
                    new_key = (y, ell + 1)
                    candidate = (
                        additive + float(utility(y)) - float(edge_cost(x, y)),
                        path + (y,),
                    )
                    pool = states.setdefault(new_key, [])
                    pool.append(candidate)
                    pool.sort(key=lambda item: (-item[0], repr(item[1])))
                    if len(pool) > k:
                        del pool[k:]
    candidates: List[Tuple[List[T], float]] = []
    target_set = set(targets)
    for (node, ell), values in states.items():
        if node not in target_set:
            continue
        for additive, path in values:
            candidates.append((list(path), additive / ell))
    candidates.sort(key=lambda item: (-item[1], repr(item[0])))
    unique: List[Tuple[List[T], float]] = []
    seen = set()
    for path, score in candidates:
        signature = tuple(path)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append((path, score))
        if len(unique) == k:
            break
    return unique
