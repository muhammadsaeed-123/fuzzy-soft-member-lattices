"""Finite-order audits used by the manuscript.

The key distinction is between (i) closure under operations inherited from an
ambient lattice and (ii) existence of least upper and greatest lower bounds in
the induced order.  Ambient nonclosure proves only that a subset is not an
ambient sublattice; it does not by itself prove that the induced order is not a
lattice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Callable, Generic, Hashable, Iterable, List, Mapping, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class AmbientClosureAudit:
    node_count: int
    unordered_distinct_pairs: int
    join_nonclosure_count: int
    meet_nonclosure_count: int
    join_nonclosure_rate: float
    meet_nonclosure_rate: float
    is_ambient_sublattice: bool
    first_join_witness: str
    first_meet_witness: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class IntrinsicLatticeAudit:
    node_count: int
    unordered_distinct_pairs: int
    pairs_without_internal_join: int
    pairs_without_internal_meet: int
    pairs_with_nonunique_minimal_upper_bounds: int
    pairs_with_nonunique_maximal_lower_bounds: int
    is_intrinsic_lattice: bool
    first_join_witness: str
    first_meet_witness: str

    def to_dict(self) -> dict:
        return asdict(self)


def _pair_text(a: object, b: object) -> str:
    return f"{a!r} ; {b!r}"


def ambient_closure_audit(
    nodes: Sequence[T],
    join_fn: Callable[[T, T], T],
    meet_fn: Callable[[T, T], T],
) -> AmbientClosureAudit:
    """Audit inherited join/meet closure over unordered, distinct pairs.

    Diagonal pairs are omitted because they are automatically closed.  Ordered
    duplicates ``(x,y)`` and ``(y,x)`` are also omitted because lattice join and
    meet are commutative.
    """
    ordered_nodes = list(dict.fromkeys(nodes))
    node_set = set(ordered_nodes)
    pair_count = len(ordered_nodes) * (len(ordered_nodes) - 1) // 2
    jv = mv = 0
    jw = mw = ""
    for a, b in combinations(ordered_nodes, 2):
        if join_fn(a, b) not in node_set:
            jv += 1
            if not jw:
                jw = _pair_text(a, b)
        if meet_fn(a, b) not in node_set:
            mv += 1
            if not mw:
                mw = _pair_text(a, b)
    den = pair_count or 1
    return AmbientClosureAudit(
        node_count=len(ordered_nodes),
        unordered_distinct_pairs=pair_count,
        join_nonclosure_count=jv,
        meet_nonclosure_count=mv,
        join_nonclosure_rate=jv / den,
        meet_nonclosure_rate=mv / den,
        is_ambient_sublattice=(jv == 0 and mv == 0),
        first_join_witness=jw,
        first_meet_witness=mw,
    )


def intrinsic_lattice_audit(
    nodes: Sequence[T],
    leq_fn: Callable[[T, T], bool],
) -> Tuple[IntrinsicLatticeAudit, pd.DataFrame]:
    """Test latticehood in the order induced on ``nodes``.

    For each unordered distinct pair, the function computes all internal upper
    and lower bounds and tests for a *least* upper bound and *greatest* lower
    bound.  It also records whether failure is caused by no bound or by multiple
    incomparable minimal/maximal bounds.  The returned witness table contains
    every failing pair and is intentionally machine-readable.
    """
    ordered_nodes: List[T] = list(dict.fromkeys(nodes))
    n = len(ordered_nodes)
    relation = np.zeros((n, n), dtype=bool)
    for i, a in enumerate(ordered_nodes):
        for j, b in enumerate(ordered_nodes):
            relation[i, j] = bool(leq_fn(a, b))

    no_join = no_meet = multi_upper = multi_lower = 0
    first_join = first_meet = ""
    witness_rows: List[dict] = []

    for i, j in combinations(range(n), 2):
        upper_idx = np.flatnonzero(relation[i] & relation[j])
        lower_idx = np.flatnonzero(relation[:, i] & relation[:, j])

        least_upper: List[int] = []
        if len(upper_idx):
            for u in upper_idx:
                if bool(np.all(relation[u, upper_idx])):
                    least_upper.append(int(u))

        greatest_lower: List[int] = []
        if len(lower_idx):
            for ell in lower_idx:
                if bool(np.all(relation[lower_idx, ell])):
                    greatest_lower.append(int(ell))

        join_ok = len(least_upper) == 1
        meet_ok = len(greatest_lower) == 1
        if join_ok and meet_ok:
            continue

        a, b = ordered_nodes[i], ordered_nodes[j]
        row = {
            "Left": repr(a),
            "Right": repr(b),
            "Upper-bound count": int(len(upper_idx)),
            "Least-upper-bound count": int(len(least_upper)),
            "Lower-bound count": int(len(lower_idx)),
            "Greatest-lower-bound count": int(len(greatest_lower)),
            "Internal join": repr(ordered_nodes[least_upper[0]]) if join_ok else "",
            "Internal meet": repr(ordered_nodes[greatest_lower[0]]) if meet_ok else "",
        }
        witness_rows.append(row)

        if not join_ok:
            no_join += 1
            if len(upper_idx) and len(least_upper) != 1:
                # In a partial order there cannot be two distinct *least* upper
                # bounds, but there can be several incomparable minimal uppers.
                minimal = [
                    int(u)
                    for u in upper_idx
                    if not any(
                        v != u and relation[v, u] and not relation[u, v]
                        for v in upper_idx
                    )
                ]
                if len(minimal) > 1:
                    multi_upper += 1
                row["Minimal upper bounds"] = " | ".join(repr(ordered_nodes[k]) for k in minimal)
            if not first_join:
                first_join = _pair_text(a, b)

        if not meet_ok:
            no_meet += 1
            if len(lower_idx) and len(greatest_lower) != 1:
                maximal = [
                    int(ell)
                    for ell in lower_idx
                    if not any(
                        v != ell and relation[ell, v] and not relation[v, ell]
                        for v in lower_idx
                    )
                ]
                if len(maximal) > 1:
                    multi_lower += 1
                row["Maximal lower bounds"] = " | ".join(repr(ordered_nodes[k]) for k in maximal)
            if not first_meet:
                first_meet = _pair_text(a, b)

    pair_count = n * (n - 1) // 2
    summary = IntrinsicLatticeAudit(
        node_count=n,
        unordered_distinct_pairs=pair_count,
        pairs_without_internal_join=no_join,
        pairs_without_internal_meet=no_meet,
        pairs_with_nonunique_minimal_upper_bounds=multi_upper,
        pairs_with_nonunique_maximal_lower_bounds=multi_lower,
        is_intrinsic_lattice=(no_join == 0 and no_meet == 0),
        first_join_witness=first_join,
        first_meet_witness=first_meet,
    )
    columns = [
        "Left",
        "Right",
        "Upper-bound count",
        "Least-upper-bound count",
        "Minimal upper bounds",
        "Lower-bound count",
        "Greatest-lower-bound count",
        "Maximal lower bounds",
        "Internal join",
        "Internal meet",
    ]
    table = pd.DataFrame(witness_rows)
    for column in columns:
        if column not in table:
            table[column] = ""
    return summary, table[columns]


def classification_label(ambient: AmbientClosureAudit, intrinsic: IntrinsicLatticeAudit) -> str:
    if ambient.is_ambient_sublattice:
        return "ambient sublattice"
    if intrinsic.is_intrinsic_lattice:
        return "intrinsic lattice, not an ambient sublattice"
    return "induced poset that is not an intrinsic lattice"


def audit_summary_table(
    name: str,
    ambient: AmbientClosureAudit,
    intrinsic: IntrinsicLatticeAudit,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Structure": name,
                "Nodes": ambient.node_count,
                "Unordered distinct pairs": ambient.unordered_distinct_pairs,
                "Ambient join nonclosure count": ambient.join_nonclosure_count,
                "Ambient join nonclosure rate": ambient.join_nonclosure_rate,
                "Ambient meet nonclosure count": ambient.meet_nonclosure_count,
                "Ambient meet nonclosure rate": ambient.meet_nonclosure_rate,
                "Pairs without internal join": intrinsic.pairs_without_internal_join,
                "Pairs without internal meet": intrinsic.pairs_without_internal_meet,
                "Intrinsic lattice": intrinsic.is_intrinsic_lattice,
                "Classification": classification_label(ambient, intrinsic),
            }
        ]
    )
