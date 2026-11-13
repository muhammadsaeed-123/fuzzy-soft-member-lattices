from order_audit import ambient_closure_audit, intrinsic_lattice_audit, classification_label


def subset_join(a, b):
    return frozenset(a | b)


def subset_meet(a, b):
    return frozenset(a & b)


def subset_leq(a, b):
    return a <= b


def test_ambient_sublattice_is_intrinsic_lattice():
    nodes = [frozenset(), frozenset({1}), frozenset({1, 2})]
    ambient = ambient_closure_audit(nodes, subset_join, subset_meet)
    intrinsic, witnesses = intrinsic_lattice_audit(nodes, subset_leq)
    assert ambient.is_ambient_sublattice
    assert intrinsic.is_intrinsic_lattice
    assert witnesses.empty
    assert classification_label(ambient, intrinsic) == "ambient sublattice"


def test_intrinsic_lattice_need_not_be_ambient_sublattice():
    # B = {empty, {1}, {2}, {1,2,3}}.  Ambient join({1},{2})={1,2}
    # is absent, but the internal join is {1,2,3}.
    nodes = [
        frozenset(),
        frozenset({1}),
        frozenset({2}),
        frozenset({1, 2, 3}),
    ]
    ambient = ambient_closure_audit(nodes, subset_join, subset_meet)
    intrinsic, witnesses = intrinsic_lattice_audit(nodes, subset_leq)
    assert not ambient.is_ambient_sublattice
    assert ambient.join_nonclosure_count == 1
    assert intrinsic.is_intrinsic_lattice
    assert witnesses.empty
    assert classification_label(ambient, intrinsic) == "intrinsic lattice, not an ambient sublattice"


def test_induced_order_can_fail_to_be_lattice():
    # The pair {1,2}, {1,3} has no upper bound in B.
    nodes = [
        frozenset(),
        frozenset({1}),
        frozenset({2}),
        frozenset({3}),
        frozenset({1, 2}),
        frozenset({1, 3}),
        frozenset({2, 3}),
    ]
    ambient = ambient_closure_audit(nodes, subset_join, subset_meet)
    intrinsic, witnesses = intrinsic_lattice_audit(nodes, subset_leq)
    assert not ambient.is_ambient_sublattice
    assert not intrinsic.is_intrinsic_lattice
    assert intrinsic.pairs_without_internal_join > 0
    assert not witnesses.empty
    assert classification_label(ambient, intrinsic) == "induced poset that is not an intrinsic lattice"


def test_unordered_distinct_pair_denominator():
    nodes = [frozenset(), frozenset({1}), frozenset({2}), frozenset({1, 2})]
    audit = ambient_closure_audit(nodes, subset_join, subset_meet)
    assert audit.unordered_distinct_pairs == 6
