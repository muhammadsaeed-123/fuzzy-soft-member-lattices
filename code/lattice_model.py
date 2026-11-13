from __future__ import annotations

import collections
import itertools
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

Node3 = Tuple[int, int, int]
GammaNode = Tuple[Node3, Node3, Node3]
Coord = Tuple[int, int]


@dataclass(frozen=True)
class ContinuousSpec:
    name: str
    raw_col: str
    orientation: str
    levels: int


CONTINUOUS_SPECS = (
    ContinuousSpec("unit_value", "avg_unit_value", "benefit", 4),
    ContinuousSpec("discount_preservation", "discount_rate", "cost", 4),
    ContinuousSpec("order_scale", "total_qty", "benefit", 4),
    ContinuousSpec("net_revenue", "net_sales", "benefit", 5),
    ContinuousSpec("order_profit", "total_profit", "benefit", 5),
    ContinuousSpec("profit_margin", "profit_margin", "benefit", 4),
)


def order_level_aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item rows to the order decision unit and retain audit-safe fields."""
    df = raw.copy()
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="raise")
    group = df.groupby("Order Id", sort=False)
    order = group.agg(
        order_date=("order_date", "first"),
        market=("Market", "first"),
        department=("Department Name", "first"),
        shipping_mode=("Shipping Mode", "first"),
        delivery_status=("Delivery Status", "first"),
        order_status=("Order Status", "first"),
        transaction_type=("Type", "first"),
        customer_id=("Customer Id", "first"),
        actual_days=("Days for shipping (real)", "first"),
        scheduled_days=("Days for shipment (scheduled)", "first"),
        gross_sales=("Sales", "sum"),
        discount_amount=("Order Item Discount", "sum"),
        net_sales=("Order Item Total", "sum"),
        total_profit=("Order Profit Per Order", "sum"),
        total_qty=("Order Item Quantity", "sum"),
        n_items=("Order Item Id", "count"),
    ).reset_index()
    order["avg_unit_value"] = order["gross_sales"] / order["total_qty"].replace(0, np.nan)
    order["discount_rate"] = order["discount_amount"] / order["gross_sales"].replace(0, np.nan)
    order["profit_margin"] = order["total_profit"] / order["net_sales"].replace(0, np.nan)
    order["reliability"] = order["delivery_status"].isin(["Advance shipping", "Shipping on time"]).astype(int)
    order["status_outcome"] = (
        order["order_status"].isin(["COMPLETE", "CLOSED"]).astype(int)
        - order["order_status"].isin(["SUSPECTED_FRAUD", "CANCELED"]).astype(int)
    )
    if order.isna().any().any():
        bad = order.columns[order.isna().any()].tolist()
        raise ValueError(f"Unexpected order-level missing values: {bad}")
    return order


def field_audit(raw: pd.DataFrame) -> pd.DataFrame:
    """Audit within-order invariance for fields used in the article."""
    fields = [
        "Market", "Customer Segment", "Shipping Mode", "Delivery Status", "Order Status",
        "Type", "Days for shipping (real)", "Days for shipment (scheduled)", "Late_delivery_risk",
        "Customer Id", "order date (DateOrders)", "Department Name", "Order Profit Per Order",
    ]
    g = raw.groupby("Order Id", sort=False)
    rows = []
    norders = g.ngroups
    for col in fields:
        varying = int((g[col].nunique(dropna=False) > 1).sum())
        rows.append({"Field": col, "Orders varying within order": varying, "Percent": 100.0 * varying / norders})
    check = (raw["Sales"] - raw["Order Item Discount"] - raw["Order Item Total"]).abs().max()
    rows.append({"Field": "Sales - discount - net total (max absolute discrepancy)", "Orders varying within order": np.nan, "Percent": float(check)})
    return pd.DataFrame(rows)


def chronological_market_split(order: pd.DataFrame, train_fraction: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Position-based split: first floor(f*n) orders inside each market."""
    tr_idx: List[int] = []
    te_idx: List[int] = []
    rows = []
    for market, g in order.sort_values(["market", "order_date", "Order Id"]).groupby("market", sort=True):
        n = len(g)
        cut = int(math.floor(train_fraction * n))
        tr = g.iloc[:cut]
        te = g.iloc[cut:]
        tr_idx.extend(tr.index.tolist())
        te_idx.extend(te.index.tolist())
        rows.append({
            "Market": market,
            "Orders": n,
            "Training": len(tr),
            "Holdout": len(te),
            "First holdout date": te["order_date"].min(),
            "Last training date": tr["order_date"].max(),
        })
    train = order.loc[tr_idx].sort_values(["order_date", "Order Id"]).copy()
    test = order.loc[te_idx].sort_values(["order_date", "Order Id"]).copy()
    return train, test, pd.DataFrame(rows)


def _strict_edges(values: pd.Series, levels: int) -> np.ndarray:
    edges = values.quantile(np.linspace(0, 1, levels + 1)).to_numpy(float)
    edges[0] = -1e-12
    edges[-1] = 1.0 + 1e-12
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges


def fit_memberships(train0: pd.DataFrame, test0: pd.DataFrame, winsor: float = 0.01):
    train = train0.copy()
    test = test0.copy()
    model: Dict[str, dict] = {}
    rep_mu: Dict[str, np.ndarray] = {}
    rep_raw: Dict[str, np.ndarray] = {}
    ranges: Dict[str, pd.DataFrame] = {}

    for spec in CONTINUOUS_SPECS:
        lo = float(train[spec.raw_col].quantile(winsor))
        hi = float(train[spec.raw_col].quantile(1 - winsor))
        if not hi > lo:
            raise ValueError(f"Degenerate winsor range for {spec.name}")
        for frame in (train, test):
            x = frame[spec.raw_col].clip(lo, hi)
            mu = (x - lo) / (hi - lo)
            if spec.orientation == "cost":
                mu = 1.0 - mu
            frame[f"{spec.name}_mu"] = mu.clip(0.0, 1.0)
        edges = _strict_edges(train[f"{spec.name}_mu"], spec.levels)
        for frame in (train, test):
            frame[f"{spec.name}_level"] = np.digitize(frame[f"{spec.name}_mu"], edges[1:-1], right=True) + 1
        idx = range(1, spec.levels + 1)
        rep_mu[spec.name] = train.groupby(f"{spec.name}_level")[f"{spec.name}_mu"].median().reindex(idx).ffill().bfill().to_numpy(float)
        rep_raw[spec.name] = train.groupby(f"{spec.name}_level")[spec.raw_col].median().reindex(idx).ffill().bfill().to_numpy(float)
        ranges[spec.name] = train.groupby(f"{spec.name}_level")[spec.raw_col].agg(["min", "max"]).reindex(idx)
        model[spec.name] = {"raw_col": spec.raw_col, "orientation": spec.orientation, "levels": spec.levels, "lo": lo, "hi": hi, "mu_edges": edges.tolist()}

    # Operational shipping coordinates. The six-day and four-day maxima are present in DataCo.
    for frame in (train, test):
        frame["actual_speed_mu"] = (1.0 - frame["actual_days"] / 6.0).clip(0, 1)
        frame["actual_speed_level"] = pd.cut(
            frame["actual_days"], bins=[-0.1, 1.5, 3.5, 5.5, 6.1], labels=[4, 3, 2, 1]
        ).astype(int)
        frame["planned_speed_mu"] = (1.0 - frame["scheduled_days"] / 4.0).clip(0, 1)
        frame["planned_speed_level"] = frame["scheduled_days"].map({4: 1, 2: 2, 1: 3, 0: 4}).astype(int)
        frame["reliability_mu"] = frame["reliability"].astype(float)
        frame["reliability_level"] = frame["reliability"] + 1

    rep_mu["actual_speed"] = train.groupby("actual_speed_level")["actual_speed_mu"].median().reindex([1, 2, 3, 4]).ffill().bfill().to_numpy(float)
    rep_raw["actual_speed"] = train.groupby("actual_speed_level")["actual_days"].median().reindex([1, 2, 3, 4]).ffill().bfill().to_numpy(float)
    rep_mu["planned_speed"] = np.array([0.0, 0.5, 0.75, 1.0])
    rep_raw["planned_speed"] = np.array([4.0, 2.0, 1.0, 0.0])
    rep_mu["reliability"] = np.array([0.0, 1.0])
    rep_raw["reliability"] = np.array([0.0, 1.0])
    model["actual_speed"] = {"levels": 4, "rule": "reverse observed actual days into four operational intervals"}
    model["planned_speed"] = {"levels": 4, "rule": "scheduled days 4,2,1,0 map to levels 1,2,3,4"}
    model["reliability"] = {"levels": 2, "rule": "advance/on-time = 1, otherwise 0"}
    return train, test, model, rep_mu, rep_raw, ranges


def add_nodes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["C_node"] = list(zip(out.unit_value_level, out.discount_preservation_level, out.order_scale_level))
    out["O_node"] = list(zip(out.actual_speed_level, out.planned_speed_level, out.reliability_level))
    out["M_node"] = list(zip(out.net_revenue_level, out.order_profit_level, out.profit_margin_level))
    return out


def h42(i: int) -> int:
    return 1 if i <= 2 else 2


def h54(i: int) -> int:
    return min(i, 4)


def h52(i: int) -> int:
    return 1 if i <= 3 else 2


def selected_maps():
    return (
        lambda c: (c[2], h42(c[0])),
        lambda o: (o[1], h42(o[0])),
        lambda m: (h54(m[0]), h52(m[1])),
    )


def alternative_maps() -> Dict[str, Tuple[Callable[[Node3], Coord], Callable[[Node3], Coord], Callable[[Node3], Coord]]]:
    # Prespecified alternatives with semantically named pairings.
    return {
        # Five distinct prespecified maps are reported in the manuscript.
        # The earlier development file also contained a duplicate alias named
        # ``value_volume``; it is removed here so map identity is explicit.
        "selected": selected_maps(),
        "unit_reliability": (
            lambda c: (c[2], h42(c[0])),
            lambda o: (o[1], o[2]),
            lambda m: (h54(m[0]), h42(m[2])),
        ),
        "legacy": (
            lambda c: (c[2], h42(c[1])),
            lambda o: (o[1], o[2]),
            lambda m: (h54(m[0]), h42(m[2])),
        ),
        "actual_service": (
            lambda c: (c[2], h42(c[0])),
            lambda o: (o[1], h42(o[0])),
            lambda m: (h54(m[0]), h42(m[2])),
        ),
        "profit_service": (
            lambda c: (c[2], h42(c[1])),
            lambda o: (o[1], h42(o[0])),
            lambda m: (h54(m[0]), h52(m[1])),
        ),
    }


def all_local_nodes():
    LC = list(itertools.product(range(1, 5), repeat=3))
    LO = list(itertools.product(range(1, 5), range(1, 5), range(1, 3)))
    LM = list(itertools.product(range(1, 6), range(1, 6), range(1, 5)))
    return LC, LO, LM


def build_pullback(maps=None) -> List[GammaNode]:
    if maps is None:
        maps = selected_maps()
    qC, qO, qM = maps
    LC, LO, LM = all_local_nodes()
    fibC: Dict[Coord, list] = collections.defaultdict(list)
    fibO: Dict[Coord, list] = collections.defaultdict(list)
    fibM: Dict[Coord, list] = collections.defaultdict(list)
    for x in LC: fibC[qC(x)].append(x)
    for x in LO: fibO[qO(x)].append(x)
    for x in LM: fibM[qM(x)].append(x)
    out: List[GammaNode] = []
    for z in sorted(set(fibC) & set(fibO) & set(fibM)):
        out.extend((c, o, m) for c in fibC[z] for o in fibO[z] for m in fibM[z])
    return out


def compatible_tuple_series(frame: pd.DataFrame, maps=None) -> pd.Series:
    if maps is None: maps = selected_maps()
    qC, qO, qM = maps
    return pd.Series(
        [((c, o, m) if qC(c) == qO(o) == qM(m) else None) for c, o, m in zip(frame.C_node, frame.O_node, frame.M_node)],
        index=frame.index,
        dtype=object,
    )


def rank_gamma(g: GammaNode) -> int:
    return sum(v - 1 for node in g for v in node)


def leq(a: GammaNode, b: GammaNode) -> bool:
    return all(x <= y for na, nb in zip(a, b) for x, y in zip(na, nb))


def join(a: GammaNode, b: GammaNode) -> GammaNode:
    return tuple(tuple(max(x, y) for x, y in zip(na, nb)) for na, nb in zip(a, b))  # type: ignore


def meet(a: GammaNode, b: GammaNode) -> GammaNode:
    return tuple(tuple(min(x, y) for x, y in zip(na, nb)) for na, nb in zip(a, b))  # type: ignore


def _flat(nodes: Sequence[GammaNode]) -> np.ndarray:
    return np.asarray([[v for local in g for v in local] for g in nodes], dtype=np.int16)


def induced_cover_graph(nodes: Sequence[GammaNode]):
    """Exact transitive reduction of a finite componentwise poset, vectorized by source."""
    nodes = list(nodes)
    n = len(nodes)
    A = _flat(nodes)
    ranks = A.sum(axis=1) - 9
    order = np.argsort(ranks, kind="stable")
    succ: Dict[GammaNode, List[GammaNode]] = {g: [] for g in nodes}
    pred: Dict[GammaNode, List[GammaNode]] = {g: [] for g in nodes}
    for ii in order:
        mask = np.all(A[ii] <= A, axis=1) & np.any(A[ii] < A, axis=1)
        cand = np.where(mask)[0]
        if len(cand) == 0:
            continue
        cand = cand[np.argsort(ranks[cand], kind="stable")]
        minima: List[int] = []
        for jj in cand:
            if not minima or not np.any(np.all(A[minima] <= A[jj], axis=1)):
                minima.append(int(jj))
        g = nodes[ii]
        for jj in minima:
            h = nodes[jj]
            succ[g].append(h)
            pred[h].append(g)
    return succ, pred


def full_product_cover_count(shape: Sequence[int]) -> int:
    return int(sum((shape[j] - 1) * math.prod(shape[k] for k in range(len(shape)) if k != j) for j in range(len(shape))))


def closure_violation_rates(nodes: Sequence[GammaNode]) -> Tuple[float, float, int, int]:
    nodes = list(nodes)
    S = set(nodes)
    jv = 0; mv = 0
    for a in nodes:
        for b in nodes:
            jv += join(a, b) not in S
            mv += meet(a, b) not in S
    den = len(nodes) ** 2
    return jv / den if den else 0.0, mv / den if den else 0.0, jv, mv


def reachable_from_sources(nodes, succ, pred, source: GammaNode | None = None):
    node_set = set(nodes)
    minima = [g for g in nodes if not pred.get(g)]
    if source is None:
        sources = minima
    else:
        sources = [source]
    seen = set(sources); stack = list(sources)
    while stack:
        g = stack.pop()
        for h in succ.get(g, []):
            if h not in seen:
                seen.add(h); stack.append(h)
    return seen


def geom(values: Sequence[float], epsilon: float = 0.05) -> float:
    x = epsilon + (1.0 - epsilon) * np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(x))))


def quality_function(rep_mu: Mapping[str, np.ndarray], operator: str = "geometric", epsilon: float = 0.05):
    def aggregate(vals):
        a = np.asarray(vals, dtype=float)
        if operator == "geometric": return geom(a, epsilon)
        if operator == "arithmetic": return float(np.mean(a))
        if operator == "minimum": return float(np.min(a))
        if operator == "harmonic": return float(len(a) / np.sum(1.0 / np.maximum(a, 1e-9)))
        if operator == "probabilistic_sum": return float(1.0 - np.prod(1.0 - a))
        raise ValueError(operator)
    def q(g: GammaNode) -> float:
        c, o, m = g
        C = [rep_mu["unit_value"][c[0]-1], rep_mu["discount_preservation"][c[1]-1], rep_mu["order_scale"][c[2]-1]]
        O = [rep_mu["actual_speed"][o[0]-1], rep_mu["planned_speed"][o[1]-1], rep_mu["reliability"][o[2]-1]]
        M = [rep_mu["net_revenue"][m[0]-1], rep_mu["order_profit"][m[1]-1], rep_mu["profit_margin"][m[2]-1]]
        return float(np.mean([aggregate(C), aggregate(O), aggregate(M)]))
    return q


def ordinal_quality(g: GammaNode) -> float:
    c, o, m = g
    vals = [(c[0]-1)/3, (c[1]-1)/3, (c[2]-1)/3, (o[0]-1)/3, (o[1]-1)/3, (o[2]-1), (m[0]-1)/4, (m[1]-1)/4, (m[2]-1)/3]
    return float(np.mean(vals))


def joint_counts(frame: pd.DataFrame, maps=None) -> Dict[GammaNode, int]:
    s = compatible_tuple_series(frame, maps)
    return {k: int(v) for k, v in s.dropna().value_counts().items()}


def evidence_function(counts: Mapping[GammaNode, int], nodes: Sequence[GammaNode] | None = None):
    mx = max(counts.values(), default=1)
    den = math.log1p(mx)
    return lambda g: math.log1p(counts.get(g, 0)) / den


def marginal_evidence_function(c_counts, o_counts, m_counts):
    mc = max(c_counts.values(), default=1); mo = max(o_counts.values(), default=1); mm = max(m_counts.values(), default=1)
    def f(g):
        c, o, m = g
        return float(np.mean([
            math.log1p(c_counts.get(c, 0)) / math.log1p(mc),
            math.log1p(o_counts.get(o, 0)) / math.log1p(mo),
            math.log1p(m_counts.get(m, 0)) / math.log1p(mm),
        ]))
    return f


def max_mean_path(nodes: Sequence[GammaNode], succ, pred, utility: Callable[[GammaNode], float], source: GammaNode, targets: Sequence[GammaNode] | None = None):
    nodes = list(nodes)
    if targets is None:
        targets = [g for g in nodes if not succ.get(g)]
    order = sorted(nodes, key=lambda g: (rank_gamma(g), g))
    dp: Dict[Tuple[GammaNode, int], float] = {(source, 1): utility(source)}
    parent: Dict[Tuple[GammaNode, int], Tuple[GammaNode, int]] = {}
    lengths: Dict[GammaNode, List[int]] = collections.defaultdict(list)
    lengths[source].append(1)
    for g in order:
        for ell in list(lengths.get(g, [])):
            base = dp[(g, ell)]
            for h in succ.get(g, []):
                key = (h, ell + 1)
                value = base + utility(h)
                if value > dp.get(key, -np.inf) + 1e-15:
                    if key not in dp: lengths[h].append(ell + 1)
                    dp[key] = value
                    parent[key] = (g, ell)
                elif abs(value - dp.get(key, -np.inf)) <= 1e-15:
                    old = parent.get(key)
                    cand = (g, ell)
                    if old is None or cand < old:
                        parent[key] = cand
    options = []
    for t in targets:
        for ell in lengths.get(t, []):
            options.append((dp[(t, ell)] / ell, t, ell, dp[(t, ell)]))
    if not options:
        raise RuntimeError("No source-to-target path")
    best = max(options, key=lambda x: (x[0], -x[2], x[1]))
    mean, t, ell, total = best
    path = []
    key = (t, ell)
    while True:
        path.append(key[0])
        if key == (source, 1): break
        key = parent[key]
    path.reverse()
    return path, float(mean), float(total)


def enumerate_paths(succ, source: GammaNode, targets: Sequence[GammaNode] | None = None, limit: int | None = None):
    if targets is None:
        targets = [g for g, hs in succ.items() if not hs]
    T = set(targets)
    out = []
    stack = [(source, [source])]
    while stack:
        g, p = stack.pop()
        if g in T:
            out.append(p)
            if limit is not None and len(out) >= limit: break
            continue
        for h in reversed(sorted(succ.get(g, []))):
            stack.append((h, p + [h]))
    return out


def path_mean(path: Sequence[GammaNode], utility: Callable[[GammaNode], float]) -> float:
    return float(np.mean([utility(g) for g in path]))


def node_distance(a: GammaNode, b: GammaNode) -> int:
    return int(sum(abs(x-y) for na, nb in zip(a,b) for x,y in zip(na,nb)))


def near_path_status(frame: pd.DataFrame, path: Sequence[GammaNode], maps=None, radius: int = 1):
    tuples = compatible_tuple_series(frame, maps)
    mask = []
    P = list(path)
    for g in tuples:
        mask.append(g is not None and min(node_distance(g, p) for p in P) <= radius)
    sub = frame.loc[np.asarray(mask, dtype=bool)]
    return float(sub.status_outcome.mean()) if len(sub) else np.nan, int(len(sub)), sub


def near_mask(frame: pd.DataFrame, path: Sequence[GammaNode], maps=None, radius: int = 1) -> np.ndarray:
    tuples = compatible_tuple_series(frame, maps).to_numpy(object)
    P = list(path)
    return np.asarray([g is not None and min(node_distance(g, p) for p in P) <= radius for g in tuples], dtype=bool)


def _safe_spearman_statistic(x: Sequence[float], y: Sequence[float]) -> float:
    """Return a Spearman statistic without constant-input warnings."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if len(xa) < 2 or np.ptp(xa) == 0 or np.ptp(ya) == 0:
        return float("nan")
    return float(stats.spearmanr(xa, ya).statistic)


def cluster_bootstrap_difference(frame: pd.DataFrame, path_a, path_b, reps=2000, seed=20260819, maps=None, radius=1):
    """Fast market-stratified customer-cluster bootstrap for two near-path means."""
    rng = np.random.default_rng(seed)
    ma = near_mask(frame, path_a, maps, radius)
    mb = near_mask(frame, path_b, maps, radius)
    y = frame["status_outcome"].to_numpy(float)
    work = frame[["market", "customer_id"]].copy()
    work["a_count"] = ma.astype(int); work["a_sum"] = y * ma
    work["b_count"] = mb.astype(int); work["b_sum"] = y * mb
    agg = work.groupby(["market", "customer_id"], sort=False)[["a_count","a_sum","b_count","b_sum"]].sum().reset_index()
    blocks = [g[["a_count","a_sum","b_count","b_sum"]].to_numpy(float) for _,g in agg.groupby("market", sort=True)]
    vals = np.empty(reps, dtype=float)
    used = 0
    for _ in range(reps):
        total = np.zeros(4, dtype=float)
        for block in blocks:
            pick = rng.integers(0, len(block), size=len(block))
            total += block[pick].sum(axis=0)
        if total[0] > 0 and total[2] > 0:
            vals[used] = total[1]/total[0] - total[3]/total[2]
            used += 1
    vals = vals[:used]
    point = (
        float(y[ma].mean() - y[mb].mean())
        if ma.any() and mb.any()
        else float("nan")
    )
    interval = (
        np.quantile(vals, [0.025, 0.975]).astype(float)
        if len(vals)
        else np.asarray([np.nan, np.nan], dtype=float)
    )
    return point, interval, vals

def spearman_bootstrap_joint_support(train: pd.DataFrame, test: pd.DataFrame, nodes: Sequence[GammaNode], reps=1000, seed=20260819, maps=None):
    """Market-stratified customer-cluster bootstrap for log joint-support transport."""
    from scipy import sparse
    tc = joint_counts(train, maps)
    x = np.log1p([tc.get(g,0) for g in nodes])
    node_index = {g:i for i,g in enumerate(nodes)}
    tuples = compatible_tuple_series(test, maps).to_numpy(object)
    meta = test[["market","customer_id"]].copy().reset_index(drop=True)
    cluster_index = pd.MultiIndex.from_arrays([meta["market"], meta["customer_id"]])
    cluster_codes, cluster_keys = pd.factorize(cluster_index, sort=False)
    rows=[]; cols=[]; data=[]
    for r,g in enumerate(tuples):
        j=node_index.get(g)
        if j is not None:
            rows.append(int(cluster_codes[r])); cols.append(j); data.append(1)
    M=sparse.csr_matrix((data,(rows,cols)), shape=(len(cluster_keys),len(nodes)), dtype=np.int32)
    markets=np.asarray([k[0] for k in cluster_keys], dtype=object)
    blocks=[np.where(markets==m)[0] for m in sorted(set(markets))]
    base=np.asarray(M.sum(axis=0)).ravel()
    rho=_safe_spearman_statistic(x, np.log1p(base))
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(reps):
        weights=np.zeros(len(cluster_keys),dtype=np.int32)
        for idx in blocks:
            pick=rng.choice(idx,size=len(idx),replace=True)
            weights += np.bincount(pick,minlength=len(cluster_keys)).astype(np.int32)
        counts=np.asarray(weights @ M).ravel()
        r=_safe_spearman_statistic(x, np.log1p(counts))
        if np.isfinite(r): vals.append(r)
    interval = (
        np.quantile(vals, [.025, .975]).astype(float)
        if vals
        else np.asarray([np.nan, np.nan], dtype=float)
    )
    return rho, interval

def jaccard(a: Iterable, b: Iterable) -> float:
    A=set(a); B=set(b)
    return len(A&B)/len(A|B) if A|B else 1.0


def format_gamma(g: GammaNode) -> str:
    return f"{g[0]} | {g[1]} | {g[2]}"
