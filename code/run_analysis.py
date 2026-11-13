from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from lattice_model import (
    GammaNode, add_nodes, alternative_maps, build_pullback, chronological_market_split,
    closure_violation_rates, cluster_bootstrap_difference, compatible_tuple_series,
    evidence_function, field_audit, fit_memberships, format_gamma, full_product_cover_count,
    induced_cover_graph, jaccard, joint_counts, leq, marginal_evidence_function,
    max_mean_path, near_mask, near_path_status, node_distance, ordinal_quality,
    order_level_aggregate, path_mean, quality_function, rank_gamma, selected_maps,
    spearman_bootstrap_joint_support, enumerate_paths, join, meet,
)
from order_audit import (
    AmbientClosureAudit, IntrinsicLatticeAudit, ambient_closure_audit,
    intrinsic_lattice_audit, audit_summary_table,
)
from validation_tools import (
    cluster_bootstrap_distance_association, compatible_distance_frame, customer_overlap_summary,
    deterministic_dependency_audit, distance_status_association,
    evidence_on_frozen_training_scale, fixed_k_profile, market_distance_summary,
    radius_coverage_profile, status_distribution,
)


def ensure_dirs(root: Path):
    t = root / "outputs" / "tables"
    f = root / "outputs" / "figures"
    t.mkdir(parents=True, exist_ok=True)
    f.mkdir(parents=True, exist_ok=True)
    return t, f


def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, float_format="%.17g")


def save_current_figure(directory: Path, stem: str, *, dpi: int = 300) -> None:
    """Save the current Matplotlib figure as vector PDF and high-resolution PNG."""
    plt.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(directory / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close()


def edge_table(succ) -> pd.DataFrame:
    rows = []
    for source in sorted(succ, key=lambda g: (rank_gamma(g), g)):
        for target in sorted(succ.get(source, []), key=lambda g: (rank_gamma(g), g)):
            rows.append({
                "Source": format_gamma(source),
                "Target": format_gamma(target),
                "Source rank": rank_gamma(source),
                "Target rank": rank_gamma(target),
                "Rank increment": rank_gamma(target) - rank_gamma(source),
            })
    return pd.DataFrame(rows, columns=["Source", "Target", "Source rank", "Target rank", "Rank increment"])


def node_support_table(nodes, train_counts, holdout_counts, quality, evidence, utility) -> pd.DataFrame:
    rows = []
    for node in sorted(nodes, key=lambda g: (rank_gamma(g), g)):
        rows.append({
            "Node": format_gamma(node),
            "Rank": rank_gamma(node),
            "Commercial": str(node[0]),
            "Logistics": str(node[1]),
            "Market": str(node[2]),
            "Joint training support": int(train_counts.get(node, 0)),
            "Joint holdout support": int(holdout_counts.get(node, 0)),
            "Quality": quality(node),
            "Evidence": evidence(node),
            "Node utility": utility(node),
        })
    return pd.DataFrame(rows)


def common_exact_rate(frame: pd.DataFrame, maps) -> float:
    return float(compatible_tuple_series(frame, maps).notna().mean())


def map_audit(train, test):
    rows=[]
    for name,maps in alternative_maps().items():
        rows.append({"Map":name,"Training exact coordination rate":common_exact_rate(train,maps),"Holdout exact coordination rate":common_exact_rate(test,maps)})
    return pd.DataFrame(rows)


def source_and_reachable(nodes, counts):
    succ,pred=induced_cover_graph(nodes)
    minima=[g for g in nodes if not pred.get(g)]
    source=max(minima,key=lambda g:(counts.get(g,0),-rank_gamma(g),g))
    seen={source}; stack=[source]
    while stack:
        g=stack.pop()
        for h in succ.get(g,[]):
            if h not in seen:
                seen.add(h);stack.append(h)
    rnodes=[g for g in nodes if g in seen]
    rsucc={g:[h for h in succ.get(g,[]) if h in seen] for g in rnodes}
    rpred={g:[h for h in pred.get(g,[]) if h in seen] for g in rnodes}
    return source,rnodes,rsucc,rpred


def raw_path_values(path, rep_raw):
    names=["unit_value","discount_preservation","order_scale","actual_speed","planned_speed","reliability","net_revenue","order_profit","profit_margin"]
    rows=[]
    for i,g in enumerate(path,1):
        c,o,m=g
        vals=[rep_raw["unit_value"][c[0]-1],rep_raw["discount_preservation"][c[1]-1],rep_raw["order_scale"][c[2]-1],rep_raw["actual_speed"][o[0]-1],rep_raw["planned_speed"][o[1]-1],rep_raw["reliability"][o[2]-1],rep_raw["net_revenue"][m[0]-1],rep_raw["order_profit"][m[1]-1],rep_raw["profit_margin"][m[2]-1]]
        row={"Stage":i}
        row.update(dict(zip(names,vals)))
        rows.append(row)
    return pd.DataFrame(rows)




def representative_level_table(train: pd.DataFrame, rep_mu: Dict[str, np.ndarray], rep_raw: Dict[str, np.ndarray], ranges: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one row for every fitted representative membership level."""
    meta = {
        "unit_value": ("Commercial", "Average unit value (USD)", "avg_unit_value"),
        "discount_preservation": ("Commercial", "Discount rate", "discount_rate"),
        "order_scale": ("Commercial", "Total item quantity", "total_qty"),
        "actual_speed": ("Logistics", "Actual shipping days", "actual_days"),
        "planned_speed": ("Logistics", "Scheduled shipping days", "scheduled_days"),
        "reliability": ("Logistics", "On-time/advance indicator", "reliability"),
        "net_revenue": ("Market", "Net order sales (USD)", "net_sales"),
        "order_profit": ("Market", "Total order profit (USD)", "total_profit"),
        "profit_margin": ("Market", "Profit margin", "profit_margin"),
    }
    rows = []
    for name, (layer, label, raw_col) in meta.items():
        levels = len(rep_mu[name])
        for level in range(1, levels + 1):
            group = train.loc[train[f"{name}_level"] == level, raw_col]
            rows.append({
                "Layer": layer,
                "Attribute": label,
                "Level": level,
                "Representative membership": float(rep_mu[name][level - 1]),
                "Representative raw value": float(rep_raw[name][level - 1]),
                "Observed training minimum": float(group.min()),
                "Observed training maximum": float(group.max()),
                "Training orders": int(group.shape[0]),
            })
    return pd.DataFrame(rows)


def descriptive_statistics(order: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for the nine order-level modeled variables."""
    cols = ["avg_unit_value", "discount_rate", "total_qty", "actual_days", "scheduled_days", "reliability", "net_sales", "total_profit", "profit_margin"]
    rows = []
    for col in cols:
        s = order[col].astype(float)
        rows.append({
            "Attribute": col, "N": int(s.count()), "Mean": float(s.mean()),
            "SD": float(s.std(ddof=1)), "Min": float(s.min()),
            "Q1": float(s.quantile(.25)), "Median": float(s.median()),
            "Q3": float(s.quantile(.75)), "Max": float(s.max()),
        })
    return pd.DataFrame(rows)


def inner_benchmark_grid(order: pd.DataFrame, maps) -> pd.DataFrame:
    """Evaluate the transparent n_min/beta grid on the middle chronological window.

    The final article treats n_min=3 and beta=0.15 as benchmark settings. This
    table documents the inner-window sensitivity rather than silently presenting
    them as universally optimal hyperparameters.
    """
    tr_idx, va_idx = [], []
    for _, group in order.sort_values(["market", "order_date", "Order Id"]).groupby("market", sort=True):
        n = len(group)
        a, b = int(math.floor(.6 * n)), int(math.floor(.8 * n))
        tr_idx.extend(group.iloc[:a].index.tolist())
        va_idx.extend(group.iloc[a:b].index.tolist())
    tr0, va0 = order.loc[tr_idx].copy(), order.loc[va_idx].copy()
    tr, va, _, rep_mu, _, _ = fit_memberships(tr0, va0)
    tr, va = add_nodes(tr), add_nodes(va)
    full = build_pullback(maps)
    train_joint = joint_counts(tr, maps)
    hold_joint = joint_counts(va, maps)
    q = quality_function(rep_mu, "geometric", .05)
    rows = []
    for nmin in [1, 2, 3, 5, 10]:
        nodes = [g for g in full if train_joint.get(g, 0) >= nmin]
        if not nodes:
            continue
        source, nodes, succ, pred = source_and_reachable(nodes, train_joint)
        ev = evidence_function(train_joint)
        hev = evidence_on_frozen_training_scale(hold_joint, max(train_joint.values(), default=1))
        for beta in np.round(np.arange(0, .251, .05), 2):
            b = float(beta)
            u = lambda g, b=b: (1 - b) * q(g) + b * ev(g)
            hu = lambda g, b=b: (1 - b) * q(g) + b * hev(g)
            path, score, _ = max_mean_path(nodes, succ, pred, u, source)
            status, nnear, _ = near_path_status(va, path, maps)
            rows.append({
                "n_min": nmin, "beta": b, "Graph nodes": len(nodes),
                "Path nodes": len(path), "Inner near-path orders": nnear,
                "Inner status outcome": status, "Inner common utility": path_mean(path, hu),
                "Training utility": score,
            })
    return pd.DataFrame(rows)


def rolling_origin(order, maps, nmin=3, beta=.15):
    rows=[]
    # Global windows are generated market-wise to preserve each market's chronology.
    for train_end,test_end in [(0.4,0.6),(0.6,0.8),(0.8,1.0)]:
        tr_idx=[];te_idx=[]
        for _,g in order.sort_values(["market","order_date","Order Id"]).groupby("market"):
            n=len(g);a=int(math.floor(train_end*n));b=int(math.floor(test_end*n))
            tr_idx+=g.iloc[:a].index.tolist();te_idx+=g.iloc[a:b].index.tolist()
        tr0=order.loc[tr_idx].copy();te0=order.loc[te_idx].copy()
        tr,te,_,rep_mu,_,_=fit_memberships(tr0,te0);tr=add_nodes(tr);te=add_nodes(te)
        full=build_pullback(maps);jc=joint_counts(tr,maps)
        nodes=[g for g in full if jc.get(g,0)>=nmin]
        if not nodes: continue
        source,nodes,succ,pred=source_and_reachable(nodes,jc)
        q=quality_function(rep_mu);ev=evidence_function(jc);u=lambda g:(1-beta)*q(g)+beta*ev(g)
        p,_,_=max_mean_path(nodes,succ,pred,u,source)
        pq,_,_=max_mean_path(nodes,succ,pred,q,source)
        sf,nf,_=near_path_status(te,p,maps);sq,nq,_=near_path_status(te,pq,maps)
        rows.append({"Training fraction":f"0-{train_end:.1f}","Test fraction":f"{train_end:.1f}-{test_end:.1f}","Full outcome":sf,"Quality-only outcome":sq,"Difference":sf-sq,"Full near orders":nf,"Quality near orders":nq,"Graph nodes":len(nodes),"Path nodes":len(p)})
    return pd.DataFrame(rows)


def fit_one_bootstrap(boot0, maps, nmin, beta):
    tr,_,_,rep_mu,_,_=fit_memberships(boot0,boot0)
    tr=add_nodes(tr);jc=joint_counts(tr,maps);full=build_pullback(maps)
    nodes=[g for g in full if jc.get(g,0)>=nmin]
    if not nodes: return None
    source,nodes,succ,pred=source_and_reachable(nodes,jc)
    q=quality_function(rep_mu);ev=evidence_function(jc);u=lambda g:(1-beta)*q(g)+beta*ev(g)
    p,score,_=max_mean_path(nodes,succ,pred,u,source)
    return p,score,len(nodes),sum(len(x) for x in succ.values())


def full_refit_bootstrap(train0, baseline_path, maps, nmin, beta, reps, seed):
    rng=np.random.default_rng(seed)
    cluster_blocks=[]
    for market,g in train0.groupby("market",sort=True):
        clusters=[h.index.to_numpy() for _,h in g.groupby("customer_id",sort=False)]
        cluster_blocks.append(clusters)
    base_nodes=set(baseline_path);base_edges=set(zip(baseline_path[:-1],baseline_path[1:]))
    rows=[]
    for b in range(reps):
        idx=[]
        for clusters in cluster_blocks:
            pick=rng.integers(0,len(clusters),size=len(clusters))
            for j in pick: idx.extend(clusters[int(j)].tolist())
        boot=train0.loc[idx].reset_index(drop=True)
        error = ""
        try:
            result=fit_one_bootstrap(boot,maps,nmin,beta)
        except Exception as exc:
            result=None
            error=f"{type(exc).__name__}: {exc}"
        if result is None:
            rows.append({"Replicate": b + 1, "Path length": np.nan, "Training score": np.nan,
                         "Feasible nodes": 0, "Cover edges": 0, "Node Jaccard": 0,
                         "Edge Jaccard": 0, "Exact baseline path": False,
                         "Node sequence": "", "Edge sequence": "",
                         "Error": error or "No feasible refit"})
            continue
        p,score,nn,ne=result
        rows.append({
            "Replicate": b + 1,
            "Path length": len(p),
            "Training score": score,
            "Feasible nodes": nn,
            "Cover edges": ne,
            "Node Jaccard": jaccard(base_nodes, p),
            "Edge Jaccard": jaccard(base_edges, zip(p[:-1], p[1:])),
            "Exact baseline path": p == list(baseline_path),
            "Node sequence": " || ".join(format_gamma(g) for g in p),
            "Edge sequence": " || ".join(f"{format_gamma(a)} -> {format_gamma(c)}" for a, c in zip(p[:-1], p[1:])),
            "Error": "",
        })
        if (b+1)%25==0: print(f"full-refit {b+1}/{reps}",flush=True)
    return pd.DataFrame(rows)


def full_refit_frequency_tables(full_refit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert replicate path strings into recurrence-frequency tables."""
    valid = full_refit.loc[full_refit["Node sequence"].astype(str).str.len() > 0]
    node_counts: Dict[str, int] = {}
    edge_counts: Dict[str, int] = {}
    for sequence in valid["Node sequence"]:
        for node in str(sequence).split(" || "):
            node_counts[node] = node_counts.get(node, 0) + 1
    for sequence in valid["Edge sequence"]:
        for edge in str(sequence).split(" || "):
            if edge:
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
    denominator = max(len(valid), 1)
    node_table = pd.DataFrame(
        [
            {"Node": node, "Replicates containing node": count, "Inclusion frequency": count / denominator}
            for node, count in sorted(node_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
    )
    edge_table_ = pd.DataFrame(
        [
            {"Edge": edge, "Replicates containing edge": count, "Inclusion frequency": count / denominator}
            for edge, count in sorted(edge_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
    )
    return node_table, edge_table_


def downstream_map_audit(train, test, rep_mu, nmin: int, beta: float) -> pd.DataFrame:
    """Run the complete graph/path/coverage audit under every declared map."""
    rows = []
    q = quality_function(rep_mu, "geometric", .05)
    for name, maps in alternative_maps().items():
        full = build_pullback(maps)
        train_joint = joint_counts(train, maps)
        holdout_joint = joint_counts(test, maps)
        supported = [node for node in full if train_joint.get(node, 0) >= nmin]
        if not supported:
            rows.append({"Map": name, "Error": "No supported node"})
            continue
        source, reachable, succ, pred = source_and_reachable(supported, train_joint)
        ev = evidence_function(train_joint)
        hev = evidence_on_frozen_training_scale(holdout_joint, max(train_joint.values(), default=1))
        utility = lambda node: (1 - beta) * q(node) + beta * ev(node)
        holdout_utility = lambda node: (1 - beta) * q(node) + beta * hev(node)
        path, score, _ = max_mean_path(reachable, succ, pred, utility, source)
        outcome, near_n, _ = near_path_status(test, path, maps)
        rows.append(
            {
                "Map": name,
                "Algebraic pullback nodes": len(full),
                "Training exact coordination rate": common_exact_rate(train, maps),
                "Holdout exact coordination rate": common_exact_rate(test, maps),
                "Supported reachable nodes": len(reachable),
                "Induced covers": sum(len(values) for values in succ.values()),
                "Path nodes": len(path),
                "Training mean utility": score,
                "Frozen-scale holdout mean utility": path_mean(path, holdout_utility),
                "Radius-one holdout orders": near_n,
                "Radius-one share of complete holdout": near_n / len(test),
                "Mean signed order-status score": outcome,
                "Path": " || ".join(format_gamma(node) for node in path),
                "Error": "",
            }
        )
    return pd.DataFrame(rows)


def source_terminal_sensitivity(nodes, succ, pred, counts, utility) -> pd.DataFrame:
    """Audit all minimal sources and source-reachable maximal targets."""
    rows = []
    minima = [node for node in nodes if not pred.get(node)]
    for source in sorted(minima, key=lambda g: (rank_gamma(g), g)):
        seen = {source}
        stack = [source]
        while stack:
            current = stack.pop()
            for nxt in succ.get(current, []):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        reachable = [node for node in nodes if node in seen]
        rsucc = {node: [nxt for nxt in succ.get(node, []) if nxt in seen] for node in reachable}
        rpred = {node: [prev for prev in pred.get(node, []) if prev in seen] for node in reachable}
        targets = [node for node in reachable if not rsucc.get(node)]
        for target in targets:
            try:
                path, score, _ = max_mean_path(reachable, rsucc, rpred, utility, source, [target])
            except RuntimeError:
                continue
            rows.append(
                {
                    "Source": format_gamma(source),
                    "Source support": int(counts.get(source, 0)),
                    "Target": format_gamma(target),
                    "Reachable nodes": len(reachable),
                    "Path nodes": len(path),
                    "Mean utility": score,
                    "Path": " || ".join(format_gamma(node) for node in path),
                }
            )
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(description="Reproduce the DataCo fuzzy-soft lattice analysis")
    ap.add_argument("--data",type=Path,default=Path(__file__).resolve().parents[1]/"data"/"DataCoSupplyChainDataset.csv")
    ap.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument("--cluster-bootstrap-reps",type=int,default=2000)
    ap.add_argument("--support-bootstrap-reps",type=int,default=1000)
    ap.add_argument("--full-refit-reps",type=int,default=500)
    ap.add_argument("--seed",type=int,default=20260819)
    args=ap.parse_args();tdir,fdir=ensure_dirs(args.out)

    raw=pd.read_csv(args.data,encoding="latin1",low_memory=False)
    audit=field_audit(raw)
    order=order_level_aggregate(raw)
    train0,test0,cutoffs=chronological_market_split(order,.8)
    train,test,transform,rep_mu,rep_raw,ranges=fit_memberships(train0,test0)
    train=add_nodes(train);test=add_nodes(test)
    maps=selected_maps();full=build_pullback(maps)
    full_succ,_=induced_cover_graph(full)
    train_joint=joint_counts(train,maps);hold_joint=joint_counts(test,maps)
    c_counts=train.C_node.value_counts().to_dict();o_counts=train.O_node.value_counts().to_dict();m_counts=train.M_node.value_counts().to_dict()

    nmin=3;beta=.15
    marginal_supported=[g for g in full if c_counts.get(g[0],0)>=nmin and o_counts.get(g[1],0)>=nmin and m_counts.get(g[2],0)>=nmin]
    joint_supported=[g for g in full if train_joint.get(g,0)>=nmin]
    source,joint,succ,pred=source_and_reachable(joint_supported,train_joint)
    msource,marginal,msucc,mpred=source_and_reachable(marginal_supported,train_joint)

    q=quality_function(rep_mu,"geometric",.05);ev=evidence_function(train_joint);hev=evidence_on_frozen_training_scale(hold_joint, max(train_joint.values(), default=1))
    u=lambda g:(1-beta)*q(g)+beta*ev(g);hu=lambda g:(1-beta)*q(g)+beta*hev(g)
    path,train_score,_=max_mean_path(joint,succ,pred,u,source)
    qpath,_,_=max_mean_path(joint,succ,pred,q,source)
    opath,_,_=max_mean_path(joint,succ,pred,ordinal_quality,source)
    mev=marginal_evidence_function(c_counts,o_counts,m_counts);mu=lambda g:(1-beta)*q(g)+beta*mev(g)
    mpath,_,_=max_mean_path(marginal,msucc,mpred,mu,msource)

    all_paths=enumerate_paths(succ,source)
    path_scores=np.asarray([path_mean(p,u) for p in all_paths])
    order_ix=np.argsort(path_scores)[::-1]
    best=float(path_scores[order_ix[0]]);second=float(path_scores[order_ix[1]])
    near1=int((path_scores>=.99*best).sum())

    # Structural audits. Ambient closure and intrinsic latticehood are distinct.
    joint_ambient = ambient_closure_audit(joint, join, meet)
    joint_intrinsic, joint_intrinsic_witnesses = intrinsic_lattice_audit(joint, leq)
    marginal_ambient = ambient_closure_audit(marginal, join, meet)
    marginal_intrinsic, marginal_intrinsic_witnesses = intrinsic_lattice_audit(marginal, leq)
    full_ambient = AmbientClosureAudit(
        node_count=len(full), unordered_distinct_pairs=len(full)*(len(full)-1)//2,
        join_nonclosure_count=0, meet_nonclosure_count=0,
        join_nonclosure_rate=0.0, meet_nonclosure_rate=0.0,
        is_ambient_sublattice=True, first_join_witness="", first_meet_witness="",
    )
    full_intrinsic = IntrinsicLatticeAudit(
        node_count=len(full), unordered_distinct_pairs=len(full)*(len(full)-1)//2,
        pairs_without_internal_join=0, pairs_without_internal_meet=0,
        pairs_with_nonunique_minimal_upper_bounds=0,
        pairs_with_nonunique_maximal_lower_bounds=0,
        is_intrinsic_lattice=True, first_join_witness="", first_meet_witness="",
    )
    closure = pd.concat([
        audit_summary_table("Full algebraic pullback", full_ambient, full_intrinsic),
        audit_summary_table("Marginal-support induced order", marginal_ambient, marginal_intrinsic),
        audit_summary_table("Joint-support induced order", joint_ambient, joint_intrinsic),
    ], ignore_index=True)
    closure["Covers"] = [sum(len(v) for v in full_succ.values()), sum(len(v) for v in msucc.values()), sum(len(v) for v in succ.values())]

    # Exploratory holdout validation; comparative claims use common-protocol diagnostics below.
    methods={"Full joint-support model":path,"Quality-only":qpath,"Ordinal-score ablation":opath,"Marginal-support construction":mpath}
    validation=[]
    for name,p in methods.items():
        status,nnear,sub=near_path_status(test,p,maps)
        validation.append({
            "Method":name,
            "Graph nodes":len(joint) if name!="Marginal-support construction" else len(marginal),
            "Path nodes":len(p),
            "Training utility on baseline scale":path_mean(p,u),
            "Frozen-scale holdout utility":path_mean(p,hu),
            "Path nodes with positive training support":sum(train_joint.get(g,0)>0 for g in p),
            "Radius-one holdout orders":nnear,
            "Radius-one share of complete holdout":nnear/len(test),
            "Radius-one share of exactly coordinated holdout":nnear/max(int(compatible_tuple_series(test,maps).notna().sum()),1),
            "Mean signed order-status score":status,
            "Positive-coded orders":int((sub.status_outcome>0).sum()),
            "Neutral-coded orders":int((sub.status_outcome==0).sum()),
            "Negative-coded orders":int((sub.status_outcome<0).sum()),
            "Estimand note":"method-specific radius-one cohort; not a common-cohort superiority estimate",
        })
    validation_df=pd.DataFrame(validation)

    # Common-protocol diagnostics evaluate every path on the same exactly coordinated holdout cohort.
    distance_frame=compatible_distance_frame(test,methods,maps)
    distance_association=distance_status_association(distance_frame)
    market_distance = market_distance_summary(distance_frame)
    distance_bootstrap, distance_bootstrap_replicates = cluster_bootstrap_distance_association(
        distance_frame, args.cluster_bootstrap_reps, args.seed + 101
    )
    radius_profile=radius_coverage_profile(distance_frame,len(test),(0,1,2,3))
    fixed_k=fixed_k_profile(distance_frame,(.05,.10,.20))

    bootstrap_rows=[];bootstrap_arrays={}
    for name,p in [("Quality-only",qpath),("Ordinal-score ablation",opath),("Marginal-support construction",mpath)]:
        point,ci,arr=cluster_bootstrap_difference(test,path,p,args.cluster_bootstrap_reps,args.seed+len(bootstrap_rows),maps)
        bootstrap_rows.append({
            "Comparison":f"Full - {name}","Point difference":point,
            "CI low":ci[0],"CI high":ci[1],"Replicates":len(arr),
            "Interpretation":"exploratory contrast of method-specific radius-one cohorts",
        })
        bootstrap_arrays[name]=arr
    boot_df=pd.DataFrame(bootstrap_rows)
    bootstrap_replicates = pd.DataFrame([
        {"Comparison": "Full - " + name, "Replicate": i+1, "Difference": float(value)}
        for name, values in bootstrap_arrays.items() for i, value in enumerate(values)
    ])

    status_tables=pd.concat([
        status_distribution(order,"All orders"),
        status_distribution(train,"Training orders"),
        status_distribution(test,"Chronological holdout"),
        status_distribution(test.loc[compatible_tuple_series(test,maps).notna()],"Exactly coordinated holdout"),
    ],ignore_index=True)
    overlap_df=customer_overlap_summary(train,test)
    dependency_df=deterministic_dependency_audit(order)
    map_downstream_df=downstream_map_audit(train,test,rep_mu,nmin,beta)
    source_sensitivity_df=source_terminal_sensitivity(joint,succ,pred,train_joint,u)
    utility_rows=[]
    for b in np.round(np.arange(0,.251,.05),2):
        ub=lambda g,b=float(b):(1-b)*q(g)+b*ev(g)
        pp,ss,_=max_mean_path(joint,succ,pred,ub,source)
        st,nn,_=near_path_status(test,pp,maps)
        utility_rows.append({"beta":float(b),"Path nodes":len(pp),"Node Jaccard with beta=0.15":jaccard(path,pp),"Training mean utility":ss,"Radius-one holdout orders":nn,"Mean signed order-status score":st,"Path":" || ".join(format_gamma(g) for g in pp)})
    utility_sensitivity_df=pd.DataFrame(utility_rows)

    roll_df=rolling_origin(order,maps,nmin,beta)
    rho,rho_ci=spearman_bootstrap_joint_support(train,test,joint,args.support_bootstrap_reps,args.seed,maps)

    # Alternative aggregation operators.
    op_rows=[]
    for opname in ["geometric","minimum","arithmetic","harmonic","probabilistic_sum"]:
        oq=quality_function(rep_mu,opname,.05);ou=lambda g:(1-beta)*oq(g)+beta*ev(g)
        pp,ss,_=max_mean_path(joint,succ,pred,ou,source)
        op_rows.append({"Operator":opname,"Path nodes":len(pp),"Node Jaccard with baseline":jaccard(path,pp),"Training utility on its own scale":ss})
    op_df=pd.DataFrame(op_rows)

    maps_df=map_audit(train,test)

    full_refit=full_refit_bootstrap(train0,path,maps,nmin,beta,args.full_refit_reps,args.seed) if args.full_refit_reps>0 else pd.DataFrame()
    if len(full_refit):
        exact=int(full_refit["Exact baseline path"].sum());n=len(full_refit)
        ci_exact=stats.binomtest(exact,n).proportion_ci(confidence_level=.95,method="exact")
        full_summary=pd.DataFrame([{
            "Replicates":n,"Exact path recoveries":exact,"Exact recovery rate":exact/n,
            "Exact binomial CI low":ci_exact.low,"Exact binomial CI high":ci_exact.high,
            "Median node Jaccard":full_refit["Node Jaccard"].median(),"Node Jaccard 2.5%":full_refit["Node Jaccard"].quantile(.025),"Node Jaccard 97.5%":full_refit["Node Jaccard"].quantile(.975),
            "Median edge Jaccard":full_refit["Edge Jaccard"].median(),"Median path length":full_refit["Path length"].median(),"Median feasible nodes":full_refit["Feasible nodes"].median()
        }])
    else: full_summary=pd.DataFrame()
    if len(full_refit):
        refit_node_frequency, refit_edge_frequency = full_refit_frequency_tables(full_refit)
    else:
        refit_node_frequency, refit_edge_frequency = pd.DataFrame(), pd.DataFrame()

    # Core machine-readable tables.
    dataset_summary=pd.DataFrame([
        ["Raw item rows",len(raw)],["Unique orders",len(order)],["Training orders",len(train)],["Chronological holdout orders",len(test)],
        ["Observation start",str(order.order_date.min())],["Observation end",str(order.order_date.max())],["Markets",order.market.nunique()],["Shipping modes",order.shipping_mode.nunique()]
    ],columns=["Item","Value"])
    representative_levels=representative_level_table(train,rep_mu,rep_raw,ranges)
    descriptive=descriptive_statistics(order)
    benchmark_grid=inner_benchmark_grid(order,maps)
    lattice=pd.DataFrame([
        ["Commercial / order formation","C4 x C4 x C4",64,144,9,train.C_node.nunique()],
        ["Logistics execution","C4 x C4 x C2",32,64,7,train.O_node.nunique()],
        ["Market / financial response","C5 x C5 x C4",100,235,11,train.M_node.nunique()],
    ],columns=["Layer","Lattice","Algebraic nodes","Covers","Height","Observed training nodes"])
    path_rows=[]
    for i,g in enumerate(path,1):
        path_rows.append({"Stage":i,"Rank":rank_gamma(g),"Commercial":str(g[0]),"Logistics":str(g[1]),"Market":str(g[2]),"Coordination tier":str(maps[0](g[0])),"Joint training support":train_joint.get(g,0),"Joint holdout support":hold_joint.get(g,0),"Quality":q(g),"Evidence":ev(g),"Node utility":u(g)})
    path_df=pd.DataFrame(path_rows).merge(raw_path_values(path,rep_raw),on="Stage")
    optimality=pd.DataFrame([{"Source":format_gamma(source),"Reachable maximal targets":sum(not succ.get(g) for g in joint),"Source-to-maximal paths":len(all_paths),"Optimal path nodes":len(path),"Optimal mean utility":best,"Second-best mean utility":second,"Absolute gap":best-second,"Relative gap percent":100*(best-second)/best,"Paths within 1%":near1}])
    evidence_transport=pd.DataFrame([{"Spearman rho":rho,"Cluster-bootstrap CI low":rho_ci[0],"Cluster-bootstrap CI high":rho_ci[1],"Nodes":len(joint)}])

    # Detailed structural artifacts supporting the manuscript audit statements.
    full_edges=edge_table(full_succ)
    marginal_edges=edge_table(msucc)
    joint_edges=edge_table(succ)
    joint_nodes_table=node_support_table(joint,train_joint,hold_joint,q,ev,u)
    path_enumeration_rows=[]
    for path_id,index in enumerate(order_ix,1):
        candidate=all_paths[int(index)]
        score=float(path_scores[int(index)])
        path_enumeration_rows.append({
            "Ranked path":path_id,
            "Path nodes":len(candidate),
            "Mean utility":score,
            "Relative to optimum":score/best if best else np.nan,
            "Within 1 percent":bool(score>=.99*best),
            "Is optimum":bool(int(index)==int(order_ix[0])),
            "Rank sequence":" -> ".join(str(rank_gamma(g)) for g in candidate),
            "Node sequence":" || ".join(format_gamma(g) for g in candidate),
        })
    path_enumeration=pd.DataFrame(path_enumeration_rows)

    tables={
        "dataset_summary":dataset_summary,"field_audit":audit,"market_split":cutoffs,"descriptive_statistics":descriptive,
        "representative_levels":representative_levels,"inner_benchmark_grid":benchmark_grid,
        "lattice_summary":lattice,"order_structure_audit":closure,
        "joint_intrinsic_lattice_witnesses":joint_intrinsic_witnesses,
        "marginal_intrinsic_lattice_witnesses":marginal_intrinsic_witnesses,
        "optimal_path":path_df,"optimality_audit":optimality,
        "method_specific_radius_one_validation":validation_df,
        "exploratory_method_specific_bootstrap":boot_df,
        "common_cohort_distance_data":distance_frame,
        "common_cohort_distance_association":distance_association,
        "common_cohort_market_summary":market_distance,
        "common_cohort_distance_bootstrap":distance_bootstrap,
        "common_cohort_distance_bootstrap_replicates":distance_bootstrap_replicates,
        "radius_coverage_profile":radius_profile,"fixed_k_coverage_profile":fixed_k,
        "status_distribution":status_tables,"customer_overlap":overlap_df,
        "deterministic_dependency_audit":dependency_df,
        "coordination_map_audit":maps_df,"coordination_map_downstream_audit":map_downstream_df,
        "utility_weight_sensitivity":utility_sensitivity_df,"source_terminal_sensitivity":source_sensitivity_df,
        "rolling_origin":roll_df,"evidence_transport":evidence_transport,"operator_sensitivity":op_df,
        "full_pullback_edges":full_edges,"marginal_cover_edges":marginal_edges,"joint_cover_edges":joint_edges,
        "joint_node_support":joint_nodes_table,"path_enumeration":path_enumeration,
    }
    # Figure-facing table names.
    tables["closure_audit"] = closure
    tables["validation"] = validation_df
    tables["external_outcome_bootstrap"] = boot_df
    tables["external_outcome_bootstrap_replicates"] = bootstrap_replicates
    if len(full_refit): tables["full_refit_bootstrap"]=full_refit
    if len(full_summary): tables["full_refit_summary"]=full_summary
    if len(refit_node_frequency): tables["full_refit_node_frequency"]=refit_node_frequency
    if len(refit_edge_frequency): tables["full_refit_edge_frequency"]=refit_edge_frequency
    for name,df in tables.items(): save_csv(df,tdir/f"{name}.csv")
    (args.out/"outputs"/"transform_model.json").write_text(json.dumps(transform,indent=2,default=str),encoding="utf-8")

    # The manuscript and raw-data workflows use the same plotting functions.
    from generate_reference_figures import generate_all
    generate_all(tdir, fdir)

    results={
        "raw_rows":len(raw),"orders":len(order),"train_orders":len(train),"holdout_orders":len(test),
        "full_pullback_nodes":len(full),"full_pullback_covers":sum(len(v) for v in full_succ.values()),
        "joint_nodes":len(joint),"joint_covers":sum(len(v) for v in succ.values()),
        "marginal_nodes":len(marginal),"marginal_covers":sum(len(v) for v in msucc.values()),
        "joint_ambient_join_nonclosure_rate_unordered":joint_ambient.join_nonclosure_rate,
        "joint_ambient_meet_nonclosure_rate_unordered":joint_ambient.meet_nonclosure_rate,
        "joint_intrinsic_lattice":joint_intrinsic.is_intrinsic_lattice,
        "joint_pairs_without_internal_join":joint_intrinsic.pairs_without_internal_join,
        "joint_pairs_without_internal_meet":joint_intrinsic.pairs_without_internal_meet,
        "path_count":len(all_paths),"path_nodes":len(path),"optimal_mean":best,
        "second_best":second,"relative_gap_percent":100*(best-second)/best,
        "holdout_utility_frozen_training_scale":path_mean(path,hu),
        "mean_signed_order_status_score_radius_one":near_path_status(test,path,maps)[0],
        "radius_one_orders":near_path_status(test,path,maps)[1],
        "radius_one_share_complete_holdout":near_path_status(test,path,maps)[1]/len(test),
        "evidence_spearman":rho,
    }
    if len(full_summary):
        summary=full_summary.iloc[0]
        results.update({
            "full_refit_replicates":int(summary["Replicates"]),
            "full_refit_exact_recoveries":int(summary["Exact path recoveries"]),
            "full_refit_exact_recovery_rate":float(summary["Exact recovery rate"]),
            "full_refit_exact_ci_low":float(summary["Exact binomial CI low"]),
            "full_refit_exact_ci_high":float(summary["Exact binomial CI high"]),
            "median_node_jaccard":float(summary["Median node Jaccard"]),
            "node_jaccard_2_5pct":float(summary["Node Jaccard 2.5%"]),
            "node_jaccard_97_5pct":float(summary["Node Jaccard 97.5%"]),
            "median_edge_jaccard":float(summary["Median edge Jaccard"]),
            "median_path_length":float(summary["Median path length"]),
            "median_feasible_nodes":float(summary["Median feasible nodes"]),
        })
    (args.out/"outputs"/"results.json").write_text(json.dumps(results,indent=2,default=float)+"\n",encoding="utf-8")
    print(json.dumps(results,indent=2,default=float))


if __name__=="__main__":
    main()
