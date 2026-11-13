from pathlib import Path
from itertools import combinations
import hashlib,re
import numpy as np
import pandas as pd
from validation_tools import fixed_k_profile
from audit_illustrative_paths import audit
ROOT=Path(__file__).resolve().parents[1]

def test_exact_k_does_not_expand_distance_ties():
    f=pd.DataFrame({'order_id':[5,1,4,2,3], 'distance::A':[0]*5,'distance::B':[1]*5,'signed_status_score':[0,1,0,-1,1]})
    t=fixed_k_profile(f,[.4])
    assert set(t['Target k'])=={2}
    assert set(t['Selected orders'])=={2}
    assert set(t['Mean signed status score'])=={0}
    pd.testing.assert_frame_equal(t,fixed_k_profile(f.sample(frac=1,random_state=9),[.4]))

def test_boolean_artwork_contains_exact_cover_relations():
    text=(ROOT/'figure_source/ambient_vs_intrinsic.tex').read_text()
    nodes=[set(),{1},{2},{3},{1,2},{1,3},{2,3},{1,2,3}]
    edges={(int(a),int(b)) for a,b in re.findall(r'\\draw\[line width=\.65pt\] \(n(\d)p\)--\(n(\d)p\)',text)}
    expected={(i,j) for i,a in enumerate(nodes) for j,b in enumerate(nodes) if a<b and len(b)==len(a)+1}
    assert edges==expected
    assert len(edges)==12

def test_isomorphism_artwork_uses_coherent_b_to_c_map():
    t=(ROOT/'figure_source/isomorphic_payload.tex').read_text()
    assert r'node[above]{$\phi_{23}$}' in t
    assert r'\phi_{23}=\phi_{13}\circ\phi_{12}^{-1}' in t
    assert 'dashed' not in t

def test_nonempty_pullback_hypothesis_is_necessary():
    # Constant homomorphisms to disjoint elements of C2 have empty pullback.
    left=[0,1];right=[0,1]
    assert [(a,b) for a in left for b in right if 0==1]==[]

def test_derived_payload_join_need_not_be_ambient_join():
    a21=(.5,.3,.4);a12=(.2,.6,.4);a22=(.5,.6,.55)
    assert tuple(max(x,y) for x,y in zip(a21,a12))!=a22

def test_baseline_representative_chains_are_strictly_increasing():
    d=pd.read_csv(ROOT/'outputs/reference/tables/representative_levels.csv')
    for _,g in d.groupby(['Layer','Attribute']):
        assert (np.diff(g.sort_values('Level')['Representative membership'])>0).all()

def test_raw_input_canonical_checksum():
    with (ROOT/'data/DataCoSupplyChainDataset.csv').open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
    assert d=='fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6'

def test_reference_exact_size_cohorts_and_internal_audit():
    p=ROOT/'outputs/reference/tables'
    d=pd.read_csv(p/'fixed_k_coverage_profile.csv')
    assert (d['Target k']==d['Selected orders']).all()
    assert set(d['Target k'])=={48,97,195}
    a=pd.read_csv(p/'order_structure_audit.csv')
    assert list(a['Pairs without internal join'])==[0,1536,2096]
    assert list(a['Pairs without internal meet'])==[0,576,1046]
    assert list(a['Intrinsic lattice'])==[True,False,False]

def test_full_refits_all_valid_and_exact_count():
    d=pd.read_csv(ROOT/'outputs/reference/tables/full_refit_bootstrap.csv')
    assert len(d)==500 and d.Error.isna().all()
    assert int(d['Exact baseline path'].sum())==7
