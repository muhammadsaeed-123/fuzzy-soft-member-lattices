#!/usr/bin/env python3
"""Render native LaTeX numerical tables from full-precision analysis CSV files."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHOD={'Full joint-support model':'Full joint model','Quality-only':'Quality-only','Ordinal-score ablation':'Ordinal ablation','Marginal-support construction':'Marginal support'}
ATTR={'avg_unit_value':'Unit value','discount_rate':'Discount rate','total_qty':'Total quantity','actual_days':'Actual days','scheduled_days':'Scheduled days','reliability':'Reliability','net_sales':'Net sales','total_profit':'Order profit','profit_margin':'Profit margin','unit_value':'Unit value','discount_preservation':'Discount preservation','order_scale':'Order scale','actual_speed':'Actual speed','planned_speed':'Planned speed','net_revenue':'Net revenue','order_profit':'Order profit'}
def esc(s):
    return str(s).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')
def fmt(x,n=3): return '--' if pd.isna(x) else f'{float(x):.{n}f}'
def integer(x): return '--' if pd.isna(x) else f'{int(x):,}'
def tup(x):return '$'+str(x).replace(' ','')+'$'
def body(headers, rows, first=.31, size='small', align=None):
    spec=align or ('@{}p{'+str(first)+r'\linewidth}'+r'*{'+str(len(headers)-1)+r'}{>{\raggedleft\arraybackslash}X}@{}')
    result='{\\'+size+r'\setlength{\tabcolsep}{3pt}'+'\n'+r'\begin{tabularx}{\linewidth}{'+spec+'}\n\\toprule\n'
    result+=' & '.join(headers)+r'\\'+'\n\\midrule\n'
    result+='\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)
    return result+'\n\\bottomrule\n\\end{tabularx}}\n'
def longbody(headers,rows,caption,label,spec):
    s=r'\begingroup\small\setlength{\tabcolsep}{3pt}'+'\n'+r'\begin{longtable}{'+spec+'}\n'
    s+=r'\caption{'+caption+r'}\label{'+label+r'}\\'+'\n\\toprule\n'+' & '.join(headers)+r'\\'+'\n\\midrule\n\\endfirsthead\n'
    s+=r'\caption[]{'+caption+r' (continued)}\\'+'\n\\toprule\n'+' & '.join(headers)+r'\\'+'\n\\midrule\n\\endhead\n'
    s+='\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)
    return s+'\n\\bottomrule\n\\end{longtable}\n\\endgroup\n'
def render(tables:Path,synthetic:Path,out:Path):
    a=out/'article';r=out/'resource1';a.mkdir(parents=True,exist_ok=True);r.mkdir(parents=True,exist_ok=True)
    load=lambda n:pd.read_csv(tables/(n+'.csv'))
    def save(stem,text,both=False):
        (r/(stem+'.tex')).write_text(text)
        if both:(a/(stem+'.tex')).write_text(text)
    cl=load('closure_audit');rows=[]
    for (_,x),name in zip(cl.iterrows(),['Full pullback','Marginal support','Joint support']):
        rows.append([name,integer(x['Nodes']),integer(x['Covers']),fmt(x['Ambient join nonclosure rate']),fmt(x['Ambient meet nonclosure rate'])])
    save('closure',body(['Structure','Nodes','Covers',r'$\bar\nu_{\vee}$',r'$\bar\nu_{\wedge}$'],rows),True)
    v=load('validation');rows=[]
    for _,x in v.iterrows():rows.append([METHOD[x.Method],integer(x['Graph nodes']),integer(x['Path nodes']),integer(x['Radius-one holdout orders']),fmt(100*x['Radius-one share of complete holdout'])+r'\%',fmt(x['Mean signed order-status score'],4)])
    save('validation',body(['Method','Graph $n$','Path $n$','Near $n$','Coverage','Signed score'],rows,first=.24,size='footnotesize'),True)
    p=load('optimal_path');rows=[]
    for _,x in p.iterrows():rows.append([integer(x.Stage),integer(x.Rank),tup(x.Commercial),tup(x.Logistics),tup(x.Market),integer(x['Joint training support']),fmt(x['Node utility'],4)])
    (a/'selected_path.tex').write_text(body(['Stage','Rank','Commercial','Logistics','Market',r'$n_{\rm tr}$',r'$u$'],rows,size='footnotesize',align='@{}rrXXXrr@{}'))
    # Data audit
    d=load('field_audit');rows=[]
    for _,x in d.iloc[:-1].iterrows():rows.append([esc(x['Field']),integer(x['Orders varying within order']),fmt(x.Percent,3)+r'\%'])
    rows.append(['Maximum accounting discrepancy','--',fmt(d.iloc[-1].Percent,6)])
    save('field_audit',body(['Field','Varying orders','Rate / discrepancy'],rows,first=.56,size='footnotesize'))
    d=load('market_split');rows=[]
    for _,x in d.iterrows():
        def date(x):return str(x)[:10]+r'\newline '+str(x)[11:16]
        rows.append([esc(x.Market),integer(x.Orders),integer(x.Training),integer(x.Holdout),date(x['Last training date']),date(x['First holdout date'])])
    save('market_split',body(['Market','Orders','Training','Holdout','Last training','First holdout'],rows,first=.18,size='footnotesize'))
    d=load('descriptive_statistics');rows=[]
    for _,x in d.iterrows():rows.append([esc(ATTR.get(x.Attribute,x.Attribute)),integer(x.N)]+[fmt(x[k],2) for k in ['Mean','SD','Min','Q1','Median','Q3','Max']])
    save('descriptive_statistics',body(['Attribute','$N$','Mean','SD','Min','Q1','Median','Q3','Max'],rows,first=.23,size='footnotesize'))
    d=load('representative_levels');rows=[]
    for _,x in d.iterrows():rows.append([esc(x.Layer),esc(ATTR.get(x.Attribute,x.Attribute)),integer(x.Level),fmt(x['Representative membership'],4),fmt(x['Representative raw value'],4),fmt(x['Observed training minimum'],4),fmt(x['Observed training maximum'],4),integer(x['Training orders'])])
    save('representative_levels',longbody(['Layer','Attribute','Level',r'$\widetilde\mu$','Raw median','Raw min','Raw max','Orders'],rows,'Training-derived representative membership levels and raw-value ranges','tab:s4','@{}llrrrrrr@{}'))
    d=load('coordination_map_audit');rows=[[esc(x.Map),fmt(x['Training exact coordination rate']*100,3)+r'\%',fmt(x['Holdout exact coordination rate']*100,3)+r'\%'] for _,x in d.iterrows()]
    save('coordination_map_audit',body(['Map','Training','Holdout'],rows,first=.50))
    d=load('inner_benchmark_grid');rows=[]
    for _,x in d.iterrows():rows.append([integer(x.n_min),fmt(x.beta,2),integer(x['Graph nodes']),integer(x['Path nodes']),integer(x['Inner near-path orders']),fmt(x['Inner status outcome']),fmt(x['Inner common utility']),fmt(x['Training utility'])])
    save('inner_grid',longbody([r'$n_{\min}$',r'$\beta$','Nodes','Path','Near $n$','Status','Holdout $u$','Train $u$'],rows,'Inner chronological benchmark grid','tab:s6','@{}rrrrrrrr@{}'))
    x=load('optimality_audit').iloc[0]
    rows=[[k,integer(x[k])] for k in ['Reachable maximal targets','Source-to-maximal paths','Optimal path nodes']]
    rows += [[k,fmt(x[k],9)] for k in ['Optimal mean utility','Second-best mean utility','Absolute gap']]
    rows += [['Relative gap',fmt(x['Relative gap percent'],6)+r'\%'],[r'Paths within 1\%',integer(x['Paths within 1%'])]]
    save('optimality',body(['Measure','Value'],rows,first=.67))
    rows=[]
    for _,x in p.iterrows(): rows.append([integer(x.Stage),integer(x.Rank),tup(x.Commercial),tup(x.Logistics),tup(x.Market),tup(x['Coordination tier']),integer(x['Joint training support']),integer(x['Joint holdout support']),fmt(x.Quality,4),fmt(x.Evidence,4),fmt(x['Node utility'],4)])
    save('full_path',longbody(['Stage','Rank','Commercial','Logistics','Market','$K$',r'$n_{tr}$',r'$n_{ho}$','$Q$','$E$','$u$'],rows,'Complete selected ordered-state path','tab:s9','@{}rrrrrrrrrrr@{}'))
    rows=[]
    for _,x in p.iterrows():rows.append([integer(x.Stage),fmt(x.unit_value,2),fmt(x.discount_preservation,4),integer(x.order_scale),integer(x.actual_speed),integer(x.planned_speed),integer(x.reliability),fmt(x.net_revenue,2),fmt(x.order_profit,2),fmt(x.profit_margin,4)])
    save('raw_path',longbody(['Stage','Unit value','Discount','Qty','Actual days','Scheduled','Reliable','Net sales','Profit','Margin'],rows,'Representative raw values attached to the selected path','tab:s10','@{}rrrrrrrrrr@{}'))
    d=load('common_cohort_distance_bootstrap');rows=[[METHOD[x.Method],fmt(x['Point Spearman correlation'],4),'['+fmt(x['Bootstrap CI low'],4)+', '+fmt(x['Bootstrap CI high'],4)+']'] for _,x in d.iterrows()]
    save('common_cohort',body(['Method',r'Spearman $\rho$',r'95\% interval'],rows,first=.39))
    d=load('fixed_k_coverage_profile');rows=[]
    for method in d.Method.unique():rows.append([METHOD[method]]+[fmt(d.loc[(d.Method==method)&(d['Target k']==k),'Mean signed status score'].iloc[0],4) for k in (48,97,195)])
    save('fixed_k',body(['Method','$k=48$','$k=97$','$k=195$'],rows,first=.39))
    d=load('external_outcome_bootstrap');rows=[]
    for _,x in d.iterrows():rows.append([esc(x.Comparison).replace('Ordinal-score ablation','Ordinal').replace('Marginal-support construction','Marginal'),fmt(x['Point difference'],4),'['+fmt(x['CI low'],4)+', '+fmt(x['CI high'],4)+']',integer(x.Replicates)])
    save('external_bootstrap',body(['Comparison','Contrast',r'95\% interval','Refits'],rows,first=.36,size='footnotesize'))
    d=load('rolling_origin');rows=[]
    for _,x in d.iterrows():rows.append([esc(x['Test fraction']),fmt(x['Full outcome']),fmt(x['Quality-only outcome']),integer(x['Full near orders']),integer(x['Quality near orders'])])
    save('rolling',body(['Window','Full score','Quality score','Full near $n$','Quality near $n$'],rows,first=.24,size='footnotesize'))
    x=load('evidence_transport').iloc[0];save('evidence_transport',body(['Measure','Value'],[[r'Spearman $\rho$',fmt(x['Spearman rho'],6)],[r'Customer-cluster 95\% interval','['+fmt(x['Cluster-bootstrap CI low'],6)+', '+fmt(x['Cluster-bootstrap CI high'],6)+']'],['Joint nodes',integer(x.Nodes)]],first=.57))
    x=load('full_refit_summary').iloc[0]
    rows=[['Replicates',integer(x.Replicates)],['Exact path recoveries',integer(x['Exact path recoveries'])],['Exact recovery rate',fmt(x['Exact recovery rate']*100,2)+r'\%'],[r'Exact binomial 95\% interval','['+fmt(x['Exact binomial CI low']*100,2)+r'\%, '+fmt(x['Exact binomial CI high']*100,2)+r'\%]'],['Median node Jaccard',fmt(x['Median node Jaccard'],4)],[r'Node Jaccard 2.5--97.5\%','['+fmt(x['Node Jaccard 2.5%'],4)+', '+fmt(x['Node Jaccard 97.5%'],4)+']'],['Median edge Jaccard',fmt(x['Median edge Jaccard'],4)],['Median path length',integer(x['Median path length'])],['Median feasible nodes',integer(x['Median feasible nodes'])]]
    save('full_refit',body(['Measure','Value'],rows,first=.62))
    d=load('operator_sensitivity');rows=[[esc(x.Operator).replace(r'\_',' '),integer(x['Path nodes']),fmt(x['Node Jaccard with baseline'],4),fmt(x['Training utility on its own scale'],6)] for _,x in d.iterrows()]
    save('operator',body(['Operator','Path $n$','Node Jaccard','Own-scale utility'],rows,first=.32))
    d=pd.read_csv(synthetic/'method_comparison.csv');names=['Domain-anchored','Robust min--max','Rank-only','Crisp level','Greedy monotone','Random mean']
    rows=[[name,integer(x.graph_nodes),fmt(x.path_nodes,2).removesuffix('.00'),fmt(x.training_mean_utility,4),fmt(x.common_cohort_spearman,4),fmt(x.matched_coverage_outcome,4)] for (_,x),name in zip(d.iterrows(),names)]
    save('synthetic_methods',body(['Method','Graph','Path','Train mean',r'$\rho$','Matched score'],rows,first=.28,size='footnotesize'))
    d=pd.read_csv(synthetic/'scaling.csv');rows=[[integer(x.dimension),integer(x.levels_per_dimension),integer(x.nodes),integer(x.covers),fmt(x.dp_seconds,7)] for _,x in d.iterrows()]
    save('synthetic_scaling',body(['Dimensions','Levels','Nodes','Covers','DP seconds'],rows,first=.24))
    print('LaTeX numerical tables:',out)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tables',type=Path,default=ROOT/'outputs/reference/tables')
    p.add_argument('--synthetic',type=Path,default=ROOT/'outputs/synthetic/tables')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/manuscript_tables')
    a=p.parse_args();render(a.tables,a.synthetic,a.out)
if __name__=='__main__':main()
