#!/usr/bin/env python3
"""Portable entry point for the complete experiment or a table-based figure check."""
from __future__ import annotations
import argparse,hashlib,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SHA='fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6'
def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'reproduction_output')
    p.add_argument('--smoke',action='store_true',help='Use included result tables; do not rerun the DataCo or synthetic resampling')
    a=p.parse_args();out=a.out.resolve()
    if out.exists() and any(out.iterdir()):p.error('Output directory is not empty. Choose a new --out directory to preserve existing work.')
    out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update({'MPLBACKEND':'Agg','PYTHONPATH':str(ROOT/'code'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    def run(script,*args):
        cmd=[sys.executable,str(ROOT/'code'/script),*map(str,args)]
        print(' '.join(cmd),flush=True);subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    if not a.smoke:
        data=ROOT/'data/DataCoSupplyChainDataset.csv'
        if not data.exists():raise FileNotFoundError('Place the canonical transaction CSV in data/; see DATA_LICENSE.md and data/README.md.')
        with data.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=SHA:raise ValueError('Transaction checksum differs from the declared experiment.')
        run('run_analysis.py','--data',data,'--out',out/'dataco','--cluster-bootstrap-reps',2000,'--support-bootstrap-reps',1000,'--full-refit-reps',500,'--seed',20260819)
        run('run_synthetic_benchmark.py','--out',out/'synthetic','--n',2000,'--bootstrap-reps',200,'--seed',20260825)
        empirical=out/'dataco/outputs/figures';synthetic=out/'synthetic/figures'
        tables=out/'dataco/outputs/tables';syntables=out/'synthetic/tables'
        run('verify_outputs.py','--generated',out/'dataco')
    else:
        tables=ROOT/'outputs/reference/tables';syntables=ROOT/'outputs/synthetic/tables'
        empirical=out/'empirical_figures';synthetic=out/'synthetic_figures'
        run('generate_reference_figures.py','--tables',tables,'--out',empirical)
        run('generate_synthetic_figures.py','--tables',syntables,'--out',synthetic)
    run('audit_illustrative_paths.py','--out',out/'theory/audit')
    run('generate_theory_figures.py','--out',out/'theory/figures','--dpi',1200)
    run('render_manuscript_tables.py','--tables',tables,'--synthetic',syntables,'--out',out/'manuscript_tables')
    run('assemble_manuscript_figures.py','--theory',out/'theory/figures','--empirical',empirical,'--synthetic',synthetic,'--out',out/'publication_figures')
    print('Completed:',out)
    return 0
if __name__=='__main__':raise SystemExit(main())
