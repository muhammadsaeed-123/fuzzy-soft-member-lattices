#!/usr/bin/env python3
"""Assemble numbered PDF, EPS and PNG figures from exact mathematical and data sources."""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def assemble(*,out:Path,empirical:Path,synthetic:Path,theory:Path)->dict:
    mapping=json.loads((ROOT/'config/figure_map.json').read_text())
    dirs={'theory':theory,'empirical':empirical,'synthetic':synthetic}
    planned=[]
    for doc in ('main_article','online_resource_1'):
        for stem,spec in mapping[doc].items():
            category,name=spec.split('/',1)
            for ext in ('.pdf','.eps','.png'):
                src=dirs[category]/(name+ext)
                if not src.is_file():raise FileNotFoundError(src)
                dst=out/doc/(stem+ext)
                if src.resolve()==dst.resolve():raise ValueError('Assembly destination must differ from source')
                planned.append((src,dst,spec,category))
    rows=[]
    for src,dst,spec,category in planned:
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        rows.append({'source_specification':spec,'relative_destination':str(dst.relative_to(out)),
                     'origin':'native TikZ source' if category=='theory' else 'machine-readable numerical results',
                     'generated_from_machine_readable_source':True,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
    report={'files':rows,'missing':[],'formats':['PDF','EPS','PNG']}
    out.mkdir(parents=True,exist_ok=True);(out/'figure_assembly.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--theory',type=Path,default=ROOT/'outputs/theory/figures')
    p.add_argument('--empirical',type=Path,default=ROOT/'outputs/reference/figures')
    p.add_argument('--synthetic',type=Path,default=ROOT/'outputs/synthetic/figures')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/publication_figures')
    a=p.parse_args();r=assemble(out=a.out,empirical=a.empirical,synthetic=a.synthetic,theory=a.theory);print(f'Assembled {len(r["files"])} figure files')
if __name__=='__main__':main()
