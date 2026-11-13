#!/usr/bin/env python3
"""Compare every numerical CSV cell against the full-precision reference run."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def compare_results(generated:dict,reference:dict)->list[dict]:
    rows=[]
    for key,value in reference.items():
        actual=generated.get(key)
        if isinstance(value,bool):ok=isinstance(actual,bool) and actual==value
        elif isinstance(value,int):ok=actual==value
        elif isinstance(value,float):ok=isinstance(actual,(int,float)) and math.isclose(actual,value,rel_tol=1e-9,abs_tol=1e-10)
        else:ok=actual==value
        rows.append({'item':'results.'+key,'status':'pass' if ok else 'fail','expected':value,'actual':actual})
    return rows

def compare_keyed_table(generated_path:Path,reference_path:Path,key_columns:list[str]|None=None)->list[dict]:
    if not generated_path.is_file():return [{'item':reference_path.name,'status':'missing'}]
    g=pd.read_csv(generated_path);r=pd.read_csv(reference_path)
    try:
        if key_columns:g=g.sort_values(key_columns).reset_index(drop=True);r=r.sort_values(key_columns).reset_index(drop=True)
        pd.testing.assert_frame_equal(g,r,check_dtype=False,check_exact=False,rtol=1e-9,atol=1e-10)
        return [{'item':reference_path.name,'status':'pass','rows':len(r),'columns':len(r.columns)}]
    except AssertionError as e:return [{'item':reference_path.name,'status':'fail','details':str(e)[:1500]}]

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generated',type=Path,required=True,help='Directory containing outputs/results.json')
    p.add_argument('--reference',type=Path,default=ROOT/'outputs/reference')
    p.add_argument('--report',type=Path)
    a=p.parse_args();g=a.generated/'outputs';checks=[]
    checks+=compare_results(json.loads((g/'results.json').read_text()),json.loads((a.reference/'results.json').read_text()))
    for rp in sorted((a.reference/'tables').glob('*.csv')):checks+=compare_keyed_table(g/'tables'/rp.name,rp)
    for rp in sorted((a.reference/'figures').glob('*.pdf')):
        path=g/'figures'/rp.name;checks.append({'item':rp.name,'status':'pass' if path.is_file() and path.stat().st_size>1000 else 'missing'})
    failed=[r for r in checks if r['status']!='pass']
    report={'overall_status':'PASS' if not failed else 'FAIL','summary':{'total':len(checks),'passed':len(checks)-len(failed),'failed':len(failed)},'checks':checks}
    out=a.report or a.generated/'verification_report.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary']));print(report['overall_status'])
    for row in failed[:10]:print(row)
    return bool(failed)
if __name__=='__main__':raise SystemExit(main())
