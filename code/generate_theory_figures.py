#!/usr/bin/env python3
"""Compile exact TikZ mathematical diagrams into PDF, EPS and 1200-dpi PNG."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def generate(source: Path, out: Path, dpi: int = 1200) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for tex in sorted(source.glob('*.tex')):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={tmp}',str(tex)],check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            pdf=out/(tex.stem+'.pdf');shutil.copy2(Path(tmp)/(tex.stem+'.pdf'),pdf)
        subprocess.run(['pdftops','-eps',str(pdf),str(out/(tex.stem+'.eps'))],check=True)
        subprocess.run(['pdftoppm','-r',str(dpi),'-png','-singlefile',str(pdf),str(out/tex.stem)],check=True)
        shutil.copy2(tex,out/tex.name)
        print(tex.stem,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'figure_source')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/theory/figures')
    parser.add_argument('--dpi',type=int,default=1200)
    args=parser.parse_args();generate(args.source,args.out,args.dpi)
if __name__=='__main__':main()
