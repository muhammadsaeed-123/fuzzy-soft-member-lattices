#!/usr/bin/env sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
python -m pytest -q
python code/reproduce.py --smoke --out smoke_output "$@"
