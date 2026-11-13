# Run guide

See README.md for dependencies.

```sh
python -m pip install -r requirements.txt
python verify_manifest.py
python -m pytest -q
python code/reproduce.py --out reproduction_output
```

The mathematical figure source is native TikZ; empirical figure source is Python, and all displayed tables are native LaTeX.
