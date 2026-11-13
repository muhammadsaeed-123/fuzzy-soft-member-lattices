.PHONY: test reproduce smoke verify
test:
	python -m pytest -q
reproduce:
	python code/reproduce.py
smoke:
	python code/reproduce.py --smoke --out smoke_output
verify:
	python verify_manifest.py
