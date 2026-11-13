$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python code/reproduce.py @args
exit $LASTEXITCODE
