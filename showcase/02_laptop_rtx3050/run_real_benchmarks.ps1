$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = (Get-Location).Path
$env:PYTHONUNBUFFERED = "1"

.\.venv\Scripts\python.exe -m research_v2.src.eval.run_real_benchmarks `
    --benchmarks mfr2 calfw agedb30

