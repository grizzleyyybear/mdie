$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = (Get-Location).Path
$env:PYTHONUNBUFFERED = "1"
$env:MDIE_SKIP_DATASET_PREFLIGHT = "auto"

.\.venv\Scripts\python.exe -m research_v2.src.preflight

