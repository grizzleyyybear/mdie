$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = (Get-Location).Path
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

.\.venv\Scripts\python.exe -m research_v2.src.run_stage1 `
    --quick `
    --batch 16 `
    --workers 2 `
    --channels-last `
    --val-pairs 200

