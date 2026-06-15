$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

.\run_smoke.ps1
.\run_stage1_quick.ps1
.\run_stage2_quick.ps1
.\run_real_benchmarks.ps1

