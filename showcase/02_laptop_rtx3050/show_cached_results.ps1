$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$stage2Path = "research_v2\results\stage2_metrics.json"
$realPath = "research_v2\results\real_benchmarks.csv"

if (-not (Test-Path $stage2Path)) {
    throw "Missing $stage2Path"
}
if (-not (Test-Path $realPath)) {
    throw "Missing $realPath"
}

$stage2 = Get-Content $stage2Path -Raw | ConvertFrom-Json
$real = Import-Csv $realPath

$lines = @()
$lines += "Controlled modified-LFW pooled metrics"
$lines += "--------------------------------------"
$stage2.PSObject.Properties | ForEach-Object {
    $model = $_.Name
    $m = $_.Value
    if ($m.PSObject.Properties.Name -contains "pooled") {
        $lines += "{0,-16} AUC={1:N3}  EER={2:P1}  pairs={3}" -f $model, [double]$m.pooled.auc, [double]$m.pooled.eer, [int]$m.pooled.n_pairs
    }
}

$lines += "--------------------------------------"
$lines += "ArcFace degradation example"
$lines += "---------------------------"
$clean = [double]$stage2.arcface.clean.eer
$mask = [double]$stage2.arcface.disguise_mask.eer
$delta = $mask - $clean
$lines += "clean EER={0:P1}  masked EER={1:P1}  delta={2:P1}" -f $clean, $mask, $delta

$lines += "---------------------------"
$lines += "Real benchmark best model by EER"
$lines += "--------------------------------"
$real | Group-Object benchmark | ForEach-Object {
    $best = $_.Group | Sort-Object { [double]$_.eer } | Select-Object -First 1
    $lines += "{0,-8} best={1,-22} AUC={2:N3}  EER={3:P1}" -f $best.benchmark, $best.model, [double]$best.auc, [double]$best.eer
}

[Console]::WriteLine(($lines -join [Environment]::NewLine))
