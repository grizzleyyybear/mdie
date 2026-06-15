$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    $created = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.11 -m venv .venv
        if ($LASTEXITCODE -eq 0) { $created = $true }
    }
    if (-not $created) {
        python -m venv .venv
    }
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch>=2.1.0" "torchvision>=0.16.0"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -c "import torch; print('torch=' + torch.__version__); print('cuda_available=' + str(torch.cuda.is_available())); print('device=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'))"

