# MDIE Laptop Package - RTX 3050 4 GB

This is the laptop-safe version of the MDIE codebase. It keeps the same source implementation as the PARAM package, but the launch scripts use smaller batches and quick settings so the project can be demonstrated on a 4 GB GPU.

## Layout

```text
02_laptop_rtx3050\
|-- research_v2\src\              MDIE source code
|-- research_v2\results\          latest lightweight result files
|-- research_v2\datasets_cache\   optional local dataset location
|-- research_v2\checkpoints\      optional local checkpoints
|-- setup_laptop_cuda121.ps1      install CUDA PyTorch + dependencies
|-- show_cached_results.ps1       print saved headline metrics without training
|-- run_smoke.ps1                 fast health check
|-- run_stage1_quick.ps1          baseline failure run
|-- run_stage2_quick.ps1          MDIE quick run
|-- run_real_benchmarks.ps1       public benchmark eval if data is present
`-- requirements.txt
```

## Setup

Open PowerShell in this folder:

```powershell
.\setup_laptop_cuda121.ps1
```

Then verify the code path:

```powershell
.\run_smoke.ps1
```

If you only need to present saved numbers without installing PyTorch:

```powershell
.\show_cached_results.ps1
```

## Run order for a laptop demo

```powershell
.\run_stage1_quick.ps1
.\run_stage2_quick.ps1
.\run_real_benchmarks.ps1
```

The quick scripts are intentionally conservative:

- batch size 16-24
- fp16 AMP when CUDA is available
- channels-last memory format
- small validation pair count
- no destructive cleanup

## Dataset note

The code expects LFW and benchmark data under:

```text
research_v2\datasets_cache\
```

For a no-internet presentation, use the already generated metrics in `research_v2\results` and the diagrams in `..\03_theory_and_diagrams`.

