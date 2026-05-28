"""
Real face-recognition benchmarks for MDIE evaluation.

Each loader is a small, pure-Python module that produces a uniform pair-list TSV
and a list of image paths. No GPU required for any of these.

Available benchmarks (registry below):
    mfr2          — Masked Face Recognition 2 (public)
    calfw         — Cross-Age LFW (public)
    agedb30       — AgeDB-30 (public)
    iiitd_surgery — IIITD Plastic Surgery (gated behind IIITD_ROOT env var)
    ijbc_occ      — IJB-C occlusion protocol (gated behind IJBC_ROOT)

Usage:
    from research_v2.src.data.benchmarks import load_benchmark
    bench = load_benchmark("mfr2")           # downloads if free + missing
    bench.pairs        # List[(Path, Path, int)]
    bench.name         # "mfr2"
    bench.folds        # Optional[List[List[int]]]  for 10-fold protocols
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict


@dataclass
class Benchmark:
    name: str
    pairs: List[Tuple[Path, Path, int]]
    folds: Optional[List[List[int]]] = None
    notes: str = ""


_REGISTRY: Dict[str, Callable[[], Benchmark]] = {}


def register(name: str):
    def deco(fn: Callable[[], Benchmark]):
        _REGISTRY[name] = fn
        return fn
    return deco


def load_benchmark(name: str) -> Benchmark:
    if name not in _REGISTRY:
        raise KeyError(f"unknown benchmark {name!r}; available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def list_benchmarks() -> List[str]:
    return list(_REGISTRY)


from . import mfr2 as _mfr2          # noqa: F401,E402
from . import calfw as _calfw        # noqa: F401,E402
from . import agedb30 as _agedb30    # noqa: F401,E402
from . import iiitd_surgery as _iiitd  # noqa: F401,E402
from . import ijbc_occ as _ijbc     # noqa: F401,E402
