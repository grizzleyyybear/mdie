"""Small shared helpers for benchmark loaders (downloads, pair-TSV IO)."""
from __future__ import annotations

import csv
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple

from ...config import DATA_DIR


def cache_dir(name: str) -> Path:
    d = DATA_DIR / "benchmarks" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(url: str, dest: Path, label: str | None = None) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [download] {label or dest.name} <- {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:  # noqa: BLE001
        print(f"  [download] FAILED ({e}); benchmark may be skipped")
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
    return dest


def unzip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        return out_dir
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    return out_dir


def write_pairs_tsv(path: Path, pairs: List[Tuple[Path, Path, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        for a, b, y in pairs:
            w.writerow([str(a), str(b), int(y)])


def read_pairs_tsv(path: Path) -> List[Tuple[Path, Path, int]]:
    out: List[Tuple[Path, Path, int]] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3:
                continue
            out.append((Path(row[0]), Path(row[1]), int(row[2])))
    return out


def env_root(var: str) -> Path | None:
    v = os.environ.get(var)
    if not v:
        return None
    p = Path(v).expanduser()
    return p if p.exists() else None
