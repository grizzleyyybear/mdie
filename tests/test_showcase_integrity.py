from __future__ import annotations

import compileall
import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "research_v2" / "src"
PARAM_SOURCE_DIR = ROOT / "showcase" / "01_param_training" / "research_v2" / "src"
LAPTOP_SOURCE_DIR = ROOT / "showcase" / "02_laptop_rtx3050" / "research_v2" / "src"

STALE_IMPORTS = re.compile(
    r"data\.pairs|data\.samplers|eval\.quick_val|models\.losses|"
    r"baseline_models|eval\.metrics|from \.pairs|from \.samplers|"
    r"from \.losses|from \.metrics"
)


def python_files(root: Path) -> dict[str, Path]:
    return {
        str(path.relative_to(root)): path
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ShowcaseIntegrityTests(unittest.TestCase):
    def test_showcase_has_three_handoff_folders(self) -> None:
        required = [
            ROOT / "showcase" / "01_param_training",
            ROOT / "showcase" / "02_laptop_rtx3050",
            ROOT / "showcase" / "03_theory_and_diagrams",
            ROOT / "showcase" / "01_param_training" / "PARAM_HANDBOOK.md",
            ROOT / "showcase" / "02_laptop_rtx3050" / "run_smoke.ps1",
            ROOT / "showcase" / "03_theory_and_diagrams" / "documents" / "PRESENTATION_NOTES.md",
            ROOT / "showcase" / "03_theory_and_diagrams" / "figures" / "mdie_research_paper.pdf",
        ]
        missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
        self.assertEqual(missing, [])

    def test_legacy_top_level_folders_are_not_back(self) -> None:
        legacy = [
            "baselines",
            "configs",
            "data",
            "demo",
            "deployment",
            "evaluation",
            "federated",
            "models",
            "scripts",
            "training",
        ]
        present = [name for name in legacy if (ROOT / name).exists()]
        self.assertEqual(present, [])

    def test_consolidated_helper_files_stay_removed(self) -> None:
        removed = [
            SOURCE_DIR / "data" / "pairs.py",
            SOURCE_DIR / "data" / "samplers.py",
            SOURCE_DIR / "eval" / "quick_val.py",
            SOURCE_DIR / "eval" / "metrics.py",
            SOURCE_DIR / "models" / "losses.py",
            SOURCE_DIR / "baselines" / "baseline_models.py",
        ]
        present = [str(path.relative_to(ROOT)) for path in removed if path.exists()]
        self.assertEqual(present, [])

    def test_no_stale_imports_after_consolidation(self) -> None:
        roots = [SOURCE_DIR, ROOT / "showcase"]
        offenders: list[str] = []
        for root in roots:
            for path in root.rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if STALE_IMPORTS.search(line):
                        offenders.append(f"{path.relative_to(ROOT)}:{line_no}:{line.strip()}")
        self.assertEqual(offenders, [])

    def test_showcase_source_copies_match_maintained_source(self) -> None:
        maintained = python_files(SOURCE_DIR)
        self.assertGreater(len(maintained), 0)
        for copy_root in (PARAM_SOURCE_DIR, LAPTOP_SOURCE_DIR):
            copied = python_files(copy_root)
            self.assertEqual(sorted(copied), sorted(maintained))
            mismatches = [
                rel for rel, source_path in maintained.items()
                if sha256(source_path) != sha256(copied[rel])
            ]
            self.assertEqual(mismatches, [])

    def test_python_sources_compile(self) -> None:
        for folder in (SOURCE_DIR, PARAM_SOURCE_DIR, LAPTOP_SOURCE_DIR):
            ok = compileall.compile_dir(str(folder), quiet=1, force=True)
            self.assertTrue(ok, f"compile failed for {folder.relative_to(ROOT)}")


if __name__ == "__main__":
    unittest.main()
