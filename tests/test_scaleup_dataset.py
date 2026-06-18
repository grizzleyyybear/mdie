"""Light, offline tests for the scale-up additions: the pluggable training
source, the IR-100 backbone, and the opt-in realistic mask. All synthetic and
CPU-only; none touch the network or any staged dataset."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


def _make_imagefolder(root: Path, n_ids: int = 3, per_id: int = 4) -> None:
    for pid in range(n_ids):
        d = root / f"{pid:07d}"
        d.mkdir(parents=True)
        for k in range(per_id):
            img = (np.random.RandomState(pid * 10 + k).rand(112, 112, 3) * 255).astype("uint8")
            cv2.imwrite(str(d / f"{k}.jpg"), img)


class TrainSourceTests(unittest.TestCase):
    def test_build_casia_from_imagefolder(self) -> None:
        from research_v2.src.data.casia import build_train_dataset
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            _make_imagefolder(cache / "casia", n_ids=4, per_id=5)
            paths, labels, names = build_train_dataset("casia", cache, min_imgs=3)
            self.assertEqual(len(paths), 20)
            self.assertEqual(len(set(labels)), 4)
            self.assertEqual(len(names), len(set(labels)))

    def test_casia_missing_raises(self) -> None:
        from research_v2.src.data.casia import prepare_casia
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                prepare_casia(Path(tmp))

    def test_unknown_source_raises(self) -> None:
        from research_v2.src.data.casia import build_train_dataset
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_train_dataset("ms1m", Path(tmp))


class IR100BackboneTests(unittest.TestCase):
    def test_ir100_output_contract(self) -> None:
        import torch
        from research_v2.src.models.backbones import build_backbone
        m = build_backbone("ir100", embedding_dim=512, return_maps=True).eval()
        with torch.no_grad():
            out = m(torch.randn(2, 3, 112, 112))
        self.assertEqual(tuple(out["embedding"].shape), (2, 512))
        self.assertEqual(tuple(out["feature_maps"].shape), (2, 512, 7, 7))
        self.assertEqual(tuple(out["feature_maps_hi"].shape), (2, 256, 14, 14))


class RealisticMaskTests(unittest.TestCase):
    def test_default_off_is_flat_polygon(self) -> None:
        os.environ.pop("MDIE_REALISTIC_MASKS", None)
        import importlib
        import research_v2.src.data.modifications as M
        importlib.reload(M)
        img = (np.random.RandomState(0).rand(112, 112, 3) * 255).astype("uint8")
        a = M._disguise_mask(img, np.random.RandomState(7))
        # independent flat reference
        out = img.copy(); h, w = img.shape[:2]
        pts = np.array([[int(w*0.18), int(h*0.55)], [int(w*0.82), int(h*0.55)],
                        [int(w*0.92), int(h*0.78)], [int(w*0.74), int(h*0.96)],
                        [int(w*0.26), int(h*0.96)], [int(w*0.08), int(h*0.78)]],
                       dtype=np.int32)
        colors = [(220, 220, 220), (110, 130, 200), (50, 70, 110)]
        c = colors[np.random.RandomState(7).randint(len(colors))]
        cv2.fillPoly(out, [pts], tuple(int(x) for x in c))
        self.assertTrue(np.array_equal(a, out))
        self.assertEqual(len(M.MODIFICATION_TYPES), 10)

    def test_realistic_on_is_deterministic(self) -> None:
        os.environ["MDIE_REALISTIC_MASKS"] = "1"
        import importlib
        import research_v2.src.data.modifications as M
        importlib.reload(M)
        try:
            img = (np.random.RandomState(1).rand(112, 112, 3) * 255).astype("uint8")
            b1 = M._disguise_mask(img, np.random.RandomState(3))
            b2 = M._disguise_mask(img, np.random.RandomState(3))
            self.assertTrue(np.array_equal(b1, b2))
            self.assertEqual(b1.shape, img.shape)
        finally:
            os.environ.pop("MDIE_REALISTIC_MASKS", None)
            importlib.reload(M)


if __name__ == "__main__":
    unittest.main()
