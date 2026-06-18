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

    def test_realistic_aug_off_is_identical_on_deterministic(self) -> None:
        """MDIE_REALISTIC_AUG=1 must change all four occluders deterministically;
        default OFF must be byte-identical to the flat synthetic baseline."""
        import importlib
        for k in ("MDIE_REALISTIC_AUG", "MDIE_REALISTIC_MASKS"):
            os.environ.pop(k, None)
        import research_v2.src.data.modifications as M
        importlib.reload(M)
        img = (np.random.RandomState(0).rand(112, 112, 3) * 255).astype("uint8")
        mods = ["disguise_glasses", "disguise_mask", "disguise_cap", "occlusion_random"]
        ref = {k: M.apply_modification(img, k, seed=5) for k in mods}

        os.environ["MDIE_REALISTIC_AUG"] = "1"
        importlib.reload(M)
        try:
            on1 = {k: M.apply_modification(img, k, seed=5) for k in mods}
            on2 = {k: M.apply_modification(img, k, seed=5) for k in mods}
            for k in mods:
                self.assertTrue(np.array_equal(on1[k], on2[k]), f"{k} not deterministic")
                self.assertFalse(np.array_equal(on1[k], ref[k]), f"{k} unchanged when ON")
            self.assertEqual(len(M.MODIFICATION_TYPES), 10)
        finally:
            os.environ.pop("MDIE_REALISTIC_AUG", None)
            importlib.reload(M)


class RMFRDPairingTests(unittest.TestCase):
    def test_two_tree_masked_unmasked_pairing(self) -> None:
        import importlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for tree, tag in (("AFDB_masked_face_dataset", "m"),
                              ("AFDB_face_dataset", "u")):
                for pid in range(5):
                    d = root / tree / f"id{pid}"
                    d.mkdir(parents=True)
                    for k in range(3):
                        img = (np.random.RandomState(pid * 7 + k).rand(112, 112, 3) * 255).astype("uint8")
                        cv2.imwrite(str(d / f"{tag}{k}.jpg"), img)
            os.environ["RMFRD_ROOT"] = str(root)
            try:
                import research_v2.src.data.benchmarks.rmfrd as R
                importlib.reload(R)
                b = R.load()
                pos = [p for p in b.pairs if p[2] == 1]
                self.assertGreater(len(pos), 0)
                # every genuine pair must cross the masked<->unmasked boundary
                for a, c, _ in pos:
                    self.assertNotEqual("masked" in str(a), "masked" in str(c))
            finally:
                os.environ.pop("RMFRD_ROOT", None)


if __name__ == "__main__":
    unittest.main()
