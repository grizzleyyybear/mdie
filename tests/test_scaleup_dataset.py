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


class BGIVisibilityTests(unittest.TestCase):
    """The two new opt-in novelties: Bone-Geometry Identity (BGI) auxiliary head
    and self-supervised visibility gating. All synthetic, CPU-only, default-OFF
    paths must stay byte-for-byte unchanged."""

    def test_geometry_signature_scale_invariant(self) -> None:
        from research_v2.src.data.landmarks import (bone_geometry_signature,
                                                    GEOM_DIM)
        lm = np.zeros((68, 2), np.float32)
        lm[36] = (30, 50); lm[39] = (45, 50); lm[42] = (67, 50); lm[45] = (82, 50)
        lm[17] = (28, 40); lm[19] = (36, 38); lm[21] = (46, 40)
        lm[22] = (66, 40); lm[24] = (76, 38); lm[26] = (84, 40)
        lm[27] = (56, 52); lm[28] = (56, 60)
        lm[2] = (24, 70); lm[3] = (26, 80); lm[4] = (30, 86); lm[5] = (34, 92)
        lm[14] = (88, 70); lm[13] = (86, 80); lm[12] = (82, 86); lm[11] = (78, 92)
        lm[7] = (50, 100); lm[8] = (56, 104); lm[9] = (62, 100)
        g = bone_geometry_signature(lm)
        self.assertEqual(g.shape, (GEOM_DIM,))
        # scale + translation invariant
        g2 = bone_geometry_signature(lm * 2.0 + 11.0)
        self.assertTrue(np.allclose(g, g2, atol=1e-4))
        # degenerate input -> all-zero "no target"
        self.assertTrue((bone_geometry_signature(np.zeros((68, 2), np.float32)) == 0).all())

    def test_geom_cache_roundtrip_and_backcompat(self) -> None:
        from research_v2.src.data.landmarks import load_geom_cache, GEOM_DIM
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.npz"
            keys = np.array(["a", "b"])
            np.savez_compressed(p, keys=keys,
                                targets=np.zeros((2, 14, 14), np.float32),
                                geoms=np.arange(2 * GEOM_DIM, dtype=np.float32).reshape(2, GEOM_DIM),
                                version=np.int64(4))
            gc = load_geom_cache(p)
            self.assertEqual(set(gc), {"a", "b"})
            self.assertEqual(gc["a"].shape, (GEOM_DIM,))
            # an old cache without the geoms key degrades gracefully to {}
            p2 = Path(tmp) / "old.npz"
            np.savez_compressed(p2, keys=keys,
                                targets=np.zeros((2, 14, 14), np.float32),
                                version=np.int64(4))
            self.assertEqual(load_geom_cache(p2), {})

    def test_dataset_aux_default_off_and_on(self) -> None:
        from research_v2.src.data.torch_dataset import PairedModificationDataset
        from research_v2.src.data.landmarks import GEOM_DIM
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(3):
                pp = Path(tmp) / f"f{i}.png"
                cv2.imwrite(str(pp), (np.random.RandomState(i).rand(112, 112, 3) * 255).astype("uint8"))
                paths.append(pp)
            labels = [0, 1, 1]
            geom = {str(p): np.random.rand(GEOM_DIM).astype(np.float32) for p in paths}
            # default: legacy 5-tuple (unchanged contract)
            self.assertEqual(len(PairedModificationDataset(paths, labels, grid=14)[0]), 5)
            # aux on: 6-tuple with the geom/visibility dict
            ds = PairedModificationDataset(paths, labels, grid=14,
                                           geom_targets=geom, return_aux=True)
            item = ds[0]
            self.assertEqual(len(item), 6)
            aux = item[5]
            self.assertEqual(set(aux), {"geom", "vis_clean", "vis_mod"})
            self.assertEqual(tuple(aux["geom"].shape), (GEOM_DIM,))
            self.assertEqual(tuple(aux["vis_mod"].shape), (14, 14))
            self.assertTrue(bool((aux["vis_clean"] == 1).all()))
            self.assertTrue(bool((aux["vis_mod"] >= 0).all() and (aux["vis_mod"] <= 1).all()))

    def test_visibility_gating_identity_at_init(self) -> None:
        import torch
        from research_v2.src.novel.region_attention import RegionAwareTokenAttention
        torch.manual_seed(3)
        a = RegionAwareTokenAttention(channels=768, dim=256, grid=14, use_visibility=False).eval()
        torch.manual_seed(3)
        b = RegionAwareTokenAttention(channels=768, dim=256, grid=14, use_visibility=True).eval()
        feat = torch.randn(2, 768, 14, 14)
        with torch.no_grad():
            oa, _ = a(feat)
            ob, _ = b(feat)
        # zero-init visibility head -> 0.5 everywhere -> gated pooling == ungated
        self.assertTrue(torch.allclose(oa, ob, atol=1e-6))

    def test_mdie_bgi_visibility_losses_flow(self) -> None:
        import torch
        from research_v2.src.novel.mdie import MDIE
        from research_v2.src.data.landmarks import GEOM_DIM
        torch.manual_seed(0)
        m = MDIE(n_identity_classes=12, n_modification_classes=10, embedding_dim=512,
                 use_region_prior=True, use_amd=True, pretrained_backbone=False,
                 backbone="ir50", use_bgi=True, use_visibility=True).train()
        B = 3
        out = m(torch.randn(B, 3, 112, 112), torch.randint(0, 12, (B,)),
                torch.randint(0, 10, (B,)),
                bone_target=torch.rand(B, 14, 14),
                geom_target=torch.rand(B, GEOM_DIM),
                visibility=(torch.rand(B, 14, 14) > 0.3).float())
        for k in ("loss_bgi", "loss_vis"):
            self.assertIn(k, out)
            self.assertTrue(torch.isfinite(out[k]).item())
        (out["loss_bgi"] + out["loss_vis"]).backward()
        self.assertGreater(float(m.bgi_head[-1].weight.grad.norm()), 0.0)
        self.assertGreater(float(m.attn.vis_head[-1].weight.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
