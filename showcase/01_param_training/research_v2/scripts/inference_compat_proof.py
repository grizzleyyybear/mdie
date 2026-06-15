"""Inference-compatibility proof: the deployed MDIE embedding is a drop-in for an
ArcFace encoder.

It demonstrates, on real held-out faces, that verification uses:
  * a SINGLE forward pass (no flip-TTA, no multi-crop),
  * a SINGLE 512-d, L2-normalised vector per image,
  * plain cosine similarity for matching (identical call shape to ArcFace),
and that cosine == dot product on the unit sphere (so any ArcFace/InsightFace
matcher, FAISS inner-product index, or cosine threshold works unchanged).

Run from showcase/02_laptop_rtx3050 with PYTHONPATH=cwd:
    python <this>.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from research_v2.src.config import CKPT_DIR, RESULTS_DIR, DATA_DIR, SETTINGS, get_device
from research_v2.src.data import build_face_dataset, prepare_lfw
from research_v2.src.data.modifications import MODIFICATION_TYPES, apply_modification
from research_v2.src.novel import MDIE

device = get_device()
SZ = 112


def _load(path):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SZ, SZ))
    return img


def _to_tensor(img):
    x = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return x.to(device)


def load_mdie():
    ck = CKPT_DIR / "mdie_full.pt"
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    st = sd.get("model", sd)
    st = {k.replace("_orig_mod.", ""): v for k, v in st.items()}
    n_id = int(st["identity_head.W"].shape[0]) if "identity_head.W" in st else 173
    st = {k: v for k, v in st.items()
          if not k.startswith(("mod_head.", "identity_head."))}
    m = MDIE(n_identity_classes=n_id,
             n_modification_classes=len(MODIFICATION_TYPES),
             embedding_dim=SETTINGS.train.embedding_dim,
             pretrained_backbone=True)
    m.load_state_dict(st, strict=False)
    return m.to(device).eval()


@torch.no_grad()
def main():
    m = load_mdie()
    lfw_dir = prepare_lfw(DATA_DIR, min_faces_per_person=8)
    paths, labels, names = build_face_dataset(lfw_dir, min_imgs=4)
    sample = [_load(p) for p in paths[:8]]

    checks = {}

    # 1. single 512-d, unit-norm, single forward (encode_verify, tta off)
    x = torch.cat([_to_tensor(s) for s in sample], dim=0)
    emb = m.encode_verify(x)
    checks["embedding_shape"] = list(emb.shape)
    checks["is_512d"] = emb.shape[1] == 512
    norms = emb.norm(dim=1)
    checks["unit_norm_max_err"] = float((norms - 1).abs().max())
    checks["is_unit_norm"] = bool((norms - 1).abs().max() < 1e-4)

    # 2. encode_verify default == encode()[0] (one forward, no TTA inflation)
    enc_emb = m.encode(x)[0]
    checks["verify_equals_encode_max_err"] = float((emb - enc_emb).abs().max())
    checks["single_forward_no_tta"] = bool((emb - enc_emb).abs().max() < 1e-5)

    # 3. cosine == dot product on unit sphere (ArcFace matcher compatibility)
    cos = torch.nn.functional.cosine_similarity(emb[0:1], emb, dim=1)
    dot = (emb[0:1] * emb).sum(dim=1)
    checks["cosine_equals_dot_max_err"] = float((cos - dot).abs().max())
    checks["cosine_is_dot"] = bool((cos - dot).abs().max() < 1e-5)

    # 4. deterministic (eval mode, same input -> same vector)
    emb2 = m.encode_verify(x)
    checks["deterministic_max_err"] = float((emb - emb2).abs().max())
    checks["is_deterministic"] = bool((emb - emb2).abs().max() < 1e-5)

    # 5. occluded gallery/probe still matches on plain cosine (security use-case)
    base = sample[0]
    occ = apply_modification(base.copy(), "disguise_mask")
    e_clean = m.encode_verify(_to_tensor(base))
    e_mask = m.encode_verify(_to_tensor(occ))
    other = m.encode_verify(_to_tensor(sample[1]))
    same = float(torch.nn.functional.cosine_similarity(e_clean, e_mask).item())
    diff = float(torch.nn.functional.cosine_similarity(e_clean, other).item())
    checks["cos_same_id_masked"] = same
    checks["cos_diff_id"] = diff
    checks["masked_self_beats_imposter"] = bool(same > diff)

    passed = all(v for k, v in checks.items()
                 if k.startswith("is_") or k.startswith("single_") or
                 k.startswith("cosine_is") or k == "masked_self_beats_imposter")
    checks["ALL_PASS"] = bool(passed)

    out = RESULTS_DIR / "inference_compat_proof.json"
    out.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    print("== ArcFace-compatibility proof (deployed MDIE embedding) ==")
    print(f"  embedding shape          : {checks['embedding_shape']}  (512-d: {checks['is_512d']})")
    print(f"  L2 unit-norm             : {checks['is_unit_norm']}  (max err {checks['unit_norm_max_err']:.2e})")
    print(f"  single forward (no TTA)  : {checks['single_forward_no_tta']}  (max err {checks['verify_equals_encode_max_err']:.2e})")
    print(f"  cosine == dot product    : {checks['cosine_is_dot']}  (max err {checks['cosine_equals_dot_max_err']:.2e})")
    print(f"  deterministic            : {checks['is_deterministic']}  (max err {checks['deterministic_max_err']:.2e})")
    print(f"  masked self vs imposter  : {same:.4f} > {diff:.4f}  -> {checks['masked_self_beats_imposter']}")
    print(f"\n  ALL_PASS = {checks['ALL_PASS']}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
