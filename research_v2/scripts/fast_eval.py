"""Fast GPU-only re-evaluation of the retrained (fused-512) MDIE variants and
baselines on the new occlusion+lighting taxonomy.

Reproduces the EXACT held-out split + verification-pair protocol used by
run_stage2 (RandomState(0) identity permutation, 80/20, build_verification_pairs
seed=42), evaluates the deployed single-512-d fused embedding per modification,
and writes stage2_metrics.json plus a security-niche family summary. The slow
CPU insightface reference is skipped on purpose (cite the value from the prior
full run); every model here runs on the GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from research_v2.src.config import SETTINGS, CKPT_DIR, RESULTS_DIR, get_device
from research_v2.src.baselines import build_baseline
from research_v2.src.data import build_face_dataset, prepare_lfw, build_verification_pairs
from research_v2.src.config import DATA_DIR
from research_v2.src.data.modifications import (MODIFICATION_TYPES, OCCLUSION_TYPES,
                                               LIGHTING_TYPES)
from research_v2.src.novel import MDIE
from research_v2.src.run_stage2 import _enc, _eval_all_mods

N_PAIRS = 3000
device = get_device()


def _load_mdie(name, n_id):
    ck = CKPT_DIR / f"{name}.pt"
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = sd.get("model_config") or {}
    st = sd.get("model", sd)
    use_region = "norata" not in name
    m = MDIE(n_identity_classes=n_id,
             n_modification_classes=len(MODIFICATION_TYPES),
             embedding_dim=SETTINGS.train.embedding_dim,
             use_region_prior=use_region,
             pretrained_backbone=True)
    st = {k.replace("_orig_mod.", ""): v for k, v in st.items()}
    st = {k: v for k, v in st.items()
          if not k.startswith(("mod_head.", "identity_head."))}
    missing, unexpected = m.load_state_dict(st, strict=False)
    bad = [k for k in missing if k.startswith(("backbone.", "fuse.", "attn.",
                                               "post_pool."))]
    if bad:
        print(f"    [warn] {name}: {len(bad)} core keys missing e.g. {bad[:3]}")
    return m.to(device).eval()


def main():
    SETTINGS.eval.n_pos_pairs = N_PAIRS
    SETTINGS.eval.n_neg_pairs = N_PAIRS
    lfw_dir = prepare_lfw(DATA_DIR, min_faces_per_person=8)
    paths, labels, names = build_face_dataset(lfw_dir, min_imgs=4)
    n_classes = len(names)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n_classes)
    train_ids = set(perm[: int(0.8 * n_classes)].tolist())
    test_ids = set(perm[int(0.8 * n_classes):].tolist())
    remap = {l: i for i, l in enumerate(sorted(train_ids))}
    n_id = len(remap)
    test_paths = [p for p, l in zip(paths, labels) if l in test_ids]
    test_lbls = [l for l in labels if l in test_ids]
    pair_set = build_verification_pairs(test_paths, test_lbls,
                                        n_pos=N_PAIRS, n_neg=N_PAIRS, seed=42)
    print(f"held-out ids={len(test_ids)}  pairs={len(pair_set)}  n_id_train={n_id}")

    results = {}

    for bname in ("facenet", "arcface", "cosface", "mobilefacenet"):
        ck = CKPT_DIR / f"baseline_{bname}.pt"
        if not ck.exists():
            print(f"  [skip] baseline {bname}"); continue
        sd_b = torch.load(ck, map_location="cpu", weights_only=False)
        if isinstance(sd_b, dict) and "model" in sd_b:
            sd_b = sd_b["model"]
        n_cls = n_id
        if "head.W" in sd_b: n_cls = int(sd_b["head.W"].shape[0])
        elif "head.weight" in sd_b: n_cls = int(sd_b["head.weight"].shape[0])
        model = build_baseline(bname, n_classes=n_cls,
                               embedding_dim=SETTINGS.train.embedding_dim)
        model.load_state_dict(sd_b, strict=False)
        model.to(device).eval()
        results[bname] = _eval_all_mods(_enc(model, device, is_mdie=False),
                                        pair_set, device)
        print(f"  [done] {bname}  pooled AUC={results[bname]['pooled']['auc']:.4f}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    name_map = {"MDIE-full": "mdie_full", "MDIE-noRATA": "mdie_norata",
                "MDIE-noAMD": "mdie_noamd", "MDIE-noICCL": "mdie_noiccl"}
    for disp, ck in name_map.items():
        if not (CKPT_DIR / f"{ck}.pt").exists():
            print(f"  [skip] {disp}"); continue
        model = _load_mdie(ck, n_id)
        results[disp] = _eval_all_mods(_enc(model, device, is_mdie=True),
                                       pair_set, device)
        print(f"  [done] {disp}  pooled AUC={results[disp]['pooled']['auc']:.4f}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # persist (json-safe: drop fpr/tpr arrays)
    safe = {m: {mod: {k: v for k, v in d.items() if k not in ("fpr", "tpr")}
                for mod, d in r.items()} for m, r in results.items()}
    (RESULTS_DIR / "stage2_metrics.json").write_text(json.dumps(safe, indent=2),
                                                      encoding="utf-8")

    def fam(model, group):
        return float(np.mean([results[model][k]["auc"] for k in group]))

    print("\n== security-niche family AUC (held-out, single 512-d fused) ==")
    print(f"{'model':16s} {'clean':>7s} {'occlusn':>8s} {'lighting':>9s} {'pooled':>7s}")
    order = ["facenet", "arcface", "cosface", "mobilefacenet",
             "MDIE-noRATA", "MDIE-noAMD", "MDIE-noICCL", "MDIE-full"]
    rows = {}
    for m in order:
        if m not in results: continue
        cl = results[m]["clean"]["auc"]; occ = fam(m, OCCLUSION_TYPES)
        lit = fam(m, LIGHTING_TYPES); pl = results[m]["pooled"]["auc"]
        rows[m] = dict(clean=cl, occlusion=occ, lighting=lit, pooled=pl)
        print(f"{m:16s} {cl:7.4f} {occ:8.4f} {lit:9.4f} {pl:7.4f}")

    (RESULTS_DIR / "security_family_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n[done] wrote stage2_metrics.json + security_family_summary.json")


if __name__ == "__main__":
    main()
