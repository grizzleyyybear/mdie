"""Per-face rigid bone-landmark targets for MDIE attention supervision.

This module turns a face image into a 7x7 *attention target* concentrated on
the rigid skeletal landmarks that survive facial alterations (brow ridge,
glabella, nasion, nasal bridge, orbital rims, zygomatic cheekbones, mandible
jaw angles, chin). Soft, easily-altered tissue (nose tip, lips/mouth, and the
ear-edge jaw endpoints) is deliberately excluded.

The landmarks are obtained from a real per-face detector (InsightFace 68-pt),
so the target follows each individual's actual bone geometry and pose — unlike
a fixed canonical template. The detector is used **only at training time** to
build the supervision target (a teacher); the MDIE encoder learns to reproduce
the bone-focused attention from image features alone, so inference needs no
landmark detector.

Cache format (npz): ``keys`` = array of image-path strings, ``targets`` =
(N, grid, grid) float32 each summing to 1. Use :func:`load_target_cache`.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

GRID = 14

# Bump this whenever the rigid-point set, their placement, or POINT_WEIGHTS
# change so a stale on-disk target cache is automatically rebuilt.
TARGET_VERSION = 4


def configure_onnxruntime_threads() -> None:
    """Force onnxruntime to use an explicit intra-op thread count.

    On cluster compute nodes (e.g. PARAM Siddhi-AI) the process runs inside a
    cgroup whose allowed CPU set is not the contiguous ``0..N-1`` range that
    onnxruntime assumes, so its default per-thread affinity pinning fails with
    ``EINVAL`` and floods stderr with thousands of ``pthread_setaffinity_np
    failed`` lines. Specifying the thread count explicitly disables that
    affinity pinning (per onnxruntime's own hint in the warning).

    insightface forwards only ``providers``/``provider_options`` to
    ``InferenceSession``, so we inject a ``SessionOptions`` globally through a
    one-time, idempotent monkeypatch on ``InferenceSession.__init__``. The patch
    only adds ``sess_options`` when the caller did not supply one, so it never
    overrides explicit configuration.
    """
    try:
        import onnxruntime as ort
    except Exception:
        return
    if getattr(ort.InferenceSession, "_mdie_threads_patched", False):
        return
    _orig_init = ort.InferenceSession.__init__
    n_threads = max(1, int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count() or 4))

    def _patched_init(self, *args, **kwargs):
        if kwargs.get("sess_options") is None and (len(args) < 2 or args[1] is None):
            so = ort.SessionOptions()
            so.intra_op_num_threads = n_threads
            so.inter_op_num_threads = 1
            kwargs["sess_options"] = so
        _orig_init(self, *args, **kwargs)

    ort.InferenceSession.__init__ = _patched_init
    ort.InferenceSession._mdie_threads_patched = True


# --- iBUG-68 rigid bone extraction -----------------------------------------
# Index reference (iBUG/dlib 68): 0-16 jaw contour (8=chin), 17-21 right brow,
# 22-26 left brow, 27-30 nose bridge (27=nasion), 31-35 nose base (soft),
# 36-41 right eye, 42-47 left eye, 48-67 mouth (soft).
def rigid_bone_points(lm: np.ndarray) -> dict[str, np.ndarray]:
    """Map 68 detected landmarks → named rigid bone points (image pixels).

    ``lm`` is (68, 2) or (68, 3) — only x, y are used. Returns a dict of
    ``name -> (x, y)``. Soft tissue (nose tip 30-35, mouth 48-67) and the
    ear-edge jaw endpoints (0, 1, 15, 16) are intentionally omitted; the
    zygomatic cheekbone is derived (not a native 68-pt landmark).
    """
    p = lm[:, :2].astype(np.float32)
    mid = lambda a, b: (p[a] + p[b]) / 2.0
    pts: dict[str, np.ndarray] = {
        # brow ridge (frontal bone / supraorbital)
        "brow_R_out": p[17], "brow_R": p[19], "brow_R_in": p[21],
        "brow_L_in": p[22], "brow_L": p[24], "brow_L_out": p[26],
        # glabella + nasion + bony nasal bridge (NOT the soft tip)
        "glabella": mid(21, 22), "nasion": p[27], "nose_bridge": p[28],
        # orbital rims (medial/lateral canthi sit on the bony rim)
        "orbital_R_out": p[36], "orbital_R_in": p[39],
        "orbital_L_in": p[42], "orbital_L_out": p[45],
        # zygomatic cheekbone — derived: from the outer eye corner toward the
        # UPPER jaw/temple contour, biased to sit high on the bony zygomatic
        # prominence (just below & lateral to the outer canthus), not the soft
        # mid-cheek. A second, lower point traces the zygomatic arch.
        "cheek_R": p[36] + 0.34 * (p[2] - p[36]),
        "cheek_L": p[45] + 0.34 * (p[14] - p[45]),
        "cheek_R_low": p[36] + 0.52 * (p[3] - p[36]),
        "cheek_L_low": p[45] + 0.52 * (p[13] - p[45]),
        # mandible jaw angle (gonial region)
        "jaw_R": p[4], "jaw_R_low": p[5],
        "jaw_L": p[12], "jaw_L_low": p[11],
        # chin (mental protuberance)
        "chin": p[8], "chin_R": p[7], "chin_L": p[9],
    }
    return pts


# colour-coded groups for overlays (region -> (point-names, hex colour))
RIGID_GROUPS = {
    "jaw/chin": (["jaw_R", "jaw_R_low", "jaw_L", "jaw_L_low",
                  "chin", "chin_R", "chin_L"], "#ff3b3b"),
    "cheekbone": (["cheek_R", "cheek_L", "cheek_R_low", "cheek_L_low"], "#ff9f1c"),
    "brow ridge": (["brow_R_out", "brow_R", "brow_R_in",
                    "brow_L_in", "brow_L", "brow_L_out", "glabella"], "#ffe119"),
    "nose bridge": (["nasion", "nose_bridge"], "#2ec4b6"),
    "orbital rim": (["orbital_R_out", "orbital_R_in",
                     "orbital_L_in", "orbital_L_out"], "#3a86ff"),
}

# Per-landmark target weights. The recognition goal is to anchor on the rigid
# bone structure as MULTIPLE DISTRIBUTED POINTS — brow ridge, orbital rims,
# nasal bridge, cheekbones, jaw angles and chin should ALL receive attention.
# Weights are kept close to uniform (every rigid bone matters) with a GENTLE
# outward gradient: the distinctive, alteration-invariant PERIPHERAL anchors
# (jaw angle, chin, zygomatic prominence, lateral orbital rim, outer brow) get
# a mild boost (~1.2-1.25), while the dead-centre nasal bridge — a transformer
# "attention sink" that otherwise hogs the single largest weight on every face —
# is gently de-emphasised (~0.8) WITHOUT being removed. This breaks the central
# collapse and spreads attention across each face's whole bony scaffold instead
# of one shared midline point.
POINT_WEIGHTS = {
    # jaw angles + chin (alteration-invariant peripheral anchors)
    "jaw_R": 1.25, "jaw_R_low": 1.1, "jaw_L": 1.25, "jaw_L_low": 1.1,
    "chin": 1.2, "chin_R": 1.05, "chin_L": 1.05,
    # zygomatic cheekbones (high prominence + arch)
    "cheek_R": 1.25, "cheek_L": 1.25, "cheek_R_low": 1.1, "cheek_L_low": 1.1,
    # brow ridge (frontal bone) — outer/lateral emphasised, inner kept in play
    "brow_R_out": 1.2, "brow_L_out": 1.2,
    "brow_R": 1.0, "brow_L": 1.0, "brow_R_in": 1.0, "brow_L_in": 1.0,
    "glabella": 0.9,
    # orbital rims — lateral canthi sit on distinctive bone
    "orbital_R_out": 1.2, "orbital_L_out": 1.2,
    "orbital_R_in": 1.0, "orbital_L_in": 1.0,
    # bony nasal bridge — de-emphasised to break the central attention sink
    "nasion": 0.8, "nose_bridge": 0.8,
}


def build_bone_target(points_norm: np.ndarray, grid: int = GRID,
                      sigma2: float = 0.004,
                      weights: np.ndarray | None = None) -> np.ndarray:
    """Splat normalized (x, y) points onto a ``grid`` map summing to 1.

    ``points_norm`` is (K, 2) with x, y in [0, 1] (x→right, y→down). Each
    landmark contributes an *equal-mass* Gaussian (its own splat is normalised
    to unit sum before being scaled by ``weights[k]``). This is the crucial
    difference from a naive sum-of-Gaussians: without per-point normalisation
    the densely-clustered central bones (glabella / nasion / nasal bridge /
    inner brow / inner orbital all sit within a couple of cells) STACK into one
    tall central peak, so the supervised attention collapses to a single
    central blob that looks identical on every face. Giving every bone the same
    total mass spreads the target across the WHOLE bony scaffold — brow ridge,
    orbital rims, nasal bridge, cheekbones, jaw angles and chin each become a
    distinct, comparably-weighted anchor — which is exactly the distributed
    multi-point supervision the model is meant to learn. ``weights`` (K,) then
    applies only a gentle peripheral/centre emphasis on top of that balance.
    """
    centres = (np.arange(grid) + 0.5) / grid
    cy, cx = np.meshgrid(centres, centres, indexing="ij")       # (g, g)
    m = np.zeros((grid, grid), dtype=np.float32)
    for k, (x0, y0) in enumerate(points_norm):
        if not (0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0):
            continue
        w = 1.0 if weights is None else float(weights[k])
        g = np.exp(-(((cx - x0) ** 2 + (cy - y0) ** 2) / sigma2))
        gs = g.sum()
        if gs <= 1e-9:
            continue
        m += w * (g / gs)                                       # equal-mass per point
    s = m.sum()
    if s <= 1e-9:
        return np.zeros((grid, grid), dtype=np.float32)
    return (m / s).astype(np.float32)


def target_from_landmarks(lm68: np.ndarray, img_w: int, img_h: int,
                          grid: int = GRID) -> np.ndarray:
    """Full pipeline: 68 pixel landmarks → weighted rigid points → target."""
    pts = rigid_bone_points(lm68)
    names = list(pts.keys())
    arr = np.stack([pts[n] for n in names], axis=0).astype(np.float32)
    arr[:, 0] /= max(img_w, 1)
    arr[:, 1] /= max(img_h, 1)
    w = np.array([POINT_WEIGHTS.get(n, 1.0) for n in names], dtype=np.float32)
    return build_bone_target(arr, grid=grid, weights=w)


# --- cache I/O --------------------------------------------------------------
def load_cache_version(npz_path: str | Path) -> int:
    """Return the TARGET_VERSION stamped into a cache npz (0 if absent/old)."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        return 0
    try:
        data = np.load(npz_path, allow_pickle=True)
        return int(data["version"]) if "version" in data.files else 0
    except Exception:  # noqa: BLE001
        return 0


def load_target_cache(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load a path→(grid,grid) target dict from an npz produced by build_cache."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        return {}
    data = np.load(npz_path, allow_pickle=True)
    keys = [str(k) for k in data["keys"]]
    targets = data["targets"]
    return {k: targets[i] for i, k in enumerate(keys)}


def build_cache(paths, out_npz: str | Path, grid: int = GRID,
                det_size: int = 160, verbose: bool = True) -> dict[str, np.ndarray]:
    """Detect 68 landmarks for every image in ``paths`` and cache bone targets.

    Uses InsightFace buffalo_l (detection + landmark_3d_68) on CPU. Images with
    no detected face get an all-zero target (the attention loss ignores those).
    """
    import cv2
    from insightface.app import FaceAnalysis

    configure_onnxruntime_threads()
    app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=["detection", "landmark_3d_68"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    keys, targets = [], []
    n_ok = 0
    paths = [str(p) for p in paths]
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        tgt = np.zeros((grid, grid), dtype=np.float32)
        if img is not None:
            faces = app.get(img)
            if faces:
                f = max(faces, key=lambda z: (z.bbox[2] - z.bbox[0]) *
                        (z.bbox[3] - z.bbox[1]))
                lm = getattr(f, "landmark_3d_68", None)
                if lm is not None:
                    h, w = img.shape[:2]
                    tgt = target_from_landmarks(lm, w, h, grid=grid)
                    if tgt.sum() > 0:
                        n_ok += 1
        keys.append(p)
        targets.append(tgt)
        if verbose and (i + 1) % 200 == 0:
            print(f"    landmarks {i+1}/{len(paths)}  (detected {n_ok})", flush=True)

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, keys=np.array(keys),
                        targets=np.stack(targets).astype(np.float32),
                        version=np.int64(TARGET_VERSION))
    if verbose:
        print(f"    cached {len(keys)} targets ({n_ok} with a detected face) "
              f"-> {out_npz}", flush=True)
    return {k: targets[i] for i, k in enumerate(keys)}
