"""
Synthetic, protocol-faithful modification generators.

Scoped to the project's security / access-control niche: recognition that must
survive *worn occlusions* (glasses, mask, cap/hat, partial occluders) and
*adverse lighting* (low light, over-exposure, harsh directional shadow), plus
two general robustness factors relevant to enrolment-vs-probe gaps (aging) and
spoof-style perturbations (adversarial). The plastic-surgery family was
intentionally removed — it is out of scope for the security use case and added
no measurable benefit. All implementations are reproducible and deterministic
and require no external generative models. Used both for failure-mode analysis
(Stage 1) and for training the modification-disentanglement objective.
"""

from __future__ import annotations

import os
from typing import Callable, Dict

import cv2
import numpy as np


# Opt-in: stronger, more realistic worn-mask synthesis for training-time
# augmentation. OFF by default so the validated LFW pipeline is byte-identical;
# enable with MDIE_REALISTIC_MASKS=1 (or pass realistic=True) for CASIA-scale runs.
def _realistic_masks_enabled() -> bool:
    return os.environ.get("MDIE_REALISTIC_MASKS", "0").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Public registry — keep order stable; index used as modification class label
# ---------------------------------------------------------------------------

MODIFICATION_TYPES = [
    "clean",
    "disguise_glasses",  # eyeglasses / sunglasses (randomised opacity)
    "disguise_mask",     # COVID-style mouth/nose covering
    "disguise_cap",      # cap/hat covering forehead + brow ridge
    "occlusion_random",  # random rectangular occluder
    "low_light",         # gamma darken + blue-shift + sensor noise
    "over_exposure",     # blown highlights / strong frontal light
    "harsh_shadow",      # directional side/back light with a hard shadow ramp
    "aging",             # wrinkles + skin desaturation (enrol-vs-probe gap)
    "adversarial",       # PGD-like additive perturbation (clipped)
]

assert len(MODIFICATION_TYPES) == 10

# Reporting groups used by the security niche evaluation.
OCCLUSION_TYPES = ["disguise_glasses", "disguise_mask", "disguise_cap",
                   "occlusion_random"]
LIGHTING_TYPES = ["low_light", "over_exposure", "harsh_shadow"]


# ---------------------------------------------------------------------------
# Individual modifiers (input/output: HxWx3 uint8 BGR or RGB; channel-agnostic)
# ---------------------------------------------------------------------------

def _identity(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    return img.copy()


def _disguise_glasses(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Eyeglasses or sunglasses. Opacity is randomised so the class spans clear
    prescription frames (faint, mostly the rims) through fully opaque sunglasses
    that hide the orbital region entirely — the hard case for the eye landmarks."""
    out = img.copy()
    h, w = img.shape[:2]
    eye_y = int(h * (0.40 + 0.04 * rng.rand()))
    r = int(w * (0.11 + 0.02 * rng.rand()))
    lx, rx = int(w * 0.36), int(w * 0.64)
    dark = rng.rand() < 0.5                      # sunglasses vs clear frames
    lens = (20, 20, 20) if dark else (70, 70, 75)
    overlay = out.copy()
    cv2.ellipse(overlay, (lx, eye_y), (r, int(r * 0.7)), 0, 0, 360, lens, -1)
    cv2.ellipse(overlay, (rx, eye_y), (r, int(r * 0.7)), 0, 0, 360, lens, -1)
    fill = 0.95 if dark else 0.45
    cv2.addWeighted(overlay, fill, out, 1 - fill, 0, out)
    # rims + bridge + temple arms (always solid, dark)
    frame = (10, 10, 10)
    cv2.ellipse(out, (lx, eye_y), (r, int(r * 0.7)), 0, 0, 360, frame, 2)
    cv2.ellipse(out, (rx, eye_y), (r, int(r * 0.7)), 0, 0, 360, frame, 2)
    cv2.line(out, (lx + r, eye_y), (rx - r, eye_y), frame, 3)
    cv2.line(out, (lx - r, eye_y), (0, eye_y - int(r * 0.2)), frame, 2)
    cv2.line(out, (rx + r, eye_y), (w, eye_y - int(r * 0.2)), frame, 2)
    return out


def _disguise_mask(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    if _realistic_masks_enabled():
        return _disguise_mask_realistic(img, rng)
    out = img.copy()
    h, w = img.shape[:2]
    pts = np.array([
        [int(w * 0.18), int(h * 0.55)],
        [int(w * 0.82), int(h * 0.55)],
        [int(w * 0.92), int(h * 0.78)],
        [int(w * 0.74), int(h * 0.96)],
        [int(w * 0.26), int(h * 0.96)],
        [int(w * 0.08), int(h * 0.78)],
    ], dtype=np.int32)
    colors = [(220, 220, 220), (110, 130, 200), (50, 70, 110)]
    color = colors[rng.randint(len(colors))]
    cv2.fillPoly(out, [pts], tuple(int(c) for c in color))
    return out


def _disguise_mask_realistic(img: np.ndarray,
                             rng: np.random.RandomState) -> np.ndarray:
    """MaskTheFace-style worn surgical/cloth mask: a curved top edge that rides
    over the nose bridge, randomised vertical coverage, fabric pleats and ear
    loops, plus mild shading. Opt-in (MDIE_REALISTIC_MASKS=1) so the default
    pipeline is unchanged. Deterministic in ``rng``; preserves the 5-tuple +
    bone-target contract (it only paints the lower face like the flat variant)."""
    out = img.copy()
    h, w = img.shape[:2]
    # Randomised geometry: how high the mask rides and how wide it sits.
    top = 0.46 + 0.10 * rng.rand()          # nose-bridge line
    side = 0.06 + 0.06 * rng.rand()         # how far it wraps to the cheeks
    bridge_dip = 0.04 + 0.05 * rng.rand()   # curve depth over the nose
    cx = w * 0.5
    # Curved top edge (samples a shallow parabola dipping at the nose bridge).
    top_edge = []
    for t in np.linspace(0.0, 1.0, 11):
        x = w * (side + (1.0 - 2.0 * side) * t)
        dip = bridge_dip * (1.0 - ((x - cx) / (w * 0.5)) ** 2)
        top_edge.append([int(x), int(h * (top + max(dip, 0.0)))])
    bottom = [
        [int(w * (1.0 - side)), int(h * 0.74)],
        [int(w * 0.74), int(h * 0.97)],
        [int(cx), int(h * 0.99)],
        [int(w * 0.26), int(h * 0.97)],
        [int(w * side), int(h * 0.74)],
    ]
    pts = np.array(top_edge + bottom, dtype=np.int32)
    fabrics = [(232, 232, 232), (120, 140, 205), (60, 80, 120),
               (180, 200, 210), (40, 55, 75)]
    color = fabrics[rng.randint(len(fabrics))]
    cv2.fillPoly(out, [pts], tuple(int(c) for c in color))
    # Horizontal pleats: a few subtly darker bands across the lower face.
    y0, y1 = int(h * (top + 0.10)), int(h * 0.94)
    shade = tuple(int(c * 0.82) for c in color)
    for yy in range(y0, y1, max(int(h * 0.07), 2)):
        cv2.line(out, (int(w * (side + 0.02)), yy),
                 (int(w * (1.0 - side - 0.02)), yy), shade, 1)
    # Ear loops: thin arcs from the upper mask corners toward the ears.
    loop = tuple(int(c * 0.5) for c in color)
    ly = int(h * (top + 0.02))
    cv2.line(out, (int(w * side), ly), (0, int(h * (top - 0.06))), loop, 2)
    cv2.line(out, (int(w * (1.0 - side)), ly),
             (w, int(h * (top - 0.06))), loop, 2)
    return out



def _disguise_cap(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Cap/hat covering the forehead and brow ridge — tests whether identity
    survives when the upper-face bone landmarks are hidden (forcing reliance on
    cheekbones, orbital rims and jaw)."""
    out = img.copy()
    h, w = img.shape[:2]
    brim_y = int(h * (0.26 + 0.06 * rng.rand()))     # lower edge of the cap
    # crown: filled band across the top of the head/forehead
    crown = np.array([
        [0, 0], [w, 0], [w, brim_y],
        [int(w * 0.82), int(brim_y * 0.86)],
        [int(w * 0.5), int(brim_y * 0.78)],
        [int(w * 0.18), int(brim_y * 0.86)],
        [0, brim_y],
    ], dtype=np.int32)
    colors = [(40, 40, 40), (30, 50, 90), (90, 40, 40), (60, 70, 70)]
    color = colors[rng.randint(len(colors))]
    cv2.fillPoly(out, [crown], tuple(int(c) for c in color))
    # brim: a darker horizontal bar just below the crown
    cv2.rectangle(out, (int(w * 0.06), brim_y),
                  (int(w * 0.94), int(brim_y + h * 0.05)),
                  tuple(int(c * 0.6) for c in color), -1)
    return out


def _occlusion_random(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    rh = rng.randint(int(h * 0.18), int(h * 0.42))
    rw = rng.randint(int(w * 0.18), int(w * 0.42))
    y0 = rng.randint(0, h - rh)
    x0 = rng.randint(0, w - rw)
    out[y0:y0+rh, x0:x0+rw] = rng.randint(0, 255, (rh, rw, 3), dtype=np.uint8)
    return out


def _aging(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    out = img.astype(np.float32)
    # desaturate
    gray = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = 0.55 * out + 0.45 * gray[..., None]
    # add high-frequency wrinkle noise mostly in mid face
    h, w = img.shape[:2]
    noise = rng.randn(h, w).astype(np.float32) * 12
    noise = cv2.GaussianBlur(noise, (3, 3), 0.6)
    mask = np.zeros((h, w), np.float32)
    mask[int(h*0.3):int(h*0.85)] = 1.0
    mask = cv2.GaussianBlur(mask, (31, 31), 10)
    out += (noise * mask)[..., None]
    # mild yellow shift
    out[..., 0] *= 0.97
    out[..., 2] *= 1.03
    return np.clip(out, 0, 255).astype(np.uint8)


def _low_light(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    out = img.astype(np.float32) / 255.0
    out = np.power(out, 2.2 + 0.4 * rng.rand())   # gamma darken (varied)
    out *= np.array([0.85, 0.85, 1.10])           # blue tint
    out += rng.randn(*out.shape).astype(np.float32) * 0.02   # sensor noise
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def _over_exposure(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Strong frontal light: lifted gamma, blown highlights, mild warm tint and a
    soft bloom — the high-key counterpart of low_light for lighting robustness."""
    out = img.astype(np.float32) / 255.0
    out = np.power(out, 0.45 + 0.1 * rng.rand())          # brighten
    out *= (1.25 + 0.15 * rng.rand())                     # global gain -> clip
    out *= np.array([1.06, 1.02, 0.95])                   # warm tint
    bloom = cv2.GaussianBlur(np.clip(out, 0, 1), (0, 0), 4)
    hi = (out > 0.85).astype(np.float32)
    out = out * (1 - 0.3 * hi) + bloom * (0.3 * hi)       # bloom in highlights
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def _harsh_shadow(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Hard directional lighting: one side (or top/bottom) of the face is brightly
    lit while the other falls into shadow, with a sharp transition ramp. Models
    side/back-lit CCTV and outdoor sun — a core security failure mode."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    angle = rng.uniform(0, np.pi)
    proj = np.cos(angle) * (xx / w - 0.5) + np.sin(angle) * (yy / h - 0.5)
    edge = rng.uniform(-0.15, 0.15)
    sharp = 9.0 + 6.0 * rng.rand()
    ramp = 1.0 / (1.0 + np.exp(-sharp * (proj - edge)))   # 0..1 across the face
    lit, shadow = 1.35, 0.35
    gain = (shadow + (lit - shadow) * ramp)[..., None]
    out = img.astype(np.float32) * gain
    out += rng.randn(h, w, 3).astype(np.float32) * 2.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _adversarial(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    # PGD-style eps=8/255 random sign perturbation (model-agnostic stand-in)
    eps = 8
    delta = rng.randint(-eps, eps + 1, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)


_DISPATCH: Dict[str, Callable] = {
    "clean": _identity,
    "disguise_glasses": _disguise_glasses,
    "disguise_mask": _disguise_mask,
    "disguise_cap": _disguise_cap,
    "occlusion_random": _occlusion_random,
    "low_light": _low_light,
    "over_exposure": _over_exposure,
    "harsh_shadow": _harsh_shadow,
    "aging": _aging,
    "adversarial": _adversarial,
}


def apply_modification(img: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    """Apply a named modification to ``img`` (HxWx3 uint8). Deterministic per ``seed``."""
    if kind not in _DISPATCH:
        raise ValueError(f"unknown modification {kind!r}; valid: {MODIFICATION_TYPES}")
    rng = np.random.RandomState(seed)
    return _DISPATCH[kind](img, rng)


class ModificationApplier:
    """Stateful applier that picks a random modification per call."""

    def __init__(self, types=None, seed: int = 0):
        self.types = list(types) if types is not None else list(MODIFICATION_TYPES)
        self.rng = np.random.RandomState(seed)

    def sample_kind(self) -> str:
        return self.types[self.rng.randint(0, len(self.types))]

    def __call__(self, img: np.ndarray, kind: str | None = None):
        if kind is None:
            kind = self.sample_kind()
        seed = int(self.rng.randint(0, 1_000_000))
        out = apply_modification(img, kind, seed=seed)
        return out, kind
