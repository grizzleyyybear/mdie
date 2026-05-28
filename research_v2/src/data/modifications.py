"""
Synthetic, protocol-faithful modification generators.

These cover the seven modification families documented in the project plan
(plastic surgery, disguise, occlusion, aging, lighting, compression, adversarial)
with reproducible, deterministic implementations that do not require external
generative models. Used both for failure-mode analysis (Stage 1) and for
training the modification-disentanglement objective in the novel method.
"""

from __future__ import annotations

from typing import Callable, Dict

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public registry — keep order stable; index used as modification class label
# ---------------------------------------------------------------------------

MODIFICATION_TYPES = [
    "clean",
    "surgery_nose",     # rhinoplasty-like nasal warp
    "surgery_jaw",      # mandibular contour change
    "disguise_glasses",
    "disguise_mask",    # COVID-style mouth/nose covering
    "occlusion_random", # random rectangular occluder
    "aging",            # wrinkles + skin desaturation
    "low_light",        # gamma + blue-shift
    "adversarial",      # PGD-like additive perturbation (clipped)
]

assert len(MODIFICATION_TYPES) == 9


# ---------------------------------------------------------------------------
# Individual modifiers (input/output: HxWx3 uint8 BGR or RGB; channel-agnostic)
# ---------------------------------------------------------------------------

def _identity(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    return img.copy()


def _surgery_nose(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Pinch the nasal bridge inward and shorten the tip — mimics rhinoplasty."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h * 0.55
    radius = min(h, w) * 0.20
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx; dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    inside = np.exp(-(r ** 2) / (2 * (radius * 0.6) ** 2))     # Gaussian falloff
    # squeeze x toward center, lift y slightly
    map_x = xx - 0.20 * dx * inside
    map_y = yy - 1.5 * inside
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT_101)


def _surgery_jaw(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Contract the jawline inward — mimics mandibular reduction."""
    h, w = img.shape[:2]
    cx = w / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Squeeze horizontally, with strength concentrated at jaw line (y ~ 0.78h)
    jaw_band = np.exp(-((yy / h - 0.82) ** 2) / (2 * 0.10 ** 2))
    map_x = xx - 0.18 * (xx - cx) * jaw_band
    map_y = yy
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT_101)


def _piecewise_warp(img, src, dst):
    """Kept for backwards compatibility — unused after surgery_* simplification."""
    return img


def _disguise_glasses(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    eye_y = int(h * 0.42)
    r = int(w * 0.12)
    cv2.ellipse(out, (int(w * 0.36), eye_y), (r, int(r * 0.65)), 0, 0, 360, (15, 15, 15), -1)
    cv2.ellipse(out, (int(w * 0.64), eye_y), (r, int(r * 0.65)), 0, 0, 360, (15, 15, 15), -1)
    cv2.line(out, (int(w * 0.36) + r, eye_y), (int(w * 0.64) - r, eye_y), (15, 15, 15), 3)
    return out


def _disguise_mask(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
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
    out = np.power(out, 2.2)        # gamma darken
    out *= np.array([0.85, 0.85, 1.10])  # blue tint
    out += rng.randn(*out.shape).astype(np.float32) * 0.015
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def _adversarial(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    # PGD-style ε=8/255 random sign perturbation (model-agnostic stand-in)
    eps = 8
    delta = rng.randint(-eps, eps + 1, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)


_DISPATCH: Dict[str, Callable] = {
    "clean": _identity,
    "surgery_nose": _surgery_nose,
    "surgery_jaw": _surgery_jaw,
    "disguise_glasses": _disguise_glasses,
    "disguise_mask": _disguise_mask,
    "occlusion_random": _occlusion_random,
    "aging": _aging,
    "low_light": _low_light,
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
