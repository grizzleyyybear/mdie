from .embeddings import extract_embeddings_for_pairs, score_pairs
from .metrics import (
    compute_eer,
    compute_roc,
    compute_tar_at_far,
    summarize_run,
)
from .occlusion_sensitivity import region_sensitivity_map

__all__ = [
    "extract_embeddings_for_pairs",
    "score_pairs",
    "compute_eer",
    "compute_roc",
    "compute_tar_at_far",
    "summarize_run",
    "region_sensitivity_map",
]
