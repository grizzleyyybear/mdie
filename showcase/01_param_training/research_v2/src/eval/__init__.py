from .embeddings import (
    compute_eer,
    compute_roc,
    compute_tar_at_far,
    extract_embeddings_for_pairs,
    quick_verification_auc,
    score_pairs,
    summarize_run,
)
__all__ = [
    "extract_embeddings_for_pairs",
    "quick_verification_auc",
    "score_pairs",
    "compute_eer",
    "compute_roc",
    "compute_tar_at_far",
    "summarize_run",
    "region_sensitivity_map",
]


def __getattr__(name):
    if name == "region_sensitivity_map":
        from .occlusion_sensitivity import region_sensitivity_map
        return region_sensitivity_map
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
