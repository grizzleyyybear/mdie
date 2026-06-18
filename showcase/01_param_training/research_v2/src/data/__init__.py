from .lfw import prepare_lfw, build_face_dataset
from .casia import prepare_casia, build_train_dataset
from .torch_dataset import (
    FaceClassificationDataset,
    IdentityBalancedSampler,
    PairSet,
    PairedModificationDataset,
    build_verification_pairs,
    make_loaders,
)

__all__ = [
    "prepare_lfw",
    "build_face_dataset",
    "prepare_casia",
    "build_train_dataset",
    "MODIFICATION_TYPES",
    "ModificationApplier",
    "apply_modification",
    "build_verification_pairs",
    "PairSet",
    "FaceClassificationDataset",
    "IdentityBalancedSampler",
    "PairedModificationDataset",
    "make_loaders",
]


def __getattr__(name):
    if name in {"MODIFICATION_TYPES", "ModificationApplier", "apply_modification"}:
        from . import modifications
        return getattr(modifications, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
