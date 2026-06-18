"""Minimal, opt-in Distributed Data Parallel helpers.

Every function is a safe no-op when distributed training is not active (no
``init_process_group`` call / ``WORLD_SIZE <= 1``), so importing and calling
these never changes single-GPU or CPU behaviour. Multi-GPU is launched with
``torchrun`` (or ``srun``), which sets ``RANK`` / ``WORLD_SIZE`` /
``LOCAL_RANK`` in the environment.

Design notes:
- The training loop builds its optimizer from the *raw* model's parameters and
  keeps a reference to that raw module for all checkpoint I/O, so saved
  state-dicts never carry a ``module.`` prefix and stay loadable by the
  single-GPU eval path.
- The MDIE step runs two forwards (clean + modified) before a single backward
  because ICCL couples the two embeddings. ``wrap_ddp`` therefore enables
  ``static_graph=True``, the documented way to allow multiple forwards before
  one backward under DDP.
"""
from __future__ import annotations

import os

import torch


def ddp_is_available_and_requested() -> bool:
    """True when the environment was launched for distributed training."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_ddp() -> tuple[int, int, int]:
    """Initialise the default process group from torchrun/srun env vars.

    Returns ``(rank, world_size, local_rank)``. When ``WORLD_SIZE <= 1`` this is
    a no-op returning ``(0, 1, 0)`` so callers can use one code path.
    """
    if not ddp_is_available_and_requested():
        return 0, 1, 0
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://",
                                world_size=world_size, rank=rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp() -> None:
    """Destroy the process group if one is active. Safe to always call."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def is_initialized() -> bool:
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except Exception:  # noqa: BLE001
        return False


def get_rank() -> int:
    if is_initialized():
        import torch.distributed as dist
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if is_initialized():
        import torch.distributed as dist
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """True on rank 0, and always True when not running distributed."""
    return get_rank() == 0


def barrier() -> None:
    if is_initialized():
        import torch.distributed as dist
        dist.barrier()


def wrap_ddp(model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
    """Wrap ``model`` in DistributedDataParallel when a group is initialised.

    Returns ``model`` unchanged when not distributed. ``static_graph=True`` is
    required because the MDIE step does two forwards before one backward.
    """
    if not is_initialized():
        return model
    from torch.nn.parallel import DistributedDataParallel as DDP

    device_ids = [local_rank] if torch.cuda.is_available() else None
    return DDP(model, device_ids=device_ids,
               output_device=local_rank if torch.cuda.is_available() else None,
               static_graph=True, find_unused_parameters=False)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module from a DDP wrapper (or the model itself)."""
    return getattr(model, "module", model)
