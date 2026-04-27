"""Thin wrappers that call easyasc kernels with the same positional signature
as the pure-PyTorch impl, and return a tensor of identical shape/dtype so the
swap is transparent at the call site.

Each adapter:
    * Accepts the exact torch positional args.
    * Accepts a trailing keyword-only ``_config: DispatchConfig`` from the
      dispatcher and reads ``kernel_simulator`` (and any future runtime knobs)
      from it.
    * Locally imports the kernel module and ``easyasc`` so an environment
      without a working easyasc install still allows the torch path to run.
    * Handles device moves if simulator mode requires CPU tensors.

Adding a new adapter:
    1. Write a function ``kernel_<op>(...)`` below that follows the torch
       signature and the contract above.
    2. Add a ``registry.register_kernel("<op>", kernel_<op>)`` line inside
       ``register_all`` at the bottom.
    3. Make sure ``use_kernel_<op>`` exists in ``config.DispatchConfig`` (add
       it if not).
"""

from __future__ import annotations

from typing import Any

import torch


def _resolve_simulator(_config: Any) -> bool:
    """Read the ``kernel_simulator`` bool off an optional ``_config`` without
    assuming it is a real ``DispatchConfig`` (tests / ad-hoc callers may pass
    None or a plain object)."""
    if _config is None:
        return True
    return bool(getattr(_config, "kernel_simulator", True))


# --- build_rotation ------------------------------------------------------

def kernel_build_rotation(r: torch.Tensor, *, _config: Any = None) -> torch.Tensor:
    """Kernel adapter for ``EWA_fully_fused_proj_packed.build_rotation``.

    Torch signature:  r: [N, 4]  ->  R: [N, 3, 3]
    The kernel writes a flat [N, 9] buffer (one row per rotation, row-major);
    we reshape it to [N, 3, 3] so callers see the same shape as the torch impl.
    """
    # Imported lazily: if easyasc or the kernel file is broken, only the
    # kernel path fails, not the import of the whole dispatch package.
    from build_rotation import build_rotation_kernel, OUT_COLS
    from easyasc.a5 import OpExec

    simulator = _resolve_simulator(_config)
    src_device = r.device

    # Simulator path wants CPU fp32.
    r_cpu = r.detach().to(dtype=torch.float32).cpu().contiguous()
    R_flat = torch.zeros((r_cpu.shape[0], OUT_COLS), dtype=torch.float32)
    R_flat = OpExec(build_rotation_kernel, simulator=simulator)(r_cpu, R_flat, r_cpu.shape[0])

    R = R_flat.view(r_cpu.shape[0], 3, 3)
    if R.device != src_device:
        R = R.to(src_device)
    return R


# --- registration --------------------------------------------------------

def register_all(registry) -> None:
    """Register every adapter defined above with ``registry``. Called once from
    ``dispatch/__init__.py`` at package import time. Add one line per new
    adapter."""
    registry.register_kernel("build_rotation", kernel_build_rotation)
    # Future adapters, e.g.:
    # registry.register_kernel("build_covariance_3d", kernel_build_covariance_3d)
