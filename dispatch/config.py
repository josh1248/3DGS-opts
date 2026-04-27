"""DispatchConfig: which op variants to use, plus shared kernel-side knobs.

One boolean per registered op. Default is False everywhere so the dispatcher
matches the existing pure-PyTorch behavior bit-for-bit until a flag is flipped.
Flip a single field to True once its kernel adapter is registered.

Keep this file boring: the only reason to touch it is to add a
``use_kernel_<op>`` field for a newly registered op.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Iterator


@dataclass
class DispatchConfig:
    # --- Projection / covariance building blocks ---
    use_kernel_build_rotation: bool = False
    use_kernel_build_scaling_rotation: bool = False
    use_kernel_build_covariance_3d: bool = False
    use_kernel_build_covariance_2d: bool = False
    use_kernel_projection_means2d_pinhole: bool = False
    use_kernel_inverse_cov2d: bool = False
    use_kernel_get_radius: bool = False
    use_kernel_get_rect: bool = False

    # --- Spherical harmonics ---
    use_kernel_eval_sh: bool = False
    use_kernel_build_color: bool = False

    # --- Tile intersection + rasterization ---
    use_kernel_compute_view_dirs_packed: bool = False
    use_kernel_isect_tiles: bool = False
    use_kernel_isect_offset_encode: bool = False
    use_kernel_rasterize_to_pixels: bool = False

    # --- Pipeline-level (for an eventual fully-fused kernel) ---
    use_kernel_fully_fused_projection_batch: bool = False

    # --- Runtime knobs shared by every kernel adapter ---
    # Forwarded to easyasc ``OpExec(simulator=...)``. Leave True if you do not
    # have a CANN toolkit / NPU available; the simulator runs on CPU.
    kernel_simulator: bool = True

    def with_updates(self, **kw) -> "DispatchConfig":
        """Return a copy of this config with the named fields overridden."""
        return replace(self, **kw)

    def kernel_flags(self) -> Iterator[str]:
        """Yield the names of every ``use_kernel_*`` field."""
        for f in fields(self):
            if f.name.startswith("use_kernel_"):
                yield f.name

    @classmethod
    def all_torch(cls) -> "DispatchConfig":
        """Explicit alias for the default: every op uses the PyTorch impl."""
        return cls()

    @classmethod
    def all_kernel(cls) -> "DispatchConfig":
        """Flip every ``use_kernel_*`` to True. Ops without a registered adapter
        warn and fall back to torch, so this is safe to use as a smoke test."""
        kw = {f.name: True for f in fields(cls) if f.name.startswith("use_kernel_")}
        return cls(**kw)
