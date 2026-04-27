"""Config-driven router between pure-PyTorch reference impls (already in
``3DGS-opts``) and easyasc kernel variants (in progress).

Typical usage (``3DGS-opts`` must be on ``sys.path`` — ``view_scene.py`` sets
that up; when importing from elsewhere, do the same):

    from dispatch import ops, DispatchConfig, set_config

    # Default: every op uses pure PyTorch. No behavior change.
    R = ops.build_rotation(r)

    # Globally flip a single op to its easyasc kernel variant:
    set_config(DispatchConfig(use_kernel_build_rotation=True, kernel_simulator=True))
    R = ops.build_rotation(r)

    # Or scope the override:
    from dispatch import using_config
    with using_config(DispatchConfig(use_kernel_build_rotation=True)):
        R = ops.build_rotation(r)

    # Or per call:
    R = ops.build_rotation(r, _config=DispatchConfig(use_kernel_build_rotation=True))

See ``dispatch/README.md`` for the contract + "how to add a new kernel adapter".
"""

from __future__ import annotations

import warnings

from . import ops
from .config import DispatchConfig
from .registry import (
    dispatch,
    get_config,
    registry,
    set_config,
    using_config,
)


# --------------------------------------------------------------------- #
# Registration of torch impls. Each import block is guarded individually
# so a broken sibling module does not wipe out every op.
# --------------------------------------------------------------------- #

def _register_torch_impls() -> None:
    reg = registry()

    try:
        from EWA_fully_fused_proj_packed import (
            build_rotation,
            build_scaling_rotation,
            build_covariance_3d,
            build_covariance_2d,
            projection_means2d_pinhole,
            inverse_cov2d_v2,
            get_radius,
            get_rect,
            torch_splat_fully_fused_projection_batch,
        )
        reg.register_torch("build_rotation", build_rotation)
        reg.register_torch("build_scaling_rotation", build_scaling_rotation)
        reg.register_torch("build_covariance_3d", build_covariance_3d)
        reg.register_torch("build_covariance_2d", build_covariance_2d)
        reg.register_torch("projection_means2d_pinhole", projection_means2d_pinhole)
        reg.register_torch("inverse_cov2d", inverse_cov2d_v2)
        reg.register_torch("get_radius", get_radius)
        reg.register_torch("get_rect", get_rect)
        reg.register_torch("fully_fused_projection_batch", torch_splat_fully_fused_projection_batch)
    except Exception as e:
        warnings.warn(
            f"dispatch: could not register EWA_fully_fused_proj_packed torch impls: {e}",
            stacklevel=2,
        )

    try:
        from sh_utils import eval_sh, build_color
        reg.register_torch("eval_sh", eval_sh)
        reg.register_torch("build_color", build_color)
    except Exception as e:
        warnings.warn(
            f"dispatch: could not register sh_utils torch impls: {e}",
            stacklevel=2,
        )

    try:
        from rasterization_utils import (
            _compute_view_dirs_packed,
            torch_isect_tiles,
            torch_isect_offset_encode,
            torch_rasterize_to_pixels_gaussian_merge,
        )
        reg.register_torch("compute_view_dirs_packed", _compute_view_dirs_packed)
        reg.register_torch("isect_tiles", torch_isect_tiles)
        reg.register_torch("isect_offset_encode", torch_isect_offset_encode)
        reg.register_torch("rasterize_to_pixels", torch_rasterize_to_pixels_gaussian_merge)
    except Exception as e:
        warnings.warn(
            f"dispatch: could not register rasterization_utils torch impls: {e}",
            stacklevel=2,
        )


def _register_kernel_adapters() -> None:
    # Registration of adapters is best-effort: if something in easyasc breaks,
    # the torch paths still work.
    try:
        from .kernel_adapters import register_all
        register_all(registry())
    except Exception as e:
        warnings.warn(
            f"dispatch: kernel adapter registration failed: {e}; kernel paths unavailable",
            stacklevel=2,
        )


_register_torch_impls()
_register_kernel_adapters()


__all__ = [
    "DispatchConfig",
    "dispatch",
    "get_config",
    "ops",
    "registry",
    "set_config",
    "using_config",
]
