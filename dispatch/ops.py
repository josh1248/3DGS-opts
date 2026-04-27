"""Public dispatched API. Call ``ops.<name>`` instead of the raw torch function;
each call routes to either the torch impl or the kernel adapter based on the
effective ``DispatchConfig``.

Keep these signatures aligned with the torch reference impls so that migrating
a call site is a one-line edit (``build_rotation(r)`` -> ``ops.build_rotation(r)``).
Every function accepts an optional trailing keyword-only ``_config`` that
overrides the global config for that call only.
"""

from __future__ import annotations

from typing import Optional

from .config import DispatchConfig
from .registry import dispatch


# --- Projection / covariance --------------------------------------------

def build_rotation(r, *, _config: Optional[DispatchConfig] = None):
    return dispatch("build_rotation", r, _config=_config)


def build_scaling_rotation(s, r, *, _config: Optional[DispatchConfig] = None):
    return dispatch("build_scaling_rotation", s, r, _config=_config)


def build_covariance_3d(s, r, *, _config: Optional[DispatchConfig] = None):
    return dispatch("build_covariance_3d", s, r, _config=_config)


def build_covariance_2d(mean3d, cov3d, mean_c, viewmatrix, K, width, height,
                        eps2d=0.3, *, _config: Optional[DispatchConfig] = None):
    return dispatch("build_covariance_2d", mean3d, cov3d, mean_c, viewmatrix, K,
                    width, height, eps2d, _config=_config)


def projection_means2d_pinhole(points, viewmat, K, near_plane, far_plane,
                               *, _config: Optional[DispatchConfig] = None):
    return dispatch("projection_means2d_pinhole", points, viewmat, K,
                    near_plane, far_plane, _config=_config)


def inverse_cov2d(cov2_00, cov2_01, cov2_11, scale=1.0,
                  *, _config: Optional[DispatchConfig] = None):
    return dispatch("inverse_cov2d", cov2_00, cov2_01, cov2_11, scale, _config=_config)


def get_radius(cov2d, *, _config: Optional[DispatchConfig] = None):
    return dispatch("get_radius", cov2d, _config=_config)


def get_rect(pix_coord, radii, width, height,
             *, _config: Optional[DispatchConfig] = None):
    return dispatch("get_rect", pix_coord, radii, width, height, _config=_config)


# --- Spherical harmonics ------------------------------------------------

def eval_sh(deg, sh, dirs, *, _config: Optional[DispatchConfig] = None):
    return dispatch("eval_sh", deg, sh, dirs, _config=_config)


def build_color(sh_degree, shs, rays_d, *, _config: Optional[DispatchConfig] = None):
    return dispatch("build_color", sh_degree, shs, rays_d, _config=_config)


# --- Tile intersection + rasterization ----------------------------------
# These have large keyword surfaces; use *args/**kwargs passthrough so we
# don't duplicate and drift their signatures here.

def compute_view_dirs_packed(*args, _config: Optional[DispatchConfig] = None, **kwargs):
    return dispatch("compute_view_dirs_packed", *args, _config=_config, **kwargs)


def isect_tiles(*args, _config: Optional[DispatchConfig] = None, **kwargs):
    return dispatch("isect_tiles", *args, _config=_config, **kwargs)


def isect_offset_encode(*args, _config: Optional[DispatchConfig] = None, **kwargs):
    return dispatch("isect_offset_encode", *args, _config=_config, **kwargs)


def rasterize_to_pixels(*args, _config: Optional[DispatchConfig] = None, **kwargs):
    return dispatch("rasterize_to_pixels", *args, _config=_config, **kwargs)


# --- Pipeline-level -----------------------------------------------------

def fully_fused_projection_batch(*args, _config: Optional[DispatchConfig] = None, **kwargs):
    return dispatch("fully_fused_projection_batch", *args, _config=_config, **kwargs)
