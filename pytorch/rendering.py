import torch
import torch.distributed
import torch.nn.functional as F
from torch import Tensor
import math

from typing import Dict, Optional, Tuple
from typing_extensions import Literal

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dependency_config import DependencyConfig


def torch_rasterization(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    colors: Tensor,  # [..., (C,) N, D] or [..., (C,) N, K, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    dependency_config: DependencyConfig,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole",
    segmented: bool = False,
    covars: Optional[Tensor] = None,
    with_ut: bool = False,
    with_eval3d: bool = False,
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize a set of 3D Gaussians (N) to a batch of image planes (C).

    All sub-operations are dispatched through dependency_config, enabling
    hardware-specific kernel injection at any level of the pipeline.

    Args:
        means: The 3D centers of the Gaussians. [..., N, 3]
        quats: The quaternions of the Gaussians (wxyz convention). [..., N, 4]
        scales: The scales of the Gaussians. [..., N, 3]
        opacities: The opacities of the Gaussians. [..., N]
        colors: Colors or SH coefficients. [..., (C,) N, D] or [..., (C,) N, K, 3]
        viewmats: World-to-cam transforms. [..., C, 4, 4]
        Ks: Camera intrinsics. [..., C, 3, 3]
        width: Image width.
        height: Image height.
        dependency_config: Provides all sub-op implementations.
        near_plane: Near clipping plane. Default 0.01.
        far_plane: Far clipping plane. Default 1e10.
        radius_clip: Skip Gaussians with 2D radius <= this. Default 0.0.
        eps2d: Epsilon added to 2D covariance eigenvalues. Default 0.3.
        sh_degree: SH degree; if set, colors are SH coefficients. Default None.
        packed: Use packed mode. Default True.
        tile_size: Tile size for rasterization. Default 16.
        backgrounds: Background colors. [..., C, D]. Default None.
        render_mode: One of "RGB", "D", "ED", "RGB+D", "RGB+ED". Default "RGB".
        absgrad: Compute absolute gradients of projected 2D means. Default False.
        segmented: Use segmented radix sort. Default False.
        covars: Optional covariance matrices; if provided, quats/scales ignored. Default None.

    Returns:
        Tuple of (render_colors, render_alphas, meta).
    """

    batch_dims = means.shape[:-2]
    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]
    I = B * C
    meta = {}


    # Step 1: Fully Fused Projection

    torch_project_results = dependency_config.fully_fused_projection_batch(
        means=means,
        covars=covars,
        quats=quats,
        scales=scales,
        opacities=opacities,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        dependency_config=dependency_config,
    )

    (   batch_ids,     # [nnz]
        camera_ids,    # [nnz]
        gaussian_ids,  # [nnz]
        indptr,        # [B*C+1]
        radii,         # [nnz]
        means2d,       # [nnz,2]
        depths,        # [nnz]
        conics,        # [nnz,3]
        compensations  # [nnz]
    ) = torch_project_results

    opacities = opacities.view(B, N)[batch_ids, gaussian_ids]  # [nnz]
    image_ids = batch_ids * C + camera_ids

    meta.update(
        {
            "batch_ids": batch_ids,
            "camera_ids": camera_ids,
            "gaussian_ids": gaussian_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
        }
    )

    # Step 2: Color Processing
    num_batch_dims = len(batch_dims)
    if sh_degree is None:
        # Colors are post-activation values: [..., N, D] or [..., C, N, D]
        if packed:
            if colors.dim() == num_batch_dims + 2:
                colors = colors.view(B, N, -1)[batch_ids, gaussian_ids]
            else:
                colors = colors.view(B, C, N, -1)[batch_ids, camera_ids, gaussian_ids]
        else:
            if colors.dim() == num_batch_dims + 2:
                colors = torch.broadcast_to(
                    colors[..., None, :, :], batch_dims + (C, N, -1)
                )
    else:
        # Colors are SH coefficients: [..., N, K, 3] or [..., C, N, K, 3]
        campos = torch.inverse(viewmats)[..., :3, 3]  # [..., C, 3]
        if packed:
            dirs = dependency_config.compute_view_dirs_packed(
                means,
                campos,
                batch_ids,
                camera_ids,
                gaussian_ids,
                indptr,
                B,
                C,
            )  # [nnz, 3]

            if colors.dim() == num_batch_dims + 3:
                shs = colors.view(B, N, -1, 3)[batch_ids, gaussian_ids]
            else:
                shs = colors.view(B, C, N, -1, 3)[batch_ids, camera_ids, gaussian_ids]
            colors = dependency_config.build_color(sh_degree, shs, dirs, dependency_config)  # [nnz, 3]

        else:
            dirs = means[..., None, :, :] - campos[..., None, :]  # [..., C, N, 3]
            if colors.dim() == num_batch_dims + 3:
                shs = torch.broadcast_to(
                    colors[..., None, :, :, :], batch_dims + (C, N, -1, 3)
                )
            else:
                shs = colors
            colors = dependency_config.build_color(sh_degree, shs, dirs, dependency_config)  # [...,C,N,3]

    meta.update({"colors": colors})

    # Rasterize to pixels
    print("Start considering colors")
    if render_mode in ["RGB+D", "RGB+ED"]:
        meta['colors'] = torch.cat((meta['colors'], depths[..., None]), dim=-1)
        print("The shape of color is", meta['colors'].shape)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(batch_dims + (C, 1), device=backgrounds.device),
                ],
                dim=-1,
            )
    elif render_mode in ["D", "ED"]:
        meta['colors'] = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(batch_dims + (C, 1), device=backgrounds.device)


    # Step 3: Tile Intersection

    tile_size = 16
    image_ids = (batch_ids.to(torch.long) * C) + camera_ids.to(torch.long)
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))

    (tiles_per_gauss, isect_ids, flatten_ids) = dependency_config.isect_tiles(
        meta["means2d"],
        meta["radii"],
        meta["depths"],
        tile_size,
        tile_width,
        tile_height,
        segmented=segmented,
        packed=packed,
        n_images=I,
        image_ids=image_ids,
        gaussian_ids=meta["gaussian_ids"],
    )

    isect_offsets = dependency_config.isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_batches": B,
            "n_cameras": C,
        }
    )

    # Step 4: Rasterize to Pixels
    print("Start rasterization to pixels...")

    render_colors, render_alphas = dependency_config.rasterize_to_pixels(
        meta["means2d"],
        meta["conics"],
        meta["colors"],
        meta["opacities"],
        meta["width"],
        meta["height"],
        meta["tile_size"],
        meta["isect_offsets"],
        meta["flatten_ids"],
        backgrounds=backgrounds,
        packed=packed,
        absgrad=absgrad,
    )

    if render_mode in ["ED", "RGB+ED"]:
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )
        print(render_colors.shape, render_alphas.shape)

    return render_colors, render_alphas, meta
