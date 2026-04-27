from rasterization_utils import torch_isect_tiles, torch_isect_offset_encode, torch_rasterize_to_pixels_per_tile_per_pixel_per_gauss, torch_rasterize_to_pixels_gaussian_merge, torch_rasterize_to_pixels_pixels_vectorized
from EWA_fully_fused_proj_packed import *
from sh_utils import build_color

import torch
import torch.distributed
import torch.nn.functional as F
from torch import Tensor
import math

from typing import Dict, Optional, Tuple
from typing_extensions import Literal


# Assign def to torch_rasterize_to_pixels
torch_rasterize_to_pixels = torch_rasterize_to_pixels_gaussian_merge


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
    distributed: bool = False, # For Multi-GPU Distributed Rasterization
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole",
    segmented: bool = False,
    covars: Optional[Tensor] = None,
    with_ut: bool = False,
    with_eval3d: bool = False,
    # # distortion
    # radial_coeffs: Optional[Tensor] = None,  # [..., C, 6] or [..., C, 4]
    # tangential_coeffs: Optional[Tensor] = None,  # [..., C, 2]
    # thin_prism_coeffs: Optional[Tensor] = None,  # [..., C, 4]
    # ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    # # rolling shutter
    # rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    # viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize a set of 3D Gaussians (N) to a batch of image planes (C).

    This function provides a handful features for 3D Gaussian rasterization, which
    we detail in the following notes. A complete profiling of the these features
    can be found in the :ref:`profiling` page.

    .. note::
        Currently, it does basic rasterization only. That means it does not support: 
        (1) sparse_grad
        (2) absgrad
        (3) with_eval3d
        (4) segmented radix sort
        (5) distributed rendering
        (6) camera models other than pinhole
        (7) distortion
        (8) rolling shutter

    .. note::
        Currently support only packed mode. Unpacked mode will be supported in the future.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [..., N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [..., N, 4]
        scales: The scales of the Gaussians. [..., N, 3]
        opacities: The opacities of the Gaussians. [..., N]
        colors: The colors of the Gaussians. [..., (C,) N, D] or [..., (C,) N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [..., C, 4, 4]
        Ks: The camera intrinsics. [..., C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [..., (C,) N, K, 3] SH coefficients,
            else the `colors` should be [..., (C,) N, D] post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [..., C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        sparse_grad: If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        rasterize_mode: The rasterization mode. Supported modes are "classic" and
            "antialiased". Default is "classic".
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        distributed: Whether to use distributed rendering. Default is False. If True,
            The input Gaussians are expected to be a subset of scene in each rank, and
            the function will collaboratively render the images for all ranks.
        camera_model: The camera model to use. Supported models are "pinhole", "ortho",
            "fisheye", and "ftheta". Default is "pinhole".
        segmented: Whether to use segmented radix sort. Default is False.
            Segmented radix sort performs sorting in segments, which is more efficient for the sorting operation itself.
            However, since it requires offset indices as input, additional global memory access is needed, which results
            in slower overall performance in most use cases.
        covars: Optional covariance matrices of the Gaussians. If provided, the `quats` and
            `scales` will be ignored. [..., N, 3, 3], Default is None.
        with_ut: Whether to use Unscented Transform (UT) for projection. Default is False.
        with_eval3d: Whether to calculate Gaussian response in 3D world space, instead
            of 2D image space. Default is False.
        radial_coeffs: Opencv pinhole/fisheye radial distortion coefficients. Default is None.
            For pinhole camera, the shape should be [..., C, 6]. For fisheye camera, the shape
            should be [..., C, 4].
        tangential_coeffs: Opencv pinhole tangential distortion coefficients. Default is None.
            The shape should be [..., C, 2] if provided.
        thin_prism_coeffs: Opencv pinhole thin prism distortion coefficients. Default is None.
            The shape should be [..., C, 4] if provided.
        ftheta_coeffs: F-Theta camera distortion coefficients shared for all cameras.
            Default is None. See `FThetaCameraDistortionParameters` for details.
        rolling_shutter: The rolling shutter type. Default `RollingShutterType.GLOBAL` means
            global shutter.
        viewmats_rs: The second viewmat when rolling shutter is used. Default is None.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [..., C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [..., C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, meta = rasterization(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'conics',
        'opacities', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size'])

    """

    batch_dims = means.shape[:-2]
    # num_batch_dims = len(batch_dims)
    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]
    I = B * C # Total Images to render
    meta = {}


    # Step 1: Torch Splat Fully Fused Projection

    torch_project_results = torch_splat_fully_fused_projection_batch(
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
        radius_clip=radius_clip
    )

    (   batch_ids, # [nnz]
        camera_ids, # [nnz]
        gaussian_ids, # [nnz]
        indptr, # [B*C+1]
        radii, # [nnz]
        means2d, # [nnz,2]
        depths,  #[nnz]
        conics, # [nnz,3]
        compensations # [nnz]
    ) = torch_project_results

    opacities = opacities.view(B, N)[batch_ids, gaussian_ids]  # [nnz]
    image_ids = batch_ids * C + camera_ids

    meta.update(
        {
            # global batch and camera ids
            "batch_ids": batch_ids,
            "camera_ids": camera_ids,
            # local gaussian_ids
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
    # Colors are post-activation values, with shape [..., N, D] or [..., C, N, D]
        if packed:
            if colors.dim() == num_batch_dims + 2:
                # Turn [..., N, D] into [nnz, D]
                colors = colors.view(B, N, -1)[batch_ids, gaussian_ids]
            else:
                # Turn [..., C, N, D] into [nnz, D]
                colors = colors.view(B, C, N, -1)[batch_ids, camera_ids, gaussian_ids]
        else:
            if colors.dim() == num_batch_dims + 2:
                # Turn [..., N, D] into [..., C, N, D]
                colors = torch.broadcast_to(
                    colors[..., None, :, :], batch_dims + (C, N, -1)
                )
            else:
                # colors is already [..., C, N, D]
                pass
    else:
        # Colors are SH coefficients, with shape [..., N, K, 3] or [..., C, N, K, 3]
        campos = torch.inverse(viewmats)[..., :3, 3]  # [..., C, 3]
        if viewmats_rs is not None:
            campos_rs = torch.inverse(viewmats_rs)[..., :3, 3]
            campos = 0.5 * (campos + campos_rs)  # [..., C, 3]
        if packed:
            dirs = _compute_view_dirs_packed(
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
                # Turn [..., N, K, 3] into [nnz, 3]
                shs = colors.view(B, N, -1, 3)[batch_ids, gaussian_ids]  # [nnz, K, 3]
            else:
                # Turn [..., C, N, K, 3] into [nnz, 3]
                shs = colors.view(B, C, N, -1, 3)[
                    batch_ids, camera_ids, gaussian_ids
                ]  # [nnz, K, 3]
            colors = build_color(sh_degree, shs, dirs) # [nnz, 3]
            
        else:
            dirs = means[..., None, :, :] - campos[..., None, :]  # [..., C, N, 3]
            if colors.dim() == num_batch_dims + 3:
                # Turn [..., N, K, 3] into [..., C, N, K, 3]
                shs = torch.broadcast_to(
                    colors[..., None, :, :, :], batch_dims + (C, N, -1, 3)
                )
            else:
                # colors is already [..., C, N, K, 3]
                shs = colors
            colors = build_color(sh_degree, shs, dirs)  # [...,C,N,3]


    meta.update(
    {
        "colors": colors,
    }
    )


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
    else:  # RGB
        pass


    # Step 3: Intersecting Tiles

    tile_size = 16
    image_ids = (batch_ids.to(torch.long) * C) + camera_ids.to(torch.long)  # [nnz] int64
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))


    ( tiles_per_gauss, # [nnz]
    isect_ids, # []
    flatten_ids ) = torch_isect_tiles(
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

    isect_offsets = torch_isect_offset_encode(isect_ids, I, tile_width, tile_height)
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

    # Step 4: Full Rasterize to Pixels
    print("Start rasterization to pixels...")
    
    render_colors, render_alphas = torch_rasterize_to_pixels(
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
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )

        print(render_colors.shape, render_alphas.shape)
    
    return render_colors, render_alphas, meta
