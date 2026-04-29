import torch
import torch.distributed
import torch.nn.functional as F
from torch import Tensor
import math

from typing import Optional, Tuple
from typing_extensions import Literal

def _compute_view_dirs_packed(
    means: Tensor,  # [..., N, 3]
    campos: Tensor,  # [..., C, 3]
    batch_ids: Tensor,  # [nnz]
    camera_ids: Tensor,  # [nnz]
    gaussian_ids: Tensor,  # [nnz]
    indptr: Tensor,  # [B*C+1]
    B: int,
    C: int,
) -> Tensor:
    """Compute view directions for packed Gaussian-camera pairs.

    This function computes the view directions (means - campos) for each
    Gaussian-camera pair in the packed format. It automatically selects between
    a simple vectorized approach or an optimized loop-based approach based on
    the data size and whether campos requires gradients.

    Args:
        means: The 3D centers of the Gaussians. [..., N, 3]
        campos: Camera positions in world coordinates [..., C, 3]
        batch_ids: The batch indices of the projected Gaussians. Int32 tensor of shape [nnz].
        camera_ids: The camera indices of the projected Gaussians. Int32 tensor of shape [nnz].
        gaussian_ids: The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        indptr: CSR-style index pointer into gaussian_ids for batch-camera pairs. Int32 tensor of shape [B*C+1].
        B: Number of batches
        C: Number of cameras

    Returns:
        dirs: View directions [nnz, 3]
    """
    N = means.shape[-2]
    nnz = batch_ids.shape[0]
    device = means.device
    means_flat = means.view(B, N, 3)
    campos_flat = campos.view(B, C, 3)

    if B * C == 1:
        # Single batch-camera pair. No indexed lookup for campos is needed.
        dirs = means_flat[0, gaussian_ids] - campos_flat[0, 0]  # [nnz, 3]
    else:
        avg_means_per_camera = nnz / (B * C)
        split_batch_camera_ops = (
            avg_means_per_camera > 10000
            and campos_flat.is_cuda
            and campos_flat.requires_grad
        )

        if not split_batch_camera_ops:
            # Simple vectorized indexing for campos.
            dirs = (
                means_flat[batch_ids, gaussian_ids] - campos_flat[batch_ids, camera_ids]
            )  # [nnz, 3]
        else:
            # For large N with pose optimization: split into B*C separate operations
            # to avoid many-to-one indexing of campos in backward pass. This speeds up the
            # backwards pass and is more impactful when GPU occupancy is high.
            dirs = torch.empty((nnz, 3), dtype=means_flat.dtype, device=device)
            indptr_cpu = indptr.cpu()
            for b_idx in range(B):
                for c_idx in range(C):
                    bc_idx = b_idx * C + c_idx
                    start_idx = indptr_cpu[bc_idx].item()
                    end_idx = indptr_cpu[bc_idx + 1].item()
                    if start_idx == end_idx:
                        continue

                    # Get the gaussian indices for this batch-camera pair and compute dirs
                    gids = gaussian_ids[start_idx:end_idx]
                    dirs[start_idx:end_idx] = (
                        means_flat[b_idx, gids] - campos_flat[b_idx, c_idx]
                    )

    return dirs


# Intersecting Tiles
@torch.no_grad()
def torch_isect_tiles(
    means2d: Tensor,          # [nnz, 2] or [..., N, 2]
    radii: Tensor,            # [nnz, 2] or [..., N, 2]  (int32 or int64 ok)
    depths: Tensor,           # [nnz] or [..., N]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    segmented: bool = False,  # ignored for now
    packed: bool = False,
    n_images: Optional[int] = None,
    image_ids: Optional[Tensor] = None,     # [nnz] if packed
    gaussian_ids: Optional[Tensor] = None,  # unused (kept for API compatibility)
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    PyTorch reference implementation of gsplat/cuda/csrc/IntersectTile.cu intersect_tile_kernel.

    Returns:
      tiles_per_gauss: int32 [..., N] or [nnz]
      isect_ids: int64 [n_isects]
      flatten_ids: int32 [n_isects]
    """
    if segmented:
        # requested: "no need to do segmented" for now
        print("Warning: segmented=True is not implemented yet. Ignoring segmented flag.")
        pass

    device = means2d.device

    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert radii.shape == (nnz, 2), radii.shape
        assert depths.shape == (nnz,), depths.shape
        assert image_ids is not None, "image_ids is required if packed is True"
        assert gaussian_ids is not None, "gaussian_ids is required if packed is True"
        assert n_images is not None, "n_images is required if packed is True"
        image_ids = image_ids.contiguous()
        gaussian_ids = gaussian_ids.contiguous()
        I = n_images
        N = None

    else:
        image_dims = means2d.shape[:-2]
        I = math.prod(image_dims)
        N = means2d.shape[-2]
        assert means2d.shape == image_dims + (N, 2), means2d.shape
        assert radii.shape == image_dims + (N, 2), radii.shape
        assert depths.shape == image_dims + (N,), depths.shape

        means2d = means2d.view(-1, 2) # Turn to shape [I*N, 2]
        radii = radii.view(-1, 2)     # Turn to shape [I*N, 2]
        depths = depths.view(-1)      # Turn to shape [I*N]
        nnz = means2d.shape[0] # = I*N, number of Gaussians need to be proceeded
    
    # Compute tile_n_bits, image_n_bits exactly like the CUDA
    n_tiles = int(tile_width * tile_height)
    # floor(log2(x)) + 1 (assumes x >= 1)
    image_n_bits = int(math.floor(math.log2(I)) + 1) if I > 0 else 0
    tile_n_bits = int(math.floor(math.log2(n_tiles)) + 1) if n_tiles > 0 else 0
    assert image_n_bits + tile_n_bits <= 32, (image_n_bits, tile_n_bits)

    # Convert mean/radius into tile-space
    means_xy = means2d.to(torch.float32)
    tile_x = means_xy[:, 0] / float(tile_size)
    tile_y = means_xy[:, 1] / float(tile_size)
    tile_rx = radii[:, 0] / float(tile_size)
    tile_ry = radii[:, 1] / float(tile_size)

    # Clamp with tile-min inclusive, tile-max exclusive
    tile_min_x = torch.floor(tile_x - tile_rx).to(torch.int64).clamp(0, tile_width)
    tile_min_y = torch.floor(tile_y - tile_ry).to(torch.int64).clamp(0, tile_height)
    tile_max_x = torch.ceil(tile_x + tile_rx).to(torch.int64).clamp(0, tile_width)
    tile_max_y = torch.ceil(tile_y + tile_ry).to(torch.int64).clamp(0, tile_height)

    # Tile per Gauss
    tiles_w = (tile_max_x - tile_min_x).clamp(min=0)
    tiles_h = (tile_max_y - tile_min_y).clamp(min=0)
    tiles_per_gauss = (tiles_w * tiles_h).to(torch.int32)  # [nnz]

    # total intersections
    n_isects = int(tiles_per_gauss.sum().item())

    if n_isects == 0:
        empty_isect = torch.empty((0,), device=device, dtype=torch.int64)
        empty_flat = torch.empty((0,), device=device, dtype=torch.int32)
        if not packed:
            # reshape tiles_per_gauss back to [..., N]
            tiles_per_gauss = tiles_per_gauss.view(*means2d.shape[:-1])[:, :, 0]  # not safe
        return tiles_per_gauss, empty_isect, empty_flat

    # Create the starts for index processing later, exclusive cumsum
    starts = torch.cumsum(tiles_per_gauss.to(torch.int32), dim=0) - tiles_per_gauss.to(torch.int32)  # [nnz]

    # Build per-intersection "which gaussian idx produced this intersection"
    flatten_ids = torch.repeat_interleave(
        torch.arange(nnz, device=device, dtype=torch.int32),
        tiles_per_gauss.to(torch.int64),
    )  # [n_isects]

    # Build per-intersection tile_id by enumerating (i,j) ranges per gaussian
    tile_ids = torch.zeros((n_isects,), device=device, dtype=torch.int64)

    # Prepare iid per gaussian
    if packed:
        iid = image_ids.to(torch.int64)  # [nnz]
    else:
        # iid = idx / N
        assert N is not None
        iid = (torch.arange(nnz, device=device, dtype=torch.int64) // N)

    # Depth bit interpreter: CUDA does int32 reinterprete then zero-extend
    depth_i32 = depths.view(torch.int32)                # reinterpret
    depth_u32 = depth_i32.to(torch.int64) & 0xFFFFFFFF     # zero-extend behavior

    # Fill tile_ids segment-by-segment
    # segment for gaussian g is [starts[g], starts[g] + tiles_per_gauss[g])
    for g in range(nnz):
        cnt = int(tiles_per_gauss[g].item())
        if cnt == 0:
            continue

        base = int(starts[g].item())
        y0 = int(tile_min_y[g].item())
        y1 = int(tile_max_y[g].item())
        x0 = int(tile_min_x[g].item())
        x1 = int(tile_max_x[g].item())

        # enumerate tiles in row-major order, like CUDA nested loops
        # for i in [y0,y1):
        #   for j in [x0,x1):
        #     tile_id = i*tile_width + j
        ids = []
        for i in range(y0, y1):
            row_base = i * tile_width
            for j in range(x0, x1):
                ids.append(row_base + j)

        tile_ids[base : base + cnt] = torch.tensor(ids, device=device, dtype=torch.int64)
    
    # Pack isect_id = (iid << (32 + tile_n_bits)) | (tile_id << 32) | depth_u32
    iid_enc = (iid << (32 + tile_n_bits)).to(torch.int64)
    isect_ids = iid_enc[flatten_ids.to(torch.int64)] | (tile_ids << 32) | depth_u32[flatten_ids.to(torch.int64)]

    if sort and isect_ids.numel() > 0:
        order = torch.argsort(isect_ids)  # ascending
        isect_ids = isect_ids[order]
        flatten_ids = flatten_ids[order]

    # reshape tiles_per_gauss back if not packed
    if not packed:
        # tiles_per_gauss currently [I*N], reshape to [..., N]
        tiles_per_gauss = tiles_per_gauss.view(*means2d.shape[:-1])[:, :, 0]  # NOTE: see below

    
    return tiles_per_gauss, isect_ids, flatten_ids

@torch.no_grad()
def torch_isect_offset_encode(
    isect_ids: Tensor,   # int64 [n_isects], MUST be sorted by isect_id
    n_images: int,       # I
    tile_width: int,
    tile_height: int,
) -> Tensor:
    """
    Pure PyTorch version of gsplat/cuda/csrc/IntersectTile.cu::launch_intersect_offset_kernel.

    Returns:
      offsets: int32 [I, tile_height, tile_width]

    Args:
        isect_ids: Intersection ids. [n_isects]
        n_images: Number of images.
        tile_width: Tile width.
        tile_height: Tile height.

    Returns:
        Offsets. [I, tile_height, tile_width]
    """
    assert isect_ids.dtype == torch.int64, isect_ids.dtype
    assert isect_ids.dim() == 1, isect_ids.shape

    device = isect_ids.device
    n_isects = isect_ids.shape[0]
    n_tiles = int(tile_width * tile_height)

    # matches CUDA: tile_n_bits = floor(log2(n_tiles)) + 1
    # (assumes n_tiles >= 1)
    tile_n_bits = int(math.floor(math.log2(n_tiles)) + 1) if n_tiles > 0 else 0
    
    if n_isects == 0:
        return torch.zeros((n_images, tile_height, tile_width), device=device, dtype=torch.int32)

    # decode tile_slot per intersection (sorted because isect_ids is sorted)
    keys = isect_ids >> 32
    iid = keys >> tile_n_bits
    tid = keys & ((1 << tile_n_bits) - 1)
    tile_slot = iid * n_tiles + tid  # int64 [n_isects], non-decreasing

    # offsets[t] = first index where tile_slot >= t
    all_tiles = torch.arange(n_images * n_tiles, device=device, dtype=torch.int64) #[675]
    offsets_flat = torch.searchsorted(tile_slot, all_tiles, right=False).to(torch.int32) #[n_insects]

    return offsets_flat.view(n_images, tile_height, tile_width)


# Helper to read gaussian params by global id g.
# In non-packed mode g indexes into a flattened [I*N] space, like CUDA comments suggest.


"""
This is implementation of Per Tile, Per Pixel, Per Gaussian, that is why it took so long, almost 11 minutes to proceed in the case with may Gaussians
"""

ALPHA_THRESHOLD = 1.0 / 255.0
T_EARLY_STOP = 1e-4




def torch_rasterize_to_pixels_per_tile_per_pixel_per_gauss(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    conics: Tensor,  # [..., N, 3] or [nnz, 3]
    colors: Tensor,  # [..., N, channels] or [nnz, channels]
    opacities: Tensor,  # [..., N] or [nnz]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., channels]
    masks: Optional[Tensor] = None,  # [..., tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterizes Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] if packed is False, [nnz, 2] if packed is True.
        conics: Inverse of the projected covariances with only upper triangle values. [..., N, 3] if packed is False, [nnz, 3] if packed is True.
        colors: Gaussian colors or ND features. [..., N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [..., N] if packed is False, [nnz] if packed is True.
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [..., tile_height, tile_width]
        flatten_ids: The global flatten indices in [I * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [..., channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [..., tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**. [..., image_height, image_width, channels]
        - **Rendered alphas**. [..., image_height, image_width, 1]
    """

    device = means2d.device
    dtype = means2d.dtype
    nnz = means2d.shape[0]
    channels = colors.shape[-1]

    I, tile_h, tile_w = isect_offsets.shape
    n_isects = int(flatten_ids.numel())
    flatten_ids = flatten_ids.to(torch.int64)
    tile_count = tile_h * tile_w

    assert (
        tile_h * tile_size >= image_height
    ), f"Assert Failed: {tile_h} * {tile_size} >= {image_height}"
    assert (
        tile_w * tile_size >= image_width
    ), f"Assert Failed: {tile_w} * {tile_size} >= {image_width}"


    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(-2)
        assert means2d.shape == image_dims + (N, 2), means2d.shape
        assert conics.shape == image_dims + (N, 3), conics.shape
        assert colors.shape == image_dims + (N, channels), colors.shape
        assert opacities.shape == image_dims + (N,), opacities.shape

    if backgrounds is not None:
        assert backgrounds.shape == image_dims + (channels,), backgrounds.shape
        backgrounds = backgrounds.contiguous()
        print("Background no need, may no support")

    if masks is not None:
        assert masks.shape == isect_offsets.shape, masks.shape
        masks = masks.contiguous()
        print("Mask no need so no support")

    # Pad the channels to the nearest supported number if necessary
    if channels > 513 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors,
                torch.zeros(*colors.shape[:-1], padded_channels, device=device),
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0




    # Outputs    
    render_colors = torch.zeros((I, image_height, image_width, channels), device=device, dtype=dtype)
    render_alphas = torch.zeros((I, image_height, image_width, 1), device=device, dtype=dtype)

    # Precompute full image pixel centers
    xs_full = (torch.arange(image_width, device=device, dtype=dtype) + 0.5) # [W]
    ys_full = (torch.arange(image_height, device=device, dtype=dtype) + 0.5) # [H]


    
    # Helper: read gaussian parameters for a given id g
    def read_gaussian(g: int):
        if packed:
            xy = means2d[g]      # [2]
            con = conics[g]      # [3]
            op = opacities[g]    # []
            col = colors[g]      # [C]
            return xy, con, op, col
        else:
            # g in [0 .. I*N-1]
            view = g // N
            gi = g - view * N
            xy = means2d[view, gi]
            con = conics[view, gi]
            op = opacities[view, gi]
            col = colors[view, gi]
            return xy, con, op, col
    
    # Helper Tile End

    def tile_end(iid: int, ty: int, tx: int) -> int:
        if tx + 1 < tile_w:
            return int(isect_offsets[iid, ty, tx + 1].item())
        if ty + 1 < tile_h:
            return int(isect_offsets[iid, ty + 1, 0].item())
        if (iid == I - 1) and (ty == tile_h - 1) and (tx == tile_w - 1):
            return n_isects
        else:
            return int(isect_offsets[iid + 1, 0, 0].item())


    for iid in range(I):

        # Precompute full image pixel centers
        xs = (torch.arange(image_width, device=device, dtype=dtype) + 0.5) # [W]
        ys = (torch.arange(image_height, device=device, dtype=dtype) + 0.5) # [H]
        bg = backgrounds[iid] if backgrounds is not None else None

        # Flatten offsets exactly like CUDA uses tile_id and tile_id+1
        offsets_flat = isect_offsets[iid].reshape(-1)  # [tile_count]

        for ty in range(tile_h):
            y0 = ty * tile_size
            y1 = min(y0 + tile_size, image_height)

            for tx in range(tile_w):
                x0 = tx * tile_size
                x1 = min(x0 + tile_size, image_width)

                tile_id = ty * tile_w + tx
                

                # Range in flatten_ids for this tile (same rule as CUDA)
                start = int(isect_offsets[iid, ty, tx].item())


                end = tile_end(iid, ty, tx)

                for i in range(y0,y1):
                    py = ys[i].item()
                    for j in range(x0,x1):
                        px = xs[j].item()

                        T = 1.0
                        pix = torch.zeros((channels,), device=device, dtype=dtype)

                        # Iterate gaussians front-to-back
                        for idx in range(start, end):

                            print(f"Processing {end- start} Gaussians insid this loop.")


                            g = int(flatten_ids[idx].item())

                            xy, con, op, col = read_gaussian(g)

                            dx = float(xy[0].item()) - px # dx = xg - px
                            dy = float(xy[1].item()) - py # dy = yg - py

                            # con = (A, B, C) with CUDA's formula:
                            A = float(con[0].item())
                            B = float(con[1].item())
                            Cc = float(con[2].item())

                            sigma = 0.5 * (A * dx * dx + Cc * dy * dy) + B * dx * dy # sigma = 1/2 (A dx^2 + C dy^2) + B dx dy
                            if sigma < 0.0:
                                continue
                                
                            vis = float(torch.exp(torch.tensor(-sigma, device=device, dtype=dtype)).item()) # vis = e^(-sigma)
                            alpha = min(0.999, float(op.item()) * vis)

                            if alpha < ALPHA_THRESHOLD:
                                continue

                            w = alpha * T
                            pix = pix + col * w

                            T = T * (1.0 - alpha)
                            if T <= T_EARLY_STOP:
                                break

                        # Write Outputs
                        render_alphas[iid, i, j, 0] = 1.0 - T
                        if bg is None:
                            render_colors[iid, i, j , :] = pix
                        else:
                            render_colors[iid, i, j, :] = pix + T * bg
    
    return render_colors, render_alphas










"""
This is the implementation of rasterization per tile, per pixel but in Gaussian Merge, which can increase the speed
"""

def torch_rasterize_to_pixels_gaussian_merge(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    conics: Tensor,  # [..., N, 3] or [nnz, 3]
    colors: Tensor,  # [..., N, channels] or [nnz, channels]
    opacities: Tensor,  # [..., N] or [nnz]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., channels]
    masks: Optional[Tensor] = None,  # [..., tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterizes Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] if packed is False, [nnz, 2] if packed is True.
        conics: Inverse of the projected covariances with only upper triangle values. [..., N, 3] if packed is False, [nnz, 3] if packed is True.
        colors: Gaussian colors or ND features. [..., N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [..., N] if packed is False, [nnz] if packed is True.
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [..., tile_height, tile_width]
        flatten_ids: The global flatten indices in [I * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [..., channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [..., tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**. [..., image_height, image_width, channels]
        - **Rendered alphas**. [..., image_height, image_width, 1]
    """

    device = means2d.device
    dtype = means2d.dtype
    nnz = means2d.shape[0]
    channels = colors.shape[-1]

    I, tile_h, tile_w = isect_offsets.shape
    n_isects = int(flatten_ids.numel())
    flatten_ids = flatten_ids.to(torch.int64)
    tile_count = tile_h * tile_w

    assert (
        tile_h * tile_size >= image_height
    ), f"Assert Failed: {tile_h} * {tile_size} >= {image_height}"
    assert (
        tile_w * tile_size >= image_width
    ), f"Assert Failed: {tile_w} * {tile_size} >= {image_width}"


    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(-2)
        assert means2d.shape == image_dims + (N, 2), means2d.shape
        assert conics.shape == image_dims + (N, 3), conics.shape
        assert colors.shape == image_dims + (N, channels), colors.shape
        assert opacities.shape == image_dims + (N,), opacities.shape

    if backgrounds is not None:
        assert backgrounds.shape == image_dims + (channels,), backgrounds.shape
        backgrounds = backgrounds.contiguous()
        print("Background no need, may no support")

    if masks is not None:
        assert masks.shape == isect_offsets.shape, masks.shape
        masks = masks.contiguous()
        print("Mask no need so no support")

    # Pad the channels to the nearest supported number if necessary
    if channels > 513 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors,
                torch.zeros(*colors.shape[:-1], padded_channels, device=device),
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0


    # Outputs    
    render_colors = torch.zeros((I, image_height, image_width, channels), device=device, dtype=dtype)
    render_alphas = torch.zeros((I, image_height, image_width, 1), device=device, dtype=dtype)

    # Precompute full image pixel centers
    xs_full = (torch.arange(image_width, device=device, dtype=dtype) + 0.5) # [W]
    ys_full = (torch.arange(image_height, device=device, dtype=dtype) + 0.5) # [H]
    
    # # Helper: read gaussian parameters for a given id g
    # def read_gaussian(g: int):
    #     if packed:
    #         xy = means2d[g]      # [2]
    #         con = conics[g]      # [3]
    #         op = opacities[g]    # []
    #         col = colors[g]      # [C]
    #         return xy, con, op, col
    #     else:
    #         # g in [0 .. I*N-1]
    #         view = g // N
    #         gi = g - view * N
    #         xy = means2d[view, gi]
    #         con = conics[view, gi]
    #         op = opacities[view, gi]
    #         col = colors[view, gi]
    #         return xy, con, op, col
    
    # Helper Tile End

    def tile_end(iid: int, ty: int, tx: int) -> int:
        if tx + 1 < tile_w:
            return int(isect_offsets[iid, ty, tx + 1].item())
        if ty + 1 < tile_h:
            return int(isect_offsets[iid, ty + 1, 0].item())
        if (iid == I - 1) and (ty == tile_h - 1) and (tx == tile_w - 1):
            return n_isects
        else:
            return int(isect_offsets[iid + 1, 0, 0].item())

    # tile loop
    for iid in range(I):
        bg = backgrounds[iid] if backgrounds is not None else None

        for ty in range(tile_h):
            for tx in range(tile_w):
                if masks is not None and not bool(masks[iid, ty, tx].item()):
                    # background fill for that tile region
                    y0 = ty * tile_size
                    x0 = tx * tile_size
                    y1 = min(y0 + tile_size, image_height)
                    x1 = min(x0 + tile_size, image_width)
                    if bg is not None and y0 < y1 and x0 < x1:
                        render_colors[iid, y0:y1, x0:x1] = bg
                    continue

                start = int(isect_offsets[iid, ty, tx].item())
                end = tile_end(iid, ty, tx)
                if end <= start:
                    # no gaussians => background
                    y0 = ty * tile_size
                    x0 = tx * tile_size
                    y1 = min(y0 + tile_size, image_height)
                    x1 = min(x0 + tile_size, image_width)
                    if bg is not None and y0 < y1 and x0 < x1:
                        render_colors[iid, y0:y1, x0:x1] = bg
                    continue

                # pixel bounds for this tile
                y0 = ty * tile_size
                x0 = tx * tile_size
                y1 = min(y0 + tile_size, image_height)
                x1 = min(x0 + tile_size, image_width)
                if y0 >= y1 or x0 >= x1:
                    continue

                # pixels in this tile (P = h*w)
                xs = xs_full[x0:x1]  # [w]
                ys = ys_full[y0:y1]  # [h]
                # meshgrid -> [h,w]
                px, py = torch.meshgrid(xs, ys, indexing="xy")
                # flatten to [P]
                P = px.numel()
                px = px.reshape(P)
                py = py.reshape(P)

                # gaussians in this tile (M)
                gids = flatten_ids[start:end]           # [M]
                mu = means2d[gids].to(dtype)            # [M,2]
                A = conics[gids, 0].to(dtype)           # [M]
                B = conics[gids, 1].to(dtype)           # [M]
                C = conics[gids, 2].to(dtype)           # [M]
                op = opacities[gids].to(dtype)          # [M]
                col = colors[gids].to(dtype)            # [M,ch]

                # compute dx,dy for all (M,P)
                dx = px[None, :] - mu[:, 0:1]          # [M,P]
                dy = py[None, :] - mu[:, 1:2]          # [M,P]

                power = -0.5 * (A[:, None]*dx*dx + 2*B[:, None]*dx*dy + C[:, None]*dy*dy)  # [M,P]
                w = torch.exp(power)                   # [M,P]
                alpha = torch.clamp(op[:, None] * w, max=0.999)  # [M,P]
                alpha = torch.where(alpha >= ALPHA_THRESHOLD, alpha, torch.zeros_like(alpha))

                # front-to-back compositing along M
                # T_0 = 1
                # T_k = prod_{m < k} (1 - alpha_m)
                one_minus = (1.0 - alpha)              # [M,P]
                # exclusive cumprod:
                # prepend ones then drop last
                T = torch.cumprod(
                    torch.cat([torch.ones((1, P), device=device, dtype=dtype), one_minus], dim=0),
                    dim=0
                )[:-1]                                  # [M,P]
                vis = alpha * T                         # [M,P]

                # colors: out[p] = sum_m vis[m,p] * col[m]
                # do (P,M) @ (M,ch) => (P,ch)
                out = (vis.transpose(0, 1) @ col).view((y1 - y0), (x1 - x0), channels)

                # final transmittance after all gaussians:
                T_final = torch.cumprod(one_minus, dim=0)[-1]  # [P]
                T_final_2d = T_final.view((y1 - y0), (x1 - x0))

                if bg is not None:
                    out = out + T_final_2d.unsqueeze(-1) * bg.view(1, 1, channels)

                render_colors[iid, y0:y1, x0:x1] = out
                render_alphas[iid, y0:y1, x0:x1, 0] = 1.0 - T_final_2d

    return render_colors, render_alphas





"""
This is the implementation of rasterization per tile, per Gaussian inside that tile but pixel vectoriezed, which can increase the speed
"""
def torch_rasterize_to_pixels_pixels_vectorized(
    means2d: Tensor,         # packed=False: [I, N, 2]; packed=True: [nnz, 2]
    conics: Tensor,          # packed=False: [I, N, 3]; packed=True: [nnz, 3]
    colors: Tensor,          # packed=False: [I, N, C]; packed=True: [nnz, C]
    opacities: Tensor,       # packed=False: [I, N];    packed=True: [nnz]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,   # [I, tile_h, tile_w]
    flatten_ids: Tensor,     # [n_isects]; packed=False: ids in [0..I*N-1], packed=True: ids in [0..nnz-1]
    backgrounds: Optional[Tensor] = None,  # [I, C] or None
    packed: bool = False,
    absgrad: bool = False,   # unused in forward; kept for API parity
) -> Tuple[Tensor, Tensor]:
    device = means2d.device
    dtype = means2d.dtype

    # Views inferred from isect_offsets
    assert isect_offsets.dim() == 3, f"isect_offsets must be [I,tile_h,tile_w], got {isect_offsets.shape}"
    I, tile_h, tile_w = isect_offsets.shape
    tile_count = tile_h * tile_w

    flatten_ids = flatten_ids.to(torch.int64)
    n_isects = int(flatten_ids.numel())

    # Infer channels
    C = colors.shape[-1]

    # Validate shapes
    if packed:
        nnz = means2d.shape[0]
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert colors.shape == (nnz, C), colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        assert means2d.dim() == 3 and means2d.shape[0] == I and means2d.shape[-1] == 2, means2d.shape
        N = means2d.shape[1]
        assert conics.shape == (I, N, 3), conics.shape
        assert colors.shape == (I, N, C), colors.shape
        assert opacities.shape == (I, N), opacities.shape

    if backgrounds is not None:
        assert backgrounds.shape == (I, C), backgrounds.shape

    # Output buffers
    render_colors = torch.zeros((I, image_height, image_width, C), device=device, dtype=dtype)
    render_alphas = torch.zeros((I, image_height, image_width, 1), device=device, dtype=dtype)

    # Precompute pixel centers
    xs_full = (torch.arange(image_width, device=device, dtype=dtype) + 0.5)   # [W]
    ys_full = (torch.arange(image_height, device=device, dtype=dtype) + 0.5)  # [H]

    # Helper: read gaussian parameters for a given id g
    def read_gaussian(g: int):
        if packed:
            xy = means2d[g]      # [2]
            con = conics[g]      # [3]
            op = opacities[g]    # []
            col = colors[g]      # [C]
            return xy, con, op, col
        else:
            # g in [0 .. I*N-1]
            view = g // N
            gi = g - view * N
            xy = means2d[view, gi]
            con = conics[view, gi]
            op = opacities[view, gi]
            col = colors[view, gi]
            return xy, con, op, col
    
    # Helper Tile End

    def tile_end(iid: int, ty: int, tx: int) -> int:
        if tx + 1 < tile_w:
            return int(isect_offsets[iid, ty, tx + 1].item())
        if ty + 1 < tile_h:
            return int(isect_offsets[iid, ty + 1, 0].item())
        if (iid == I - 1) and (ty == tile_h - 1) and (tx == tile_w - 1):
            return n_isects
        else:
            return int(isect_offsets[iid + 1, 0, 0].item())

    for iid in range(I):
        bg = backgrounds[iid] if backgrounds is not None else None
        offsets_flat = isect_offsets[iid].reshape(-1)  # [tile_count]

        for tile_id in range(tile_count):
            ty = tile_id // tile_w
            tx = tile_id - ty * tile_w

            y0 = ty * tile_size
            x0 = tx * tile_size
            if y0 >= image_height or x0 >= image_width:
                continue
            y1 = min(y0 + tile_size, image_height)
            x1 = min(x0 + tile_size, image_width)

            # Range in flatten_ids for this tile (same rule as CUDA)
            start = int(isect_offsets[iid, ty, tx].item())
            end = tile_end(iid, ty, tx)

            if end <= start:
                if bg is not None:
                    render_colors[iid, y0:y1, x0:x1, :] = bg.view(1, 1, C)
                continue

            # Pixel grid for tile (broadcastable)
            xs = xs_full[x0:x1].view(1, -1)   # [1, Wt]
            ys = ys_full[y0:y1].view(-1, 1)   # [Ht, 1]

            Ht, Wt = (y1 - y0), (x1 - x0)
            T = torch.ones((Ht, Wt), device=device, dtype=dtype)
            pix = torch.zeros((Ht, Wt, C), device=device, dtype=dtype)

            # sequential over gaussians in tile
            for idx in range(start, end):
                g = int(flatten_ids[idx].item())
                xy, con, op, col = read_gaussian(g)

                dx = xy[0] - xs
                dy = xy[1] - ys

                A, B, Cc = con[0], con[1], con[2]
                sigma = 0.5 * (A * dx * dx + Cc * dy * dy) + B * dx * dy

                vis = torch.exp(-sigma)
                alpha = torch.minimum(op * vis, torch.tensor(0.999, device=device, dtype=dtype))

                valid = (sigma >= 0) & (alpha >= ALPHA_THRESHOLD) & (T > T_EARLY_STOP)
                if not bool(valid.any().item()):
                    continue

                w = alpha * T
                pix = pix + w.unsqueeze(-1) * col.view(1, 1, C)

                # update T only where still active (mimics early-stop-ish behavior)
                T = torch.where(valid, T * (1.0 - alpha), T)

            render_alphas[iid, y0:y1, x0:x1, 0] = 1.0 - T
            if bg is None:
                render_colors[iid, y0:y1, x0:x1, :] = pix
            else:
                render_colors[iid, y0:y1, x0:x1, :] = pix + T.unsqueeze(-1) * bg.view(1, 1, C)

    return render_colors, render_alphas