from __future__ import annotations

from typing import Dict, Tuple

import pdb
import torch
import torch.nn as nn
import math
from einops import reduce

import numpy as np
# from plyfile import PlyData
# from gaussian_splatting.gauss_model import GaussModel
# from gaussian_splatting.gauss_render import GaussRenderer
from gaussian_splatting.utils.camera_utils import Camera, to_viewpoint_camera
from gaussian_splatting.utils.sh_utils import eval_sh
import gaussian_splatting.utils as utils

# Torhc-Splat GPU Code implemented in Torch


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

# Adding 1 to transfer into homogeneous coordinate
def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


"""
Build the covariance matrix for each Gaussian, which is used to compute the splatting weights.
N is the number of Gaussians.
s: [N, 3] scaling along the local x, y, z axis
r: [N, 4] rotation in quaternion (w, x, y, z
L: [N, 3, 3] scaling and rotation matrix, which is used to compute the covariance
cov3d: [N, 3, 3] covariance matrix in 3D space
cov2d: [N, 2, 2] covariance matrix in 2D screen space
"""

# From quaternation to rotation
# r/q: [N, 4] (w, x, y, z)
# R: [N, 3, 3]
def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


# Build the scaling and rotation matrix L, which is used to compute the covariance.
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

# Build the memory efficient lowerdiag from L (from (N,3,3) to (N,6)).
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


# Build Covariance matrix 3D with shape (N,3,3)
def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance
    # symm = strip_symmetric(actual_covariance)
    # return symm


"""
- mean3d: [N, 3] mean of the Gaussian in 3D space
- cov3d: [N, 3, 3] covariance of the Gaussian in 3D space
- viewmatrix: [4, 4] w2c transform matrix
- fov_x, fov_y: scalar, field of view in x and y direction
- focal_x, focal_y: scalar, focal length in x and y direction
- cov2d: [N, 2, 2] covariance of the Gaussian in 2D screen space
"""

def build_covariance_2d(
    mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y
):
    # The following models the steps outlined by equations 29
	# and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	# Additionally considers aspect / scaling of viewport.
	# Transposes used to account for row-/column-major conventions.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3] # Camera coordinates of the 3D Gaussians (Coordinates of the 3D Gaussians transformed into the Camera Local Coordinate Space)
    # Where camera is at (0,0,0)

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # Eq.29 locally affine transform 
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    # Affline Transform J (N,3,3)
    # Approximate the projection of the 3D covariance to pixel space with a first-order Taylor expansion at t in the camera frame.

    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d) # same device and dtype as mean3d
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    # J[..., 2, 0] = tx / t.norm(dim=-1) # discard
    # J[..., 2, 1] = ty / t.norm(dim=-1) # discard
    # J[..., 2, 2] = tz / t.norm(dim=-1) # discard
    W = viewmatrix[:3,:3].T # transpose to correct viewmatrix as following my notebook, t = W @ mean3d + t
    # J: (N,3,3)
    # W: (3,3)
    # cov3d: (N,3,3)

    cov2d = J @ W @ cov3d @ W.T @ J.permute(0,2,1) # J.T is J.permute(0,2,1) # Got shape (N,3,3)
    
    # Follow equation 24 to skip last row and last column, getting cov2d with shape (N,2,2)


    # add low pass filter here according to E.q. 32
    # Choose a Gaussian low-pass filter h(x) = G_v^h(x) where the variance matrix V^h shape is (2,2)
    filter = torch.eye(2,2).to(cov2d) * 0.3 # Shape (2,2), diagonal with variance 0.3
    return cov2d[:, :2, :2] + filter[None] # Adding to shape (N,2,2)
    # cov[0][0] += 0.3f
    # cov[1][1] += 0.3f

# Move from X_world to X_ndc with view matrix and projection matrix.
def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= 0.2
    return p_proj, p_view, in_mask


@torch.no_grad()
# Radius of the bounding circles containing the projected Gaussian ellipse
# Just simple function (a-x)(c-x) - b^2 = 0
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
# Bounding rectangle of the projected Gaussian ellipse, which is used to determine the rasterization range of each Gaussian.
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


def assert_check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_inputs(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    render_mode: str,
    sh_degree: Optional[int],
    N: int,
    C: int
) -> None:

    assert_check(means.shape == (N, 3), f"Invalid shape for means: {means.shape}")
    assert_check(quats.shape == (N, 4), f"Invalid shape for quats: {quats.shape}")
    assert_check(scales.shape == (N, 3), f"Invalid shape for scales: {scales.shape}")
    assert_check(opacities.shape == (N,), f"Invalid shape for opacities: {opacities.shape}")
    assert_check(viewmats.shape == (4, 4), f"Invalid shape for viewmats: {viewmats.shape}") # Exinstrics Matrix
    assert_check(Ks.shape == (3, 3), f"Invalid shape for Ks: {Ks.shape}") # Intrinsics Matrix
    assert_check(render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], f"Invalid render_mode: {render_mode}")

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [ N, D] or [C, N, D]
        assert_check((colors.dim() == 2 and colors.shape[0] == N) or
                    (colors.dim() == 3 and colors.shape[:2] == (C, N)),
                    f"Invalid shape for colors: {colors.shape}")
    else:
        # treat colors as SH coefficients, should be in shape [N, K, 3] or [C, N, K, 3]
        # Allowing for activating partial SH bands

        # K is number of SH coefficients, at least (sh_degree+1)**2
        # SH coefficients allow a Gaussian to have view-different colour
        assert_check((colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3) or
                    (colors.dim() == 4 and colors.shape[:2] == (C, N) and colors.shape[3] == 3),
                    f"Invalid shape for colors: {colors.shape}")

        assert_check((sh_degree + 1) ** 2 <= colors.shape[-2], f"Invalid sh_degree for colors shape: {colors.shape}")




class Rasterizer:

    """
    Variable of Rasterizer:
    - self.tile_size: int, the size (in pixels) of the square blocks used for tiled rendering (32)
    - self.padded_width: int, the width of the image, padding to be multiple of tile_size
    - self.padded_height: int, the height of the image, padding to be multiple of tile_size
    - self.tile_grid: Tensor, shape (Num_Tiles, 2), list of top-left coordinates for each tile in the image (non-overlapping window)
    - self.pix_coord: Tensor, shape (W,H,2), grid of (x,y) coordinates for every single pixel in the image -> Grid Paper for you to later draw on it with RGB
    """

    def __init__(self, white_bkgd=True, **kwargs):
        self.debug = False
        self.white_bkgd = white_bkgd
        self.tile_size = None
        self.padded_width = None
        self.padded_height = None
        self.tile_grid = None
        self.pix_coord = None


    def gpu_rasterize_splats(
        self,
        w2c: Tensor, # Shape (4,4)
        Knorm: Tensor, # Shape (3,3)
        width: int,
        height: int,
        tile_size: int, # int values
        splats: dict, # dictionary of 5 keys: "mean", "rotation", "scale", "opacity", "color"
        active_sh_degree: int, # int
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
    
        if self.tile_grid == None:

            self.tile_size = tile_size
            self.padded_width = math.ceil(width/tile_size)*tile_size
            self.padded_height = math.ceil(height/tile_size)*tile_size
            self.tile_grid = torch.stack(torch.meshgrid(torch.arange(0, self.padded_height, tile_size), \
                                                        torch.arange(0, self.padded_width, tile_size), indexing='ij'), dim=-1).view(-1, 2).to(splats["mean"].device)
            self.pix_coord = torch.stack(torch.meshgrid(torch.arange(self.padded_width), torch.arange(self.padded_height), indexing='xy'), dim=-1).to(splats["mean"].device)
            # Shape of self.pix_coord is (H,W,2); with padded -> H= 19*32=608, W = 28*32 = 896, Original= (584, 876)
            # print("Check Shape Analysis Of Preprocessing Data")
            # print(f"tile_size: {tile_size}")
            # print(f"padded_width: {self.padded_width}")
            # print(f"padded_height: {self.padded_height}")
            # print(f"tile_grid shape: {self.tile_grid.shape}")
            # print(f"pix_coord shape: {self.pix_coord.shape}")
            # print("-" * 50)

        means = splats["mean"]  # splats["means"]  # [N, 3]
        quats = splats["rotation"]  # splats["quats"]  # [N, 4]
        scales = splats["scale"]  # splats["scales"]  # torch.exp(splats["scales"])  # [N, 3]
        opacities = splats["opacity"].flatten()  # splats["opacities"]  # torch.sigmoid(splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None) #
        colors = splats["color"] # .unflatten(-1, (-1, 3))  # torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, 1, 3]

        rasterize_mode = "classic"
        Ks = Knorm * Knorm.new_tensor((width, height, 1))[:, None]
        # render_colors, render_depth, info = self._ascend_rasterization(
        # rets = self._gpu_rasterization(
        #     means=mean