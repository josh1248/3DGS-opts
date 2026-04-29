import torch
import math
import numpy as np

import math
from math import sqrt

# Assuming the use of PinHole camera Model for projection only

"""
Build the covariance matrix for each Gaussian, which is used to compute the splatting weights.
N is the number of Gaussians.
s: [N, 3] scaling along the local x, y, z axis
r: [N, 4] rotation in quaternion (w, x, y, z)
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

    R = torch.zeros((q.size(0), 3, 3), device=r.device, dtype=r.dtype)

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
def build_scaling_rotation(s, r, dependency_config):
    L = torch.zeros((s.shape[0], 3, 3), dtype=s.dtype, device=s.device)
    R = dependency_config.build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

# Build the memory efficient lowerdiag from L (from (N,3,3) to (N,6)).
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=L.dtype, device=L.device)
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
def build_covariance_3d(s, r, dependency_config):
    L = dependency_config.build_scaling_rotation(s, r, dependency_config)
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
    mean3d, cov3d, mean_c, viewmatrix,K, width,height,eps2d=0.3
):
    # The following models the steps outlined by gsplat CUDA code
    # world -> camera
    R = viewmatrix[:3, :3]
    t = viewmatrix[:3, 3]
    z = mean_c[:, 2]

    # Pinhole camera model projection
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x, y = mean_c[:, 0], mean_c[:, 1]
    eps = torch.finfo(mean3d.dtype).eps
    rz = 1.0 / torch.clamp(z, min=eps)


    # Using PinHole Camera model
    tan_fovx = 0.5 * width / fx
    tan_fovy = 0.5 * height / fy
    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    
    eps = torch.finfo(mean3d.dtype).eps
    rz = 1.0 / torch.clamp(z, min=eps)

    # truncate the influences of gaussians far outside the frustum.

    # tx = z * torch.minimum(torch.tensor(lim_x_pos, device=z.device, dtype=z.dtype),
    #                        torch.maximum(torch.tensor(-lim_x_neg, device=z.device, dtype=z.dtype),
    #                                      x * rz))
    # ty = z * torch.minimum(torch.tensor(lim_y_pos, device=z.device, dtype=z.dtype),
    #                        torch.maximum(torch.tensor(-lim_y_neg, device=z.device, dtype=z.dtype),
    #                                      y * rz))

    # rz2 = rz * rz


    # Convert limits to tensors on the same device/dtype as z (no extra copy if already tensor)
    lim_x_pos_t = torch.as_tensor(lim_x_pos, device=z.device, dtype=z.dtype)
    lim_x_neg_t = torch.as_tensor(lim_x_neg, device=z.device, dtype=z.dtype)
    lim_y_pos_t = torch.as_tensor(lim_y_pos, device=z.device, dtype=z.dtype)
    lim_y_neg_t = torch.as_tensor(lim_y_neg, device=z.device, dtype=z.dtype)

    # Clamp x*rz to [-lim_x_neg, lim_x_pos] and y*rz to [-lim_y_neg, lim_y_pos]
    xrz = x * rz
    yrz = y * rz

    tx = z * torch.minimum(lim_x_pos_t, torch.maximum(-lim_x_neg_t, xrz))
    ty = z * torch.minimum(lim_y_pos_t, torch.maximum(-lim_y_neg_t, yrz))

    rz2 = rz * rz

    
    # Eq.29 locally affine transform 
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    # Affline Transform J (N,3,3)
    # Approximate the projection of the 3D covariance to pixel space with a first-order Taylor expansion at t in the camera frame.

    J = torch.zeros(mean3d.shape[0], 2, 3).to(mean3d) # Represently as [N,2,3] for efficient
    J[:, 0, 0] = fx * rz
    J[:, 0, 2] = -fx * tx * rz2
    J[:, 1, 1] = fy * rz
    J[:, 1, 2] = -fy * ty * rz2

    # J[..., 2, 0] = tx / t.norm(dim=-1) # discard
    # J[..., 2, 1] = ty / t.norm(dim=-1) # discard
    # J[..., 2, 2] = tz / t.norm(dim=-1) # discard
    cov_c = R @ cov3d @ R.T
    cov2d = J @ cov_c @ J.transpose(1, 2)    

    # add low pass filter here
    # Choose a Gaussian low-pass filter h(x) = G_v^h(x) where the variance matrix V^h shape is (2,2)
    filter = torch.eye(2,2).to(cov2d) * eps2d # Shape (2,2), diagonal with variance eps2d

    det_orig = cov2d[..., 0,0] * cov2d[...,1,1] - cov2d[...,0,1]*cov2d[...,1,0]
    # Add blur, calculate compensation and det of the blurred
    cov2d_blur = cov2d[:, :2, :2] + filter[None] # Adding to shape (N,2,2)
    det_blur = cov2d_blur[..., 0, 0] * cov2d_blur[..., 1, 1] - cov2d_blur[..., 0, 1] * cov2d_blur[..., 1, 0]
    compensation = torch.sqrt(torch.clamp(det_orig / det_blur, min=0))

    return cov2d_blur,det_blur, compensation
    # cov[0][0] += 0.3f
    # cov[1][1] += 0.3f

# def projection_ndc(points, viewmatrix, projmatrix, near_plane, far_plane):
#     points_o = homogeneous(points) # object space
#     points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
#     p_w = 1.0 / (points_h[..., -1:] + 0.000001)
#     p_proj = points_h * p_w
#     p_view = points_o @ viewmatrix
#     in_mask = (p_view[..., 2] >= near_plane) & (p_view[..., 2] <= far_plane)
#     return p_proj, p_view, in_mask


def projection_means2d_pinhole(points, viewmat, K, near_plane, far_plane):
    """
    Project 3D world points to 2D pixel coordinates using pinhole camera model
    compatible with gsplat.

    Args
        points:   [N, 3] world coordinates
        viewmat:  [4, 4] world-to-camera matrix
        K:        [3, 3] camera intrinsics
        near_plane: float
        far_plane: float

    Returns
        means2D: [N,2] pixel coordinates
        depths:  [N] camera-space z
        in_mask: [N] valid depth mask
    """

    # --- world -> camera ---
    R = viewmat[:3, :3]      # rotation
    t = viewmat[:3, 3]       # translation

    # Pc = R Pw + t
    points_cam = points @ R.T + t

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    # --- depth mask ---
    in_mask = (z >= near_plane) & (z <= far_plane)

    # --- intrinsics ---
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # --- perspective divide ---
    eps = torch.finfo(points.dtype).eps
    inv_z = 1.0 / torch.clamp(z, min=eps)

    u = fx * (x * inv_z) + cx
    v = fy * (y * inv_z) + cy

    means2D = torch.stack([u, v], dim=-1)
    depths = z

    return means2D, points_cam, depths, in_mask



def inverse_cov2d_v2(cov2_00, cov2_01, cov2_11, scale=1):
    det = cov2_00 * cov2_11 - cov2_01 * cov2_01
    inv_x_0 = cov2_11 / det * scale
    inv_x_1 = -cov2_01 / det * scale
    inv_x_2 = cov2_00 / det * scale
    return inv_x_0, inv_x_1, inv_x_2

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


ALPHA_THRESHOLD = 1.0 / 255.0


# Build the packed torch rasterizatin
def torch_splat_fully_fused_projection_batch(
    means,          # [B,N,3] or [N,3] if B=1
    covars=None,    # [B,N,6] or [N,6]
    quats=None,     # [B,N,4] or [N,4]
    scales=None,    # [B,N,3] or [N,3]
    opacities=None, # [B,N] or [N]
    viewmats=None,  # [B,C,4,4] or [C,4,4] if B=1
    Ks=None,        # [B,C,3,3] or [C,3,3] if B=1
    width=640,
    height=480,
    eps2d=0.3,
    near_plane=0.01,
    far_plane=1e10,
    radius_clip=0.0,
    dependency_config=None,
):
    """
    Flexible batched projection: works for B>1 or B=1.
    Returns packed outputs across all batch-camera pairs.
    """

    # Detect single-batch case and add batch dimension
    single_batch = False
    if means.ndim == 2:  # [N,3]
        means = means.unsqueeze(0)  # [1,N,3]
        if covars is not None: covars = covars.unsqueeze(0)
        if quats is not None: quats = quats.unsqueeze(0)
        if scales is not None: scales = scales.unsqueeze(0)
        if opacities is not None: opacities = opacities.unsqueeze(0)
        single_batch = True

    if viewmats.ndim == 3:  # [C,4,4]
        viewmats = viewmats.unsqueeze(0)  # [1,C,4,4]
        Ks = Ks.unsqueeze(0)               # [1,C,3,3]


    # print("means shape:", means.shape)              # Should be [B,N,3]
    # print("viewmats shape:", viewmats.shape)        # Should be [B,C,4,4]
    # print("Ks shape:", Ks.shape)                    # Should be [B,C,3,3]
    B, N, _ = means.shape
    C = viewmats.shape[1]

    batch_ids_list = []
    camera_ids_list = []
    gaussian_ids_list = []
    radii_list = []
    means2D_list = []
    depths_list = []
    conics_list = []
    compensation_list = []


    for b in range(B):
        for c in range(C):
            mean_b = means[b]             
            cov_b = covars[b] if covars is not None else None
            quat_b = quats[b] if quats is not None else None
            scale_b = scales[b] if scales is not None else None
            opac_b = opacities[b] if opacities is not None else None
            viewmat_bc = viewmats[b, c]
            K_bc = Ks[b, c]

            # Inverse view
            print("Projecting batch", b, "camera", c)

            # camtoworld = torch.inverse(viewmat_bc)
            # camera = Camera(width=width, height=height, intrinsic=K_bc, c2w=camtoworld)

            # Project points to NDC

            # Using camera.world_view_transform instead of viewmatbc, beecause world_view_transform here is the transpose of viewmatbc

            means2D, means_c, depths, in_mask = dependency_config.projection_means2d_pinhole(mean_b, viewmat_bc, K_bc, near_plane, far_plane) # change from viewmat_bc to camera world view
            if not in_mask.any():
                continue

            print("Total number of near/far plane masked value is ", in_mask.sum().item())

            means2D = means2D[in_mask]
            depths = depths[in_mask]
            means_c = means_c[in_mask]

            # Apply mask to other tensors
            if cov_b is not None: cov_b = cov_b[in_mask]
            if quat_b is not None: quat_b = quat_b[in_mask]
            if scale_b is not None: scale_b = scale_b[in_mask]
            if opac_b is not None: opac_b = opac_b[in_mask]

            idxs = torch.arange(N, device=means.device)[in_mask]

            # Build Cov3D
            cov3d = dependency_config.build_covariance_3d(scale_b, quat_b, dependency_config)

            # Build Cov2D
            # FoVx, FoVy = compute_fov(K_bc, width, height)
            # focal_x, focal_y = K_bc[0,0], K_bc[1,1]


            cov2d, det_blur, compensation = dependency_config.build_covariance_2d(
                mean3d=mean_b[in_mask],
                cov3d=cov3d,
                mean_c=means_c,
                viewmatrix=viewmat_bc,
                K=K_bc,
                width=width,
                height=height,
                eps2d=eps2d
            )

            # cov2_00 = cov2d[:, 0, 0]
            # cov2_01 = cov2d[:, 0, 1]
            # cov2_11 = cov2d[:, 1, 1]
            valid_mask = det_blur > 0
            extend = torch.full_like(depths, 3.33, dtype=mean_b.dtype)
            # Apply valid with opacity thresholding
            if opac_b is not None:
                opac_b_temp = opac_b.clone()
                # if compensation is not None:
                #     opac_b_temp = opac_b_temp * compensation
                
                pass_alpha = (opac_b_temp >= ALPHA_THRESHOLD)
                valid_mask = valid_mask & pass_alpha
                ratio = opac_b_temp / ALPHA_THRESHOLD

                # Calculate Extend Op
                extend_op = torch.empty_like(opac_b_temp)
                extend_op[pass_alpha] = torch.sqrt(2.0 * torch.log(ratio[pass_alpha]))
                extend_op[~pass_alpha] = float("inf")
                extend = torch.minimum(extend, extend_op)
            

            # Radii from covariance diagonal
            if cov2d is not None:
                sx = torch.sqrt(torch.clamp(cov2d[..., 0, 0], min=0.0))
                sy = torch.sqrt(torch.clamp(cov2d[..., 1, 1], min=0.0))
                radius_x = torch.ceil(extend * sx)
                radius_y = torch.ceil(extend * sy)

                # Radius clipping
                valid_mask = valid_mask & ~((radius_x <= radius_clip) & (radius_y <= radius_clip))

                valid_mask = valid_mask & ~(
                    (means2D[..., 0] + radius_x <= 0) |
                    (means2D[..., 0] - radius_x >= width) |
                    (means2D[..., 1] + radius_y <= 0) |
                    (means2D[..., 1] - radius_y >= height)
                )

            # # Print all det < 0
            # if not valid_mask.all():
            #     print(f"Warning: Batch {b} Camera {c} has {(~valid_mask).sum().item()} invalid Gaussians")
            # else:
            #     print(f"Batch {b} Camera {c} all Gaussians")
            # # Early skip if nothing valid
            if not valid_mask.any():
                continue

            # test if all valid
            # valid_mask = [True] * len(valid_mask)

            # Apply mask consistently
            cov2d = cov2d[valid_mask]
            idxs = idxs[valid_mask]
            means2D = means2D[valid_mask]
            depths = depths[valid_mask]
            compensation = compensation[valid_mask] if compensation is not None else None

            radius_x = radius_x[valid_mask].to(torch.int32)
            radius_y = radius_y[valid_mask].to(torch.int32)
            radii = torch.stack([radius_x, radius_y], dim=-1)  # [nnz, 2] int32

            print("Total number of masked value is", valid_mask.sum().item())

            # Find conics
            inv_00, inv_01, inv_11 = dependency_config.inverse_cov2d(cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1], scale=1.0)
            conics = torch.stack([inv_00, inv_01, inv_11], dim=-1)


            radii = torch.stack([radius_x, radius_y], dim=-1)  # [N,2]

            # Return everything with appending
            batch_ids_list.append(torch.full_like(idxs, b, dtype=torch.int32))
            camera_ids_list.append(torch.full_like(idxs, c, dtype=torch.int32))
            gaussian_ids_list.append(idxs)
            radii_list.append(radii)
            means2D_list.append(means2D)
            depths_list.append(depths)
            conics_list.append(conics)
            compensation_list.append(compensation if compensation is not None else torch.ones_like(depths))

    # Concatenate results
    batch_ids = torch.cat(batch_ids_list, dim=0)
    camera_ids = torch.cat(camera_ids_list, dim=0)
    gaussian_ids = torch.cat(gaussian_ids_list, dim=0)
    radii = torch.cat(radii_list, dim=0)
    means2D = torch.cat(means2D_list, dim=0)
    depths = torch.cat(depths_list, dim=0)
    conics = torch.cat(conics_list, dim=0)
    compensations = torch.cat(compensation_list, dim=0)

    # # If it was originally B=1, return without batch dimension
    # if single_batch:
    #     batch_ids = batch_ids.squeeze(0)
    #     camera_ids = camera_ids.squeeze(0)
    
    indptr = [0]
    for b in range(B):
        for c in range(C):
            mask = (batch_ids == b) & (camera_ids == c)
            nnz = mask.sum().item()
            indptr.append(indptr[-1] + nnz)
    indptr = torch.tensor(indptr, dtype=torch.int32, device=means.device)  # [B*C+1]

    return batch_ids, camera_ids, gaussian_ids, indptr, radii, means2D, depths, conics, compensations

