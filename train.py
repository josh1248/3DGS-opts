"""Train 3D Gaussian Splatting on a single ScanNet++ scene using the local
`rendering.torch_rasterization` pipeline (no gsplat dependency).

Cold start: parses the three per-scene .npz files via `data_cache.load_scene`
and writes a `.pt` cache. Subsequent runs read the .pt directly.
"""
VOXEL_SIZE = 1 / 128
NUM_GS = 4
ALIGN_RATIO = 5
O_BIAS_INIT = .1
SCA_BIAS_INIT = VOXEL_SIZE
N_training_VIEWS = 8
N_VIEWS = 8

N_STEPS = 400
EVAL_PER = 20
PROFILE = True
SCENE = '31a2c91c43'

import os
import argparse
import datetime
from gc import collect

import cv2
import numpy as np
from tqdm import trange

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from sample import depth_to_world, voxel_downsample, tsdf_fusion
from data_cache import load_scene
from dependency_config import DependencyConfig
from pytorch.rendering import torch_rasterization
from pytorch.EWA_fully_fused_proj_packed import (
    build_rotation,
    build_scaling_rotation,
    build_covariance_3d,
    projection_means2d_pinhole,
    build_covariance_2d,
    inverse_cov2d_v2,
    torch_splat_fully_fused_projection_batch,
    get_radius,
    get_rect,
)
from pytorch.sh_utils import eval_sh, build_color
from pytorch.rasterization_utils import (
    _compute_view_dirs_packed,
    torch_isect_tiles,
    torch_isect_offset_encode,
    torch_rasterize_to_pixels_gaussian_merge,
)

default_dependency_config = DependencyConfig(
    renderer=torch_rasterization,
    compute_view_dirs_packed=_compute_view_dirs_packed,
    eval_sh=eval_sh,
    build_color=build_color,
    build_rotation=build_rotation,
    build_scaling_rotation=build_scaling_rotation,
    build_covariance_3d=build_covariance_3d,
    projection_means2d_pinhole=projection_means2d_pinhole,
    build_covariance_2d=build_covariance_2d,
    inverse_cov2d=inverse_cov2d_v2,
    fully_fused_projection_batch=torch_splat_fully_fused_projection_batch,
    get_radius=get_radius,
    get_rect=get_rect,
    isect_tiles=torch_isect_tiles,
    isect_offset_encode=torch_isect_offset_encode,
    rasterize_to_pixels=torch_rasterize_to_pixels_gaussian_merge,
)


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
DTYPE = torch.float

XYZ_SCALE = torch.tensor(ALIGN_RATIO * VOXEL_SIZE, device=DEVICE, dtype=DTYPE)
O_BIAS = -(1 / torch.tensor(O_BIAS_INIT, device=DEVICE, dtype=DTYPE) - 1).log()
SCA_BIAS = torch.expm1(torch.tensor(SCA_BIAS_INIT, device=DEVICE, dtype=DTYPE)).log()


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default=SCENE,
                        help='scene name for training')
    parser.add_argument('--profile', action='store_true',
                        help='whether to profile the code')
    parser.add_argument('--device', type=str, default=str(DEVICE),
                        help='torch device override (e.g. cpu, cuda)')
    return parser.parse_args()


def _empty_cache(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def _render(xyz, quat, sca, o, rgb, viewmats, Ks, width, height, dependency_config):
    """Wrap the renderer with the post-processing the loop expects."""
    out = dependency_config.renderer(
        means=xyz.reshape(-1, 3),
        quats=quat.reshape(-1, 4),
        scales=sca.reshape(-1, 3),
        opacities=o.reshape(-1),
        colors=rgb.reshape(-1, 3),
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        dependency_config=dependency_config,
        render_mode='RGB+ED',
    )[0][..., :3]
    # [..., C, H, W, 3] -> [C, 3, H, W]
    return out.movedim(-1, -3).reshape(-1, 3, height, width).clamp(0, 1)


def main():
    arg = args_parse()
    scene = arg.scene
    device = torch.device(arg.device)

    print(f'Backend: local torch_rasterization (rendering.py)')
    print(f'Device:  {device}')
    print(f'Scene:   {scene}')

    time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    vis_dir = f'vis-local-{time}'
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Loading data (cached on second run onwards)
    print('Loading data')
    scene_data = load_scene(scene)

    # 32 indexes used for training, picked from 64 supervise views.
    idx32 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
             33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]

    print('Copying data')
    images = scene_data['train_img'].to(device=device, dtype=torch.float)
    images = F.interpolate(images, scale_factor=.5, mode='area')   # [64, 3, 584, 876]

    intrinsics = scene_data['train_K'].to(device=device, dtype=DTYPE)
    extrinsics = scene_data['train_extr'].to(device=device, dtype=DTYPE)

    images_nv = scene_data['novel_img'].to(device=device, dtype=torch.float)
    images_nv = F.interpolate(images_nv, scale_factor=.5, mode='area')

    intrinsics_nv = scene_data['novel_K'].to(device=device, dtype=DTYPE)
    extrinsics_nv = scene_data['novel_extr'].to(device=device, dtype=DTYPE)

    collect()
    _empty_cache(device)

    # 2. Sampling anchors via TSDF fusion of the depth predictions
    print('Sampling anchors')
    depth_ma = scene_data['depth_pred'].to(device=device, dtype=DTYPE)
    anchors = tsdf_fusion(depth_ma, intrinsics[idx32], extrinsics[idx32], VOXEL_SIZE)

    del depth_ma
    collect()
    _empty_cache(device)

    # 3. Initialize Gaussian features
    # 14 = 3 xyz offset + 1 opacity + 3 scale + 4 quat + 3 RGB
    feature = .1 * torch.randn(anchors.shape[0], NUM_GS, 14, device=device, dtype=DTYPE)

    init_save = f'data/local/anchors_features_init_{time}.pt'
    os.makedirs(os.path.dirname(init_save), exist_ok=True)
    torch.save({
        'anchors':    anchors.cpu(),
        'feature':    feature.cpu(),
        'voxel_size': VOXEL_SIZE,
        'num_gs':     NUM_GS,
    }, init_save)
    print(f'Saved init to {init_save}')

    optim = torch.optim.Adam([feature.requires_grad_()], .01)

    # Denormalise intrinsics: the npz stores them normalized to [0,1] image extents.
    intrinsics[:, :1]    *= images.shape[3]
    intrinsics[:, 1:2]   *= images.shape[2]
    intrinsics_nv[:, :1] *= images_nv.shape[3]
    intrinsics_nv[:, 1:2] *= images_nv.shape[2]

    pbar = trange(N_STEPS, desc=f'Num GS = {anchors.shape[0] * NUM_GS}')
    writer = SummaryWriter(f'tensorboard-local/run-{time}')

    prof = None
    if PROFILE:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        prof = torch.profiler.profile(
            activities=activities,
            with_stack=True,
            record_shapes=False,
            profile_memory=True,
            schedule=torch.profiler.schedule(wait=10, warmup=10, active=1, repeat=0, skip_first=0),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('profile-local'),
        )
        prof.start()

    psnr = float('nan')
    psnr_nv = float('nan')

    for i in pbar:
        # Decode the 14-D feature vector into the per-Gaussian parameters.
        xyz  = anchors[:, None] + XYZ_SCALE * feature[..., :3].tanh()
        o    = (feature[..., 3:4] + O_BIAS).sigmoid()
        sca  = F.softplus(feature[..., 4:7] + SCA_BIAS)
        quat = F.normalize(feature[..., 7:11], dim=2)
        rgb  = feature[..., 11:]

        views = torch.randint(0, images.shape[0], (N_training_VIEWS,), device=device)

        render = _render(
            xyz, quat, sca, o, rgb,
            viewmats=extrinsics[views],
            Ks=intrinsics[views],
            width=images.shape[3],
            height=images.shape[2],
            dependency_config=default_dependency_config,
        )

        optim.zero_grad()
        loss = F.l1_loss(render, images[views])
        loss.backward()
        optim.step()

        if PROFILE and prof is not None:
            prof.step()

        with torch.inference_mode():
            psnr = -10 * F.mse_loss(render, images[views], reduction='none').mean((1, 2, 3)).log10_().mean().item()
        writer.add_scalar('train PSNR', psnr, i, new_style=True)

        if not i % EVAL_PER:
            with torch.inference_mode():
                render_nv = _render(
                    xyz, quat, sca, o, rgb,
                    viewmats=extrinsics_nv,
                    Ks=intrinsics_nv,
                    width=images_nv.shape[3],
                    height=images_nv.shape[2],
                    dependency_config=default_dependency_config,
                )

            psnr_nv = -10 * F.mse_loss(render_nv, images_nv, reduction='none').mean((1, 2, 3)).log10_().mean().item()
            writer.add_scalar('val PSNR', psnr_nv, i, new_style=True)

            additional_views = torch.cat([
                views,
                torch.randint(0, images.shape[0], (N_VIEWS - N_training_VIEWS,), device=device),
            ])
            render_broadcasted = render.repeat(N_VIEWS // render.shape[0], 1, 1, 1)
            vis = torch.stack((images[additional_views], render_broadcasted, images_nv, render_nv)).mul_(255).round_().byte()
            vis = vis.permute(0, 3, 1, 4, 2)[..., [2, 1, 0]].flatten(0, 1).flatten(1, 2).cpu().numpy()
            vis = cv2.resize(vis, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
            cv2.imwrite(f'{vis_dir}/{i:04}.png', vis, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        pbar.set_postfix({'train PSNR': psnr, 'val PSNR': psnr_nv})

    if PROFILE and prof is not None:
        prof.stop()

    # final_save = f'data/local/anchors_features_trained_{time}.pt'
    # os.makedirs(os.path.dirname(final_save), exist_ok=True)
    # torch.save({
    #     'anchors':    anchors.cpu(),
    #     'feature':    feature.cpu(),
    #     'voxel_size': VOXEL_SIZE,
    #     'num_gs':     NUM_GS,
    # }, final_save)
    # print(f'Saved trained to {final_save}')

    writer.close()


if __name__ == '__main__':
    main()
