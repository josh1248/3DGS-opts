# Migration Progress (caa 27 April 2026)

| stage                    | op name                         | Type      | torch source                                                           | pytorch impl | A2 sim | A2 compiled | A5 sim | A5 compiled |
|--------------------------|---------------------------------|----------|------------------------------------------------------------------------|:------------:|--------|---|--|--|
| (top-level class)                 | `renderer`                      | Forward  | `pytorch.rendering.torch_rasterization`                                | ✅           | - | - | - | - |
| Parameter preprocessing         | `compute_view_dirs_packed`      | Forward  | `rasterization_utils._compute_view_dirs_packed`                        | ✅           | - | - | - | - |
| Parameter preprocessing          | `compute_view_dirs_packed`      | Backward | `rasterization_utils._compute_view_dirs_packed`                        | (autograd)   | - | - | - | - |
| Color preprocessing (Spherical Harmonics) | `eval_sh`                       | Forward  | `sh_utils.eval_sh`                                                     | ✅           | - | - | - | - |
| Color preprocessing (Spherical Harmonics) | `eval_sh`                       | Backward | `sh_utils.eval_sh`                                                     | (autograd)   | - | - | - | - |
| Color preprocessing (Spherical Harmonics) | `build_color`                   | Forward  | `sh_utils.build_color`                                                 | ✅           | - | - | - | - |
| Color preprocessing (Spherical Harmonics) | `build_color`                   | Backward | `sh_utils.build_color`                                                 | (autograd)   | - | - | - | - |
| 3D->2D Projection        | `build_rotation`                | Forward  | `EWA_fully_fused_proj_packed.build_rotation`                           | ✅           | ✅ | ✅ | ✅ | - |
| 3D->2D Projection          | `build_rotation`                | Backward | `EWA_fully_fused_proj_packed.build_rotation`                           | (autograd)   | - | - | - | - |
| 3D->2D Projection        | `build_scaling_rotation`        | Forward  | `EWA_fully_fused_proj_packed.build_scaling_rotation`                   | ✅           | - | - | - | - |
| 3D->2D Projection        | `build_scaling_rotation`        | Backward | `EWA_fully_fused_proj_packed.build_scaling_rotation`                   | (autograd)   | - | - | - | - |
| 3D->2D Projection     | `build_covariance_3d`           | Forward  | `EWA_fully_fused_proj_packed.build_covariance_3d`                      | ✅           | - | - | ✅ | - |
| 3D->2D Projection     | `build_covariance_3d`           | Backward | `EWA_fully_fused_proj_packed.build_covariance_3d`                      | (autograd)   | - | - | - | - |
| 3D->2D Projection     | `projection_means2d_pinhole`    | Forward  | `EWA_fully_fused_proj_packed.projection_means2d_pinhole`               | ✅           | - | - | - | - |
| 3D->2D Projection     | `projection_means2d_pinhole`    | Backward | `EWA_fully_fused_proj_packed.projection_means2d_pinhole`               | (autograd)   | - | - | - | - |
| 3D->2D Projection     | `build_covariance_2d`           | Forward  | `EWA_fully_fused_proj_packed.build_covariance_2d`                      | ✅           | - | - | - | - |
| 3D->2D Projection     | `build_covariance_2d`           | Backward | `EWA_fully_fused_proj_packed.build_covariance_2d`                      | (autograd)   | - | - | - | - |
| 3D->2D Projection     | `inverse_cov2d`                 | Forward  | `EWA_fully_fused_proj_packed.inverse_cov2d_v2`                         | ✅           | - | - | ✅ | - |
| 3D->2D Projection     | `inverse_cov2d`                 | Backward | `EWA_fully_fused_proj_packed.inverse_cov2d_v2`                         | (autograd)   | - | - | - | - |
| 3D->2D Projection     | `fully_fused_projection_batch`  | Forward  | `EWA_fully_fused_proj_packed.torch_splat_fully_fused_projection_batch` | ✅           | - | - | - | - |
| 3D->2D Projection     | `fully_fused_projection_batch`  | Backward | `EWA_fully_fused_proj_packed.torch_splat_fully_fused_projection_batch` | (autograd)   | - | - | - | - |
| Culling       | `get_radius`                    | Forward  | `EWA_fully_fused_proj_packed.get_radius`                               | ✅ (no_grad) | - | - | - | - |
| Culling       | `get_rect`                      | Forward  | `EWA_fully_fused_proj_packed.get_rect`                                 | ✅ (no_grad) | - | - | - | - |
| Gaussian Sorting             | `isect_tiles`                   | Forward  | `rasterization_utils.torch_isect_tiles`                                | ✅ (no_grad) | - | - | - | - |
| Gaussian Sorting             | `isect_offset_encode`           | Forward  | `rasterization_utils.torch_isect_offset_encode`                        | ✅ (no_grad) | - | - | - | - |
| Image rendering            | `rasterize_to_pixels`           | Forward  | `rasterization_utils.torch_rasterize_to_pixels_gaussian_merge`         | ✅           | - | - | - | - |
| Image rendering            | `rasterize_to_pixels`           | Backward | `rasterization_utils.torch_rasterize_to_pixels_gaussian_merge`         | (autograd)   | - | - | - | - |

# How to migrate

## Migration_config.py
Controls which methods are used via global store (meant to be a container interface for dependency injection)

## A2 / A5 sim


## A2 / A5 compiled