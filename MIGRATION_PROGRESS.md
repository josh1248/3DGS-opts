## Op list

| stage                    | op name                         | F/B      | pytorch impl | torch source                                                           | kernel adapter |
|--------------------------|---------------------------------|----------|:------------:|------------------------------------------------------------------------|----------------|
| Quat → Rotation          | `build_rotation`                | Forward  | ✅           | `EWA_fully_fused_proj_packed.build_rotation`                           | ✅ `build_rotation.py` (fwd) |
| Quat → Rotation          | `build_rotation`                | Backward | (autograd)   | `EWA_fully_fused_proj_packed.build_rotation`                           | — |
| Scale ⊗ Rotation         | `build_scaling_rotation`        | Forward  | ✅           | `EWA_fully_fused_proj_packed.build_scaling_rotation`                   | — |
| Scale ⊗ Rotation         | `build_scaling_rotation`        | Backward | (autograd)   | `EWA_fully_fused_proj_packed.build_scaling_rotation`                   | — |
| 3D Covariance            | `build_covariance_3d`           | Forward  | ✅           | `EWA_fully_fused_proj_packed.build_covariance_3d`                      | — |
| 3D Covariance            | `build_covariance_3d`           | Backward | (autograd)   | `EWA_fully_fused_proj_packed.build_covariance_3d`                      | — |
| Projection               | `projection_means2d_pinhole`    | Forward  | ✅           | `EWA_fully_fused_proj_packed.projection_means2d_pinhole`               | — |
| Projection               | `projection_means2d_pinhole`    | Backward | (autograd)   | `EWA_fully_fused_proj_packed.projection_means2d_pinhole`               | — |
| 2D Covariance (EWA)      | `build_covariance_2d`           | Forward  | ✅           | `EWA_fully_fused_proj_packed.build_covariance_2d`                      | — |
| 2D Covariance (EWA)      | `build_covariance_2d`           | Backward | (autograd)   | `EWA_fully_fused_proj_packed.build_covariance_2d`                      | — |
| 2D Covariance (EWA)      | `inverse_cov2d`                 | Forward  | ✅           | `EWA_fully_fused_proj_packed.inverse_cov2d_v2`                         | — |
| 2D Covariance (EWA)      | `inverse_cov2d`                 | Backward | (autograd)   | `EWA_fully_fused_proj_packed.inverse_cov2d_v2`                         | — |
| Culling / Bounding       | `get_radius`                    | Forward  | ✅ (no_grad) | `EWA_fully_fused_proj_packed.get_radius`                               | — |
| Culling / Bounding       | `get_rect`                      | Forward  | ✅ (no_grad) | `EWA_fully_fused_proj_packed.get_rect`                                 | — |
| View Directions          | `compute_view_dirs_packed`      | Forward  | ✅           | `rasterization_utils._compute_view_dirs_packed`                        | — |
| View Directions          | `compute_view_dirs_packed`      | Backward | (autograd)   | `rasterization_utils._compute_view_dirs_packed`                        | — |
| SH / Color               | `eval_sh`                       | Forward  | ✅           | `sh_utils.eval_sh`                                                     | — |
| SH / Color               | `eval_sh`                       | Backward | (autograd)   | `sh_utils.eval_sh`                                                     | — |
| SH / Color               | `build_color`                   | Forward  | ✅           | `sh_utils.build_color`                                                 | — |
| SH / Color               | `build_color`                   | Backward | (autograd)   | `sh_utils.build_color`                                                 | — |
| Tile Binning             | `isect_tiles`                   | Forward  | ✅ (no_grad) | `rasterization_utils.torch_isect_tiles`                                | — |
| Tile Binning             | `isect_offset_encode`           | Forward  | ✅ (no_grad) | `rasterization_utils.torch_isect_offset_encode`                        | — |
| Rasterization            | `rasterize_to_pixels`           | Forward  | ✅           | `rasterization_utils.torch_rasterize_to_pixels_gaussian_merge`         | — |
| Rasterization            | `rasterize_to_pixels`           | Backward | (autograd)   | `rasterization_utils.torch_rasterize_to_pixels_gaussian_merge`         | — |
| Orchestration            | `fully_fused_projection_batch`  | Forward  | ✅           | `EWA_fully_fused_proj_packed.torch_splat_fully_fused_projection_batch` | — |
| Orchestration            | `fully_fused_projection_batch`  | Backward | (autograd)   | `EWA_fully_fused_proj_packed.torch_splat_fully_fused_projection_batch` | — |