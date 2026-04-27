from typing import Callable


class DependencyConfig:
    def __init__(
        self,
        # Top-level renderer
        renderer: Callable,

        # Parameter Preprocessing
        compute_view_dirs_packed: Callable,

        # Spherical Harmonics
        eval_sh: Callable,
        build_color: Callable,

        # 3D -> 2D Projection
        build_rotation: Callable,
        build_scaling_rotation: Callable,
        build_covariance_3d: Callable,
        projection_means2d_pinhole: Callable,
        build_covariance_2d: Callable,
        inverse_cov2d: Callable,
        fully_fused_projection_batch: Callable,

        # Culling
        get_radius: Callable,
        get_rect: Callable,

        # Sorting
        isect_tiles: Callable,
        isect_offset_encode: Callable,

        # Image Rendering
        rasterize_to_pixels: Callable,
    ):
        self.renderer = renderer
        self.compute_view_dirs_packed = compute_view_dirs_packed
        self.eval_sh = eval_sh
        self.build_color = build_color
        self.build_rotation = build_rotation
        self.build_scaling_rotation = build_scaling_rotation
        self.build_covariance_3d = build_covariance_3d
        self.projection_means2d_pinhole = projection_means2d_pinhole
        self.build_covariance_2d = build_covariance_2d
        self.inverse_cov2d = inverse_cov2d
        self.fully_fused_projection_batch = fully_fused_projection_batch
        self.get_radius = get_radius
        self.get_rect = get_rect
        self.isect_tiles = isect_tiles
        self.isect_offset_encode = isect_offset_encode
        self.rasterize_to_pixels = rasterize_to_pixels
