# TorchSplat: A Deep Dive into Pure-PyTorch 3D Gaussian Splatting

## Table of Contents

1. [Architecture Overview & Data Flow](#1-architecture-overview--data-flow)
2. [File 1: `rasterizer_torchsplat.py` — The Entry Point](#2-file-1-rasterizer_torchsplatpy--the-entry-point)
3. [File 2: `EWA_fully_fused_proj_packed.py` — Projection Engine](#3-file-2-ewa_fully_fused_proj_packedpy--projection-engine)
4. [File 3: `rendering.py` — The Pipeline Orchestrator](#4-file-3-renderingpy--the-pipeline-orchestrator)
5. [File 4: `rasterization_utils.py` — Tile Intersection & Pixel Rasterization](#5-file-4-rasterization_utilspy--tile-intersection--pixel-rasterization)
6. [File 5: `sh_utils.py` — Spherical Harmonics Color Evaluation](#6-file-5-sh_utilspy--spherical-harmonics-color-evaluation)
7. [Comparison with gsplat (nerfstudio)](#7-comparison-with-gsplat-nerfstudio)
8. [End-to-End Walkthrough: From 3D Gaussians to Pixels](#8-end-to-end-walkthrough-from-3d-gaussians-to-pixels)

---

## 1. Architecture Overview & Data Flow

TorchSplat implements the **forward rasterization pass** of 3D Gaussian Splatting (3DGS) entirely in PyTorch tensor operations — no custom CUDA kernels. This means every operation is differentiable via PyTorch's autograd, which is critical for the optimization loop that makes 3DGS work (gradient-based updates to Gaussian parameters to minimize a photometric loss).

The pipeline has four sequential stages that mirror the original 3DGS paper (Kerbl et al., 2023) and the gsplat library:

```
┌─────────────────────────────────────────────────────────────────┐
│                   rasterizer_torchsplat.py                      │
│              (Entry point: Rasterizer class)                    │
│  • Accepts raw Gaussian params (mean, quat, scale, opacity, SH)│
│  • Sets up tile grid, pixel coordinate grid                     │
│  • Normalizes camera intrinsics                                 │
│  • Calls into rendering.py                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       rendering.py                              │
│               (Pipeline Orchestrator)                            │
│                                                                 │
│  Step 1: EWA Projection ──► EWA_fully_fused_proj_packed.py      │
│          (3D→2D, cov3D→cov2D, culling, radii)                  │
│                                                                 │
│  Step 2: Color Processing                                       │
│          (SH evaluation OR pass-through) ──► sh_utils.py        │
│                                                                 │
│  Step 3: Tile Intersection ──► rasterization_utils.py           │
│          (which tiles does each 2D Gaussian overlap?)           │
│                                                                 │
│  Step 4: Per-Pixel Rasterization ──► rasterization_utils.py     │
│          (alpha-compositing within each tile)                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key data representations that flow through the pipeline:**

- **Packed format**: Instead of carrying `[N, ...]` tensors for all N Gaussians (many of which may be culled), the pipeline "packs" only the surviving Gaussians into a flat `[nnz, ...]` tensor with companion `batch_ids`, `camera_ids`, and `gaussian_ids` index arrays. This is directly inspired by gsplat's packed mode and is analogous to a sparse CSR representation.

- **Conics**: The 2×2 inverse covariance matrix is stored as 3 values `(A, B, C)` representing the upper triangle of the symmetric matrix. This is the compact representation used in the actual CUDA rasterizer too.

- **isect_ids / flatten_ids**: Intersection IDs encode `(image_id, tile_id, depth)` into a single int64 for global sorting. After sorting, `flatten_ids` gives you the Gaussian index for each intersection, in front-to-back order per tile.

---

## 2. File 1: `rasterizer_torchsplat.py` — The Entry Point

### Role in the 3DGS Algorithm

This file is the **user-facing API**. In a training loop, you'd call `Rasterizer.gpu_rasterize_splats(...)` each iteration, passing your current Gaussian parameters and camera pose. It corresponds to the top-level `rasterization()` function in gsplat.

### What It Does

**Initialization (`__init__`)**:
- Stores a `white_bkgd` flag (whether to composite over white or black background — this matters for loss computation since the original 3DGS paper uses a white background for certain datasets).
- Lazily initializes `tile_grid` and `pix_coord` on first call.

**`gpu_rasterize_splats` method**:

1. **Tile Grid Setup**: Computes a padded image size (rounded up to multiples of `tile_size`) and creates:
   - `tile_grid`: Shape `(num_tiles, 2)` — the top-left (row, col) of every tile. Used later to know which pixel region each tile covers.
   - `pix_coord`: Shape `(padded_W, padded_H, 2)` — a meshgrid of `(x, y)` pixel coordinates. This is the "canvas" that Gaussians will be splatted onto.

2. **Input Extraction**: Unpacks the `splats` dictionary into separate tensors: `means [N,3]`, `quats [N,4]`, `scales [N,3]`, `opacities [N,1]→[N]`, `colors [N,D]`.

3. **Intrinsics Denormalization**: The input `Knorm` is a normalized 3×3 intrinsics matrix. The line:
   ```python
   Ks = Knorm * Knorm.new_tensor((width, height, 1))[:, None]
   ```
   scales the first row by `width` and second row by `height`, converting from normalized coordinates (where principal point and focal length are in [0,1]) to pixel coordinates. This is a gsplat convention.

### Supplementary: Why Tile Grids?

In the original 3DGS CUDA implementation, the image is divided into 16×16 pixel tiles. Each CUDA thread block processes one tile. This is efficient because:
- Each Gaussian only overlaps a small number of tiles (determined by its 2D radius).
- Within a tile, all 256 pixels share the same set of overlapping Gaussians, so you load each Gaussian once into shared memory and evaluate it against all 256 pixels.

TorchSplat replicates this tiling structure in Python/PyTorch, not for GPU thread-block efficiency, but to maintain algorithmic equivalence with the CUDA version.

### Helper Functions in This File

The file also contains several standalone functions that implement parts of the 3DGS math. These appear to be an **earlier, standalone version** of the projection code (before the batched version in `EWA_fully_fused_proj_packed.py` was written). They include `build_rotation`, `build_covariance_3d`, `build_covariance_2d`, `projection_ndc`, `get_radius`, `get_rect`, and `validate_inputs`. The actual pipeline in `rendering.py` calls the versions in `EWA_fully_fused_proj_packed.py`, but these standalone functions are useful for understanding the math in isolation.

### `validate_inputs`

This function enforces shape constraints on all inputs before the pipeline begins. Notable checks:
- If `sh_degree` is `None`, colors are treated as post-activation RGB values with shape `[N, D]`.
- If `sh_degree` is set, colors are treated as SH coefficient tensors with shape `[N, K, 3]`, where `K ≥ (sh_degree+1)²` (the number of SH basis functions up to that degree).

---

## 3. File 2: `EWA_fully_fused_proj_packed.py` — Projection Engine

### Role in the 3DGS Algorithm

This is the heart of **Stage 1: EWA Splatting Projection**. It takes every 3D Gaussian and projects it into 2D screen space, producing:
- 2D mean position (pixel coordinates)
- 2D covariance matrix (the projected ellipse shape)
- Depth (for sorting)
- Conics (inverse of 2D covariance, used for fast evaluation during rasterization)
- Radii (bounding box of the 2D ellipse, for tile intersection)

The name "EWA" comes from the seminal paper *"EWA Splatting"* by Zwicker et al. (2002), which introduced the Elliptical Weighted Average framework for projecting 3D Gaussians to 2D.

### Step-by-Step Math and Code

#### 3.1: Quaternion → Rotation Matrix (`build_rotation`)

Each Gaussian has an orientation represented as a unit quaternion `q = (w, x, y, z)`. The function first normalizes it, then converts to a 3×3 rotation matrix using the standard formula:

$$
R = \begin{pmatrix}
1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\
2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)
\end{pmatrix}
$$

**Why quaternions?** They avoid gimbal lock, are compact (4 values vs 9 for a matrix), and are easy to normalize (just divide by the norm). During optimization, gradients flow back through this conversion to update the quaternion parameters.

#### 3.2: 3D Covariance Matrix (`build_covariance_3d`)

A 3D Gaussian is parameterized by:
- **Mean** μ ∈ ℝ³ (position)
- **Scale** s ∈ ℝ³ (anisotropic scaling along local axes)
- **Rotation** R ∈ SO(3) (orientation)

The covariance matrix Σ₃D is constructed as:

$$
S = \text{diag}(s_x, s_y, s_z), \quad L = R \cdot S
$$

$$
\Sigma_{3D} = L \cdot L^T = R \cdot S \cdot S^T \cdot R^T
$$

In code, `build_scaling_rotation(s, r)` creates L by:
1. Building a diagonal matrix from the scale vector.
2. Left-multiplying by the rotation matrix.

Then `build_covariance_3d` computes `L @ L.T`.

**Why L·Lᵀ instead of directly parameterizing Σ?** This factorization guarantees Σ is always symmetric positive semi-definite — a mathematical requirement for any valid covariance matrix. If you parameterized Σ directly, gradient updates could violate this constraint.

#### 3.3: World → Camera Coordinates (`projection_means2d_pinhole`)

Each 3D Gaussian mean is transformed from world space to camera space:

$$
\mathbf{p}_{cam} = R_{view} \cdot \mathbf{p}_{world} + \mathbf{t}_{view}
$$

where `R_view` and `t_view` come from the 4×4 world-to-camera (extrinsics) matrix. In code:

```python
points_cam = points @ R.T + t
```

Then perspective projection gives pixel coordinates:

$$
u = f_x \cdot \frac{x_{cam}}{z_{cam}} + c_x, \quad v = f_y \cdot \frac{y_{cam}}{z_{cam}} + c_y
$$

A near/far plane mask culls Gaussians outside the valid depth range.

#### 3.4: 3D → 2D Covariance via Jacobian (`build_covariance_2d`)

This is the core EWA step. The perspective projection `π: ℝ³ → ℝ²` is **nonlinear** (due to the `1/z` division), so we can't directly transform the 3D covariance to 2D. Instead, we use a **first-order Taylor expansion** (local linearization) at the Gaussian's camera-space position.

The Jacobian of the pinhole projection at point `(x, y, z)` in camera space is:

$$
J = \begin{pmatrix}
\frac{f_x}{z} & 0 & -\frac{f_x \cdot x}{z^2} \\
0 & \frac{f_y}{z} & -\frac{f_y \cdot y}{z^2}
\end{pmatrix}
$$

This is a 2×3 matrix (the code uses `[N, 2, 3]` shape). The third row of a full 3×3 Jacobian is discarded because we only care about the projection to 2D image coordinates.

**Important detail**: The code first clamps `x/z` and `y/z` to lie within the frustum (with a 0.3× margin beyond the field-of-view tangent), storing the clamped values as `tx` and `ty`. This prevents Gaussians far outside the view frustum from producing wildly large projected covariances that would waste computation.

The 2D covariance is then:

$$
\Sigma_{cam} = R_{view} \cdot \Sigma_{3D} \cdot R_{view}^T
$$

$$
\Sigma_{2D} = J \cdot \Sigma_{cam} \cdot J^T
$$

In code:
```python
cov_c = R @ cov3d @ R.T          # Transform 3D cov to camera frame
cov2d = J @ cov_c @ J.transpose(1, 2)  # Project to 2D
```

#### 3.5: Low-Pass Filter (Anti-Aliasing)

A Gaussian that's very far from the camera might project to a sub-pixel ellipse. Sampling this at pixel centers would alias. Following Zwicker et al. (Eq. 32), a low-pass filter is applied:

$$
\Sigma_{2D}' = \Sigma_{2D} + \epsilon \cdot I_{2 \times 2}
$$

where ε = 0.3 (the `eps2d` parameter). This effectively ensures each Gaussian covers at least ~3 pixels in diameter.

The code also computes a **compensation factor**:

$$
\text{compensation} = \sqrt{\frac{\det(\Sigma_{2D})}{\det(\Sigma_{2D}')}}
$$

This records how much the filter inflated the Gaussian, which could be used to adjust opacity (the `antialiased` render mode in gsplat uses this).

#### 3.6: Inverse Covariance → Conics (`inverse_cov2d_v2`)

During rasterization, we need to evaluate the Gaussian at each pixel. The exponent of a 2D Gaussian is:

$$
\sigma(dx, dy) = \frac{1}{2} \begin{pmatrix} dx & dy \end{pmatrix} \Sigma_{2D}^{-1} \begin{pmatrix} dx \\ dy \end{pmatrix}
$$

For a symmetric 2×2 matrix, the inverse is:

$$
\Sigma_{2D} = \begin{pmatrix} a & b \\ b & c \end{pmatrix}, \quad \Sigma_{2D}^{-1} = \frac{1}{ac - b^2}\begin{pmatrix} c & -b \\ -b & a \end{pmatrix}
$$

The function `inverse_cov2d_v2` computes and returns the three unique values `(inv_00, inv_01, inv_11)` — called **conics** because they define a conic section (an ellipse in this case). These are stored as `[nnz, 3]` for efficient rasterization.

#### 3.7: Bounding Radii (`get_radius` and the `extend` logic)

To determine which tiles a projected Gaussian overlaps, we need its bounding radius in pixels. The code computes this in a more sophisticated way than the simple eigenvalue approach in `rasterizer_torchsplat.py`:

1. Compute `extend` = min(3.33, sqrt(2·ln(opacity/threshold))). This is the number of standard deviations at which the Gaussian's contribution falls below the alpha threshold (1/255). If opacity is low, the effective radius shrinks — a nice optimization.

2. Compute axis-aligned radii: `radius_x = ceil(extend · sqrt(σ_xx))` and `radius_y = ceil(extend · sqrt(σ_yy))`, where σ_xx and σ_yy are the diagonal entries of Σ₂D.

3. Cull Gaussians whose bounding box lies entirely outside the image.

This produces `radii [nnz, 2]` — per-axis integer radii used for tile intersection.

#### 3.8: Packed Output Assembly

The function `torch_splat_fully_fused_projection_batch` loops over all (batch, camera) pairs, applies the above steps, and concatenates surviving Gaussians into packed tensors:

- `batch_ids [nnz]`: which batch each surviving Gaussian came from
- `camera_ids [nnz]`: which camera
- `gaussian_ids [nnz]`: original index in the N Gaussians
- `means2D [nnz, 2]`: projected pixel positions
- `depths [nnz]`: camera-space z values
- `conics [nnz, 3]`: inverse covariance elements
- `radii [nnz, 2]`: bounding radii
- `compensations [nnz]`: anti-aliasing compensation
- `indptr [B*C+1]`: CSR-style pointers into the packed arrays for each (batch, camera) pair

---

## 4. File 3: `rendering.py` — The Pipeline Orchestrator

### Role in the 3DGS Algorithm

This is the **conductor** that calls the other modules in sequence. The function `torch_rasterization(...)` accepts the same arguments as gsplat's `rasterization()` function and returns `(render_colors, render_alphas, meta)`.

### Step-by-Step Walkthrough

#### Step 1: Projection

```python
torch_project_results = torch_splat_fully_fused_projection_batch(...)
```

This calls into `EWA_fully_fused_proj_packed.py` (described above). The packed results are destructured, and `opacities` is re-indexed from `[B, N]` to `[nnz]` using `batch_ids` and `gaussian_ids`.

#### Step 2: Color Processing

Two branches depending on whether SH coefficients are being used:

**Branch A: `sh_degree is None` (post-activation colors)**

Colors are already in RGB space with shape `[N, D]`. They just need to be re-indexed into packed format: `colors[batch_ids, gaussian_ids]` → `[nnz, D]`.

**Branch B: `sh_degree is not None` (SH coefficients)**

This is where view-dependent color comes from. The pipeline:

1. Computes camera positions in world space: `campos = inverse(viewmats)[..., :3, 3]`
2. Computes view directions: `dirs = means - campos` (per Gaussian-camera pair)
3. Calls `build_color(sh_degree, shs, dirs)` from `sh_utils.py` to evaluate SH and get RGB colors.

The function `_compute_view_dirs_packed` (defined in `rasterization_utils.py`) handles this efficiently for packed data, with special handling for single batch-camera pairs and an optimization for large scenes where splitting by batch-camera avoids expensive many-to-one backward indexing.

**Render Mode Handling**: After color computation, if the render mode includes depth ("D" or "ED"), the depth values are concatenated as an extra channel to the color tensor. This lets the same rasterization machinery produce depth maps alongside color images.

#### Step 3: Tile Intersection

```python
tiles_per_gauss, isect_ids, flatten_ids = torch_isect_tiles(...)
isect_offsets = torch_isect_offset_encode(...)
```

This calls into `rasterization_utils.py`. The goal: for every tile in the image, determine which Gaussians overlap it, sorted front-to-back by depth. Details in Section 5.

#### Step 4: Per-Pixel Rasterization

```python
render_colors, render_alphas = torch_rasterize_to_pixels(...)
```

The default implementation is `torch_rasterize_to_pixels_gaussian_merge`, which processes each tile by vectorizing across all pixels within the tile. Details in Section 5.

**Expected Depth Mode**: If `render_mode` is "ED" (expected depth), the accumulated depth is normalized by the accumulated alpha to give the expected depth:

$$
D_{expected}(p) = \frac{\sum_i d_i \alpha_i T_i}{\sum_i \alpha_i T_i}
$$

---

## 5. File 4: `rasterization_utils.py` — Tile Intersection & Pixel Rasterization

This is the largest file (~978 lines) and contains the most algorithmically dense code. It implements two major subsystems: **tile intersection/sorting** and **alpha-compositing rasterization**.

### 5.1: `_compute_view_dirs_packed` — View Direction Computation

Computes the (unnormalized) direction from each camera to each Gaussian, for SH evaluation. For a Gaussian `g` seen by camera `c`, the view direction is simply:

$$
\mathbf{d} = \boldsymbol{\mu}_g - \mathbf{p}_{cam,c}
$$

The function has two code paths:
- **Simple vectorized**: Direct indexing `means[batch_ids, gaussian_ids] - campos[batch_ids, camera_ids]`. Fast for small scenes.
- **Split loop**: For large scenes with pose optimization enabled (`campos.requires_grad`), iterates over each (batch, camera) pair separately. This avoids a pathological many-to-one gradient scatter in the backward pass where all nnz gradients would write to the same `campos` entries.

### 5.2: `torch_isect_tiles` — Tile Intersection

#### What It Does

For each projected 2D Gaussian, determine which 16×16 tiles it overlaps. Then encode each (image, tile, Gaussian, depth) intersection into a single 64-bit integer for sorting.

#### The Algorithm

1. **Convert to tile space**: Divide pixel-space means and radii by `tile_size` to get tile-space coordinates.

2. **Compute tile bounding box**: For Gaussian `g`:
   ```
   tile_min_x = floor(tile_x - tile_rx),   tile_max_x = ceil(tile_x + tile_rx)
   tile_min_y = floor(tile_y - tile_ry),   tile_max_y = ceil(tile_y + tile_ry)
   ```
   Clamped to `[0, tile_width)` and `[0, tile_height)`.

3. **Count tiles per Gaussian**: `tiles_per_gauss = (max_x - min_x) * (max_y - min_y)`.

4. **Enumerate intersections**: For each Gaussian, iterate over its tile bounding box and emit one intersection per tile.

5. **Encode intersection IDs**: Each intersection is packed into an int64:

   ```
   isect_id = (image_id << (32 + tile_n_bits)) | (tile_id << 32) | depth_as_u32
   ```

   - `image_id`: which image (batch × camera) this belongs to
   - `tile_id`: `row * tile_width + col`
   - `depth_as_u32`: the float32 depth reinterpreted as uint32

   **Why this encoding?** When you sort these int64 values, you get all intersections grouped by image, then by tile within each image, then by depth within each tile — exactly the front-to-back order needed for alpha compositing. This is the same trick used in the CUDA kernel.

6. **Sort**: `torch.argsort(isect_ids)` gives the global ordering. `flatten_ids` (which Gaussian produced each intersection) is reordered accordingly.

#### Supplementary: Why Bit-Packing Depth?

The depth is a float32, but we need it as part of an integer sort key. The key insight: **IEEE 754 float32 positive values, when reinterpreted as unsigned integers, maintain their ordering**. So `reinterpret_cast<uint32_t>(depth)` gives a sortable integer. The code does this with:
```python
depth_i32 = depths.view(torch.int32)
depth_u32 = depth_i32.to(torch.int64) & 0xFFFFFFFF
```

### 5.3: `torch_isect_offset_encode` — Building the Offset Table

After sorting, we need to quickly find "for tile T in image I, where do its intersections start and end in the sorted array?" This function builds an offset table `[I, tile_height, tile_width]` where `offsets[i, ty, tx]` is the index of the first intersection belonging to tile `(ty, tx)` in image `i`.

It works by:
1. Decoding the `image_id` and `tile_id` from each sorted `isect_id`.
2. Computing a linear `tile_slot = image_id * n_tiles + tile_id`.
3. Using `torch.searchsorted` to find the first occurrence of each possible tile_slot value.

This is equivalent to the `isect_offset_encode` CUDA kernel in gsplat.

### 5.4: Alpha Compositing — Three Implementations

The file provides **three different implementations** of the same algorithm (front-to-back alpha compositing), at different levels of vectorization. All produce identical results. The default used by `rendering.py` is the **Gaussian Merge** version.

#### The Core Alpha-Compositing Formula

For a pixel `p`, the Gaussians overlapping its tile are processed front-to-back (by depth). For Gaussian `i`:

1. **Compute displacement**: $\Delta x = \mu_x^{(i)} - p_x$, $\Delta y = \mu_y^{(i)} - p_y$

2. **Evaluate the Gaussian**: Using the conic (inverse covariance) `(A, B, C)`:

$$
\sigma = \frac{1}{2}(A \cdot \Delta x^2 + C \cdot \Delta y^2) + B \cdot \Delta x \cdot \Delta y
$$

This is the quadratic form $\frac{1}{2} \mathbf{d}^T \Sigma^{-1} \mathbf{d}$ expanded for the symmetric 2×2 case. Note the factor of 2 on the off-diagonal term is absorbed because the conics store the full inverse (not just upper triangle times 2).

3. **Compute alpha**:

$$
\alpha_i = \min(0.999, \; o_i \cdot e^{-\sigma})
$$

where $o_i$ is the Gaussian's opacity. Clamped to prevent full occlusion from a single Gaussian.

4. **Accumulate color** using volume-rendering-style compositing:

$$
C_p = \sum_{i=1}^{M} c_i \cdot \alpha_i \cdot T_i, \quad \text{where } T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)
$$

$T_i$ is the **transmittance** — the fraction of light that has not yet been absorbed by Gaussians 1 through $i-1$. $T_1 = 1$ (nothing in front).

5. **Early termination**: If $T_i < 10^{-4}$, the pixel is essentially fully opaque and remaining Gaussians are skipped.

6. **Background**: After all Gaussians, remaining transmittance is filled with background color:

$$
C_p = C_p^{accumulated} + T_{final} \cdot C_{background}
$$

7. **Alpha output**: $\alpha_p = 1 - T_{final}$.

#### Implementation A: `torch_rasterize_to_pixels_per_tile_per_pixel_per_gauss`

The most literal, educational implementation. **Four nested Python loops**: image → tile → pixel → Gaussian. Each Gaussian is evaluated at each pixel individually using scalar Python math. Extremely slow (the comments note ~11 minutes for a typical scene) but perfectly maps to the algorithm description.

This is invaluable for debugging because you can set a breakpoint at any pixel/Gaussian and inspect every intermediate value.

#### Implementation B: `torch_rasterize_to_pixels_gaussian_merge` (DEFAULT)

**Three nested loops** (image → tile → Gaussian), but **all pixels within a tile are processed simultaneously** as a vectorized batch.

For a tile with `h×w` pixels and `M` Gaussians:
1. Create pixel coordinate grids: `px [P]`, `py [P]` where `P = h*w`.
2. For all M Gaussians, compute `dx [M, P]`, `dy [M, P]` — the displacement from every Gaussian center to every pixel.
3. Vectorized sigma: `power = -0.5 * (A·dx² + 2B·dx·dy + C·dy²)` → shape `[M, P]`.
4. Alpha: `clamp(opacity * exp(power), max=0.999)` → shape `[M, P]`.
5. Transmittance via **exclusive cumulative product**:
   ```python
   T = cumprod(cat([ones(1,P), 1-alpha], dim=0), dim=0)[:-1]   # [M, P]
   ```
   This is a clever trick: prepending a row of ones and taking cumprod gives $T_i = \prod_{j<i}(1-\alpha_j)$ without a loop.
6. Visibility weights: `vis = alpha * T` → shape `[M, P]`.
7. Final color: `(vis.T @ col)` — a single matrix multiply: `[P, M] @ [M, channels]` → `[P, channels]`.

**Key insight**: This version doesn't implement per-pixel early stopping. It processes all M Gaussians for all P pixels. For correctness this is fine (the transmittance naturally drives contributions to near-zero), but it wastes computation on fully-occluded pixels. The CUDA version handles this with per-pixel early termination within warps.

#### Implementation C: `torch_rasterize_to_pixels_pixels_vectorized`

**Three nested loops** like B, but iterates over Gaussians **sequentially** while keeping pixels vectorized. It maintains a running `T [Ht, Wt]` transmittance tensor and accumulates `pix [Ht, Wt, C]` color.

Per Gaussian, it:
1. Computes `sigma`, `alpha` as 2D grids over the tile.
2. Creates a `valid` mask: `(sigma >= 0) & (alpha >= threshold) & (T > early_stop)`.
3. Updates `pix += (alpha * T)[..., None] * color` and `T = where(valid, T*(1-alpha), T)`.

This is closer to the CUDA kernel's logic (sequential Gaussian processing with per-pixel early stopping via the `valid` mask), while still vectorizing across all pixels in the tile.

---

## 6. File 5: `sh_utils.py` — Spherical Harmonics Color Evaluation

### Role in the 3DGS Algorithm

In 3DGS, each Gaussian doesn't store a single RGB color. Instead, it stores **Spherical Harmonics (SH) coefficients** that encode a function on the sphere. Given a viewing direction, the SH function is evaluated to produce the RGB color. This enables **view-dependent appearance** — the same Gaussian can look different from different angles, capturing effects like specular highlights and reflections.

### The Math of Spherical Harmonics

SH basis functions $Y_l^m(\theta, \phi)$ form an orthonormal basis on the sphere, analogous to how Fourier basis functions work on a line. Each "band" $l$ has $2l+1$ functions, for a total of $(l+1)^2$ functions up to degree $l$.

For real-valued SH (used in graphics), we can express everything in terms of Cartesian unit direction $(x, y, z)$ instead of spherical angles. The color at direction **d** is:

$$
c(\mathbf{d}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} k_l^m \cdot Y_l^m(\mathbf{d})
$$

where $k_l^m$ are the learned SH coefficients (3 per basis function, one for each color channel).

### The SH Basis Functions (Hardcoded)

The file defines constants `C0` through `C4` and evaluates up to degree 4 (25 coefficients per color channel):

**Degree 0** (1 function — constant/diffuse):

$$
Y_0^0 = C_0 = 0.2821
$$

This is just a constant. With degree 0 only, you get view-independent color (like the original paper's SH degree 0 mode).

**Degree 1** (3 functions — linear/directional):

$$
Y_1^{-1} = C_1 \cdot y, \quad Y_1^0 = C_1 \cdot z, \quad Y_1^1 = C_1 \cdot x
$$

where $C_1 = 0.4886$. These capture broad directional variation.

**Degree 2** (5 functions — quadratic):

$$
Y_2^{-2} = C_2^0 \cdot xy, \quad Y_2^{-1} = C_2^1 \cdot yz, \quad Y_2^0 = C_2^2 \cdot (2z^2 - x^2 - y^2)
$$
$$
Y_2^1 = C_2^3 \cdot xz, \quad Y_2^2 = C_2^4 \cdot (x^2 - y^2)
$$

These capture more complex angular variation like specular lobes.

**Degrees 3 and 4** follow the same pattern with increasingly complex polynomials in $(x, y, z)$. The code hardcodes all the polynomial expressions and their normalization constants.

### `eval_sh(deg, sh, dirs)`

The function evaluates:

```python
result = C0 * sh[..., 0]    # degree 0
if deg > 0:
    result += -C1*y*sh[...,1] + C1*z*sh[...,2] - C1*x*sh[...,3]   # degree 1
if deg > 1:
    # ... degree 2 terms
# etc.
```

The `sh` tensor has shape `[..., C, K]` where K is the number of SH coefficients and C is the number of color channels (3 for RGB). The `dirs` tensor has shape `[..., 3]`.

**Note the negative signs**: The $Y_1^{-1}$ and $Y_1^1$ terms have negative signs. This is a convention choice — some implementations negate certain SH bands. As long as the SH coefficients are trained consistently with these signs, the results are correct.

### `build_color(sh_degree, shs, rays_d)`

A wrapper that:
1. Permutes the SH tensor from `[N, K, 3]` to a shape compatible with `eval_sh` (which expects `[..., C, K]`).
2. Calls `eval_sh`.
3. Adds 0.5 and clamps to `[0, ∞)`. The 0.5 offset comes from the convention that SH coefficient `sh0 = (color - 0.5) / C0`, so the inverse is `color = eval_sh * C0 * (1/C0) + 0.5 = eval_sh + 0.5`. Actually, looking at the code: `result = C0 * sh[..., 0]` already multiplies by C0, so the SH coefficients stored in the model are `(rgb - 0.5) / C0` and evaluation gives back `rgb - 0.5`. Adding 0.5 recovers the original color space.

### Supplementary: Why SH Instead of Just RGB?

If you only store a single RGB per Gaussian, the scene looks flat — like diffuse-only rendering. Real objects have view-dependent appearance: metals reflect, glass refracts, even matte objects have subtle shading variations. SH coefficients let each Gaussian "know" what color it should appear from each viewing angle, which is critical for photorealistic novel view synthesis.

The trade-off: each Gaussian now stores $(L+1)^2 \times 3$ color parameters instead of just 3. At degree 3 (the default in the original 3DGS paper), that's $16 \times 3 = 48$ color parameters per Gaussian. TorchSplat supports up to degree 4 (75 parameters).

---

## 7. Comparison with gsplat (nerfstudio)

TorchSplat is explicitly modeled after the gsplat library. Here's how they correspond:

| gsplat (CUDA) | TorchSplat (PyTorch) | Notes |
|---|---|---|
| `fully_fused_projection_packed_fwd` (CUDA kernel) | `torch_splat_fully_fused_projection_batch` | gsplat fuses projection, covariance computation, culling, and conic computation into a single CUDA kernel. TorchSplat does these as sequential PyTorch ops. |
| `isect_tiles` (CUDA kernel) | `torch_isect_tiles` | gsplat uses atomics and parallel prefix sums. TorchSplat uses a Python loop over Gaussians. |
| `isect_offset_encode` (CUDA kernel) | `torch_isect_offset_encode` | gsplat uses a parallel scan; TorchSplat uses `searchsorted`. |
| `rasterize_to_pixels_fwd` (CUDA kernel) | Three `torch_rasterize_to_pixels_*` variants | gsplat uses shared memory, warp-level primitives, and per-pixel early termination. TorchSplat vectorizes across pixels within a tile. |
| `SphericalHarmonics` (CUDA) | `eval_sh` / `build_color` | Mathematically identical. gsplat fuses SH evaluation into the projection kernel. |
| `rasterization()` Python API | `torch_rasterization()` | Nearly identical function signatures. TorchSplat preserves the same argument names and return format. |

### Key Differences

1. **Memory**: TorchSplat's tile intersection uses a Python loop over Gaussians and materializes large intermediate tensors (`[M, P]` grids in the Gaussian Merge rasterizer). gsplat's CUDA kernels use shared memory and process one Gaussian at a time per thread block, using orders of magnitude less memory.

2. **Speed**: The Python loops in tile intersection are extremely slow for large scenes. gsplat's kernels run in milliseconds; TorchSplat may take minutes for the same scene.

3. **Backward pass**: TorchSplat gets autograd for free (all ops are standard PyTorch). gsplat implements custom CUDA backward kernels for efficiency.

4. **Packed mode only**: TorchSplat currently only supports packed mode. gsplat supports both packed (memory-efficient) and unpacked (potentially faster for small scenes) modes.

5. **Radii**: gsplat uses a single circular radius; TorchSplat uses per-axis `(radius_x, radius_y)` radii, which is tighter for elongated Gaussians.

---

## 8. End-to-End Walkthrough: From 3D Gaussians to Pixels

Let's trace a single rendering call through the entire codebase.

**Setup**: You have 10,000 Gaussians and one camera.

### Step 1: Entry (`rasterizer_torchsplat.py`)

```python
rasterizer = Rasterizer(white_bkgd=True)
render_colors, render_alphas = rasterizer.gpu_rasterize_splats(
    w2c=viewmat, Knorm=K_normalized, width=800, height=600,
    tile_size=16, splats=splats_dict, active_sh_degree=3
)
```

- Padded dimensions: 800→800 (already multiple of 16), 600→608 (next multiple of 16).
- `tile_grid` is created: 50×38 = 1900 tiles, each 16×16 pixels.
- `pix_coord` is created: 800×608 grid of (x,y) coordinates.
- Intrinsics are denormalized from [0,1] to pixel space.
- Calls `torch_rasterization(...)` in `rendering.py`.

### Step 2: Projection (`EWA_fully_fused_proj_packed.py`)

For each of the 10,000 Gaussians:
1. Transform mean to camera space → get depth z and pixel position (u,v).
2. Cull if z < 0.01 or z > 1e10. Say 9,500 survive.
3. Build 3×3 rotation matrix from quaternion.
4. Compute Σ₃D = R·S·Sᵀ·Rᵀ.
5. Compute Jacobian J at the Gaussian's camera-space position.
6. Compute Σ₂D = J·(R_view·Σ₃D·R_viewᵀ)·Jᵀ + 0.3·I (with low-pass filter).
7. Invert Σ₂D to get conics (A, B, C).
8. Compute per-axis radii from diagonal of Σ₂D and opacity.
9. Cull if radii too small or bounding box outside image. Say 8,000 survive.

**Output**: packed tensors with nnz=8,000 entries.

### Step 3: SH Color Evaluation (`sh_utils.py`)

For each of the 8,000 surviving Gaussians:
1. Compute view direction: `d = mean_world - cam_position`.
2. Extract (x,y,z) components of the direction.
3. Evaluate SH up to degree 3: weighted sum of 16 basis functions → 3 values (R,G,B).
4. Add 0.5, clamp to [0, ∞).

**Output**: `colors [8000, 3]`.

### Step 4: Tile Intersection (`rasterization_utils.py`)

For each of the 8,000 Gaussians:
1. Convert pixel-space mean and radii to tile space (divide by 16).
2. Compute tile bounding box: e.g., a Gaussian at pixel (320, 240) with radius (24, 16) covers tiles (18,14) to (22,16) = ~12 tiles.
3. For each covered tile, emit one intersection.

Say total intersections = 150,000 (average ~19 tiles per Gaussian).

Each intersection is encoded as: `isect_id = (0 << 38) | (tile_id << 32) | depth_u32`.

Sort all 150,000 `isect_ids` → now intersections are ordered by tile, then by depth within each tile.

Build offset table: `isect_offsets[0, ty, tx]` = first intersection index for tile (ty, tx).

### Step 5: Per-Pixel Rasterization (`rasterization_utils.py`)

For each of the 1900 tiles:
1. Look up the range `[start, end)` in the sorted intersection list.
2. Gather the means2d, conics, colors, opacities for those Gaussians.
3. Create a 16×16 pixel grid for this tile (256 pixels).
4. For all M Gaussians in the tile (say M=80):
   - Compute `dx [80, 256]`, `dy [80, 256]`.
   - Compute `sigma [80, 256]`, `alpha [80, 256]`.
   - Compute transmittance via exclusive cumprod along Gaussian axis.
   - Compute `vis = alpha * T` → `[80, 256]`.
   - Matrix multiply: `vis.T @ colors` → `[256, 3]`.
5. Reshape to `[16, 16, 3]` and write to the output image.
6. Fill remaining transmittance with white background.

### Final Output

- `render_colors [1, 600, 800, 3]`: the rendered RGB image.
- `render_alphas [1, 600, 800, 1]`: per-pixel opacity (useful for compositing or loss masking).

Because every operation used standard PyTorch ops, calling `.backward()` on a loss computed from `render_colors` will propagate gradients all the way back through the rasterization, SH evaluation, projection, and into the Gaussian parameters (means, quaternions, scales, opacities, SH coefficients). This is what drives the 3DGS optimization loop.

---

*Document generated as a reference for the TorchSplat repository (github.com/Akirahai/TestOnly), a pure-PyTorch implementation of 3D Gaussian Splatting inspired by the gsplat library (github.com/nerfstudio-project/gsplat).*
