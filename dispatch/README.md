# dispatch

Config-driven router between pure-PyTorch reference impls (already in
`3DGS-opts`) and easyasc kernel variants (being ported incrementally). Pick
which impl to use for each op via a single `DispatchConfig` of booleans.

## Why

`3DGS-opts` is being ported op-by-op from pure PyTorch to easyasc kernels.
While both versions coexist, callers should be able to pick the impl for each
op with a one-line flag flip, without edit-every-call-site churn. `dispatch/`
is the one place that tracks `(torch impl, kernel adapter)` pairs and the
runtime toggle.

## Usage

```python
# 3DGS-opts must be on sys.path — view_scene.py does this with the sys.path
# insert trick. If you import dispatch from elsewhere, do the same.
from dispatch import ops, DispatchConfig, set_config, using_config

# 1. Default: every op uses pure PyTorch. No behavior change.
R = ops.build_rotation(r)

# 2. Global toggle — subsequent ops.* calls use the kernel variant:
set_config(DispatchConfig(use_kernel_build_rotation=True, kernel_simulator=True))
R = ops.build_rotation(r)

# 3. Scoped override via context manager:
with using_config(DispatchConfig(use_kernel_build_rotation=True)):
    R = ops.build_rotation(r)

# 4. Per-call override (leaves the global config alone):
R = ops.build_rotation(r, _config=DispatchConfig(use_kernel_build_rotation=True))
```

If `use_kernel_<op>` is True but no kernel adapter is registered for that op,
the dispatcher warns and falls back to the torch impl — safe to flip toggles
before adapters land.

## How it picks an impl

For each `ops.<op>(...)` call:

1. Resolve the effective `DispatchConfig`:
   `_config=` kwarg ➜ global config from `set_config` ➜ default `DispatchConfig()`.
2. If `config.use_kernel_<op>` is True **and** a kernel adapter is registered,
   call the adapter with `_config=` forwarded.
3. Otherwise, call the torch impl.

## Files

| file | role |
|------|------|
| `config.py`          | `DispatchConfig` dataclass (one bool per op + shared kernel knobs) |
| `registry.py`        | `OpRegistry`, global config state, `dispatch()` helper, `using_config` |
| `ops.py`             | Public function surface (`ops.build_rotation(...)`, …) |
| `kernel_adapters.py` | Adapters that wrap easyasc kernels to match the torch signatures |
| `__init__.py`        | Wires everything together; registers torch impls + kernel adapters |

## Adding a new kernel adapter

Say the easyasc kernel version of `build_covariance_3d` lands in
`cov3d_kernel.py`. Three files change:

**1. `kernel_adapters.py`** — write an adapter and register it:

```python
def kernel_build_covariance_3d(s, r, *, _config=None):
    from cov3d_kernel import build_covariance_3d_kernel
    from easyasc.a5 import OpExec

    simulator = _resolve_simulator(_config)
    src_device = s.device
    s_cpu = s.detach().to(dtype=torch.float32).cpu().contiguous()
    r_cpu = r.detach().to(dtype=torch.float32).cpu().contiguous()
    cov3d = torch.zeros((s_cpu.shape[0], 3, 3), dtype=torch.float32)
    cov3d = OpExec(build_covariance_3d_kernel, simulator=simulator)(s_cpu, r_cpu, cov3d, s_cpu.shape[0])
    if cov3d.device != src_device:
        cov3d = cov3d.to(src_device)
    return cov3d

def register_all(registry):
    registry.register_kernel("build_rotation", kernel_build_rotation)
    registry.register_kernel("build_covariance_3d", kernel_build_covariance_3d)  # <-- new
```

The adapter **must**:

* accept the exact torch positional/keyword args, plus a trailing keyword-only
  `_config`.
* return a tensor of the same shape, dtype, and device as the torch impl would
  — so callers of `ops.build_covariance_3d` don't need to know which backend
  ran.

**2. `config.py`** — make sure the toggle exists. If it doesn't, add:

```python
use_kernel_build_covariance_3d: bool = False
```

Default **must** be `False`, to keep the pure-torch path as the baseline.

**3. (validate)** — flip the flag and compare against the torch output:

```python
import torch
from dispatch import ops, DispatchConfig

torch_out = ops.build_covariance_3d(s, r)  # default: torch
kernel_out = ops.build_covariance_3d(
    s, r, _config=DispatchConfig(use_kernel_build_covariance_3d=True)
)
torch.testing.assert_close(torch_out, kernel_out, rtol=1e-4, atol=1e-4)
```

## Op list

Each differentiable op has a separate row for **Forward** and **Backward** so the
table doubles as a kernel-port checklist: a kernel can land for the forward
half before its backward half exists. Ops decorated with `@torch.no_grad()` (or
that produce only integer indices / sort orders) appear once, as **Forward**
only.

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

Keep this table in sync as adapters land. The kernel-adapter column should call
out which half is wired (`✅ … (fwd)` vs `✅ … (fwd+bwd)`) so we can see at a
glance which ops still need a backward kernel before training can flow through
them end-to-end.

## Suggested porting order

The two axes that determine porting cost are (a) **stage in the pipeline** —
how many other ops depend on it, and (b) **gradient surface** — forward-only
ops are roughly half the work of differentiable ones, since you skip the
adjoint kernel and the `torch.autograd.Function` glue. The plan below walks
those two axes from "small/easy/independent" to "large/coupled", so each
landing kernel can be flag-flipped on in isolation against the torch baseline.

### Phase 0 — already done

* `build_rotation` forward kernel. Good warm-up: tiny, no fan-in, easy to
  validate with `torch.testing.assert_close` against the torch path.

### Phase 1 — leaf forwards, no_grad ops first

These are forward-only by construction (`@torch.no_grad()` in the source) and
have small input/output footprints. Nothing in the rest of the pipeline reads
their gradients, so once a kernel matches the torch output bit-for-bit we can
switch the flag and forget about it.

1. `inverse_cov2d` — pure elementwise on `(N, 3)` reads. Note: this one *is*
   differentiable in the torch path; keep the kernel forward-only initially
   and route the backward through the autograd.Function fallback.
2. `get_radius` — eigenvalue-of-a-2x2 expression; closed-form, no_grad.
3. `get_rect` — clip + add; trivial.
4. `isect_tiles` — sort + bucket. The sort is the interesting kernel here.
5. `isect_offset_encode` — prefix-sum / bucket index; complements `isect_tiles`.

These five are independent of each other and can land in parallel. They also
don't need autograd.Function wrapping, so the adapter pattern stays close to
what `build_rotation` already does.

### Phase 2 — differentiable forwards on the geometry chain

Now climb the geometry stack. These all require gradients in training, but in
this phase we only port the **forward** kernel and let the backward fall back
to torch via a wrapping `torch.autograd.Function`. That gives the kernel team
forward perf wins immediately while the adjoints are still being designed.

6. `build_scaling_rotation` — depends on `build_rotation`; small matmul.
7. `build_covariance_3d` — `L @ L^T`; one matmul, easy adjoint to plan later.
8. `compute_view_dirs_packed` — gather + subtract; the indexing pattern is
   the part worth getting right on NPU.
9. `projection_means2d_pinhole` — small per-point math, but produces `means_c`
   and `depths` that downstream stages depend on; validate carefully.
10. `eval_sh` then `build_color` — SH expansion is a fixed-degree polynomial
    over `(nnz, 3)` directions; ideal for an NPU vectorization pass.
11. `build_covariance_2d` — the most arithmetically dense forward in this
    group (Jacobian construction + two matmuls + low-pass filter). Save it
    for last in the phase so the simpler ones stress-test the adapter pattern
    first.

### Phase 3 — backwards for the geometry chain

Re-walk Phase 2 and add the backward kernel for each, in the same order.
Validate by comparing `loss.backward()` gradients against the torch path with
a tight `assert_close`. The order matters: bugs in earlier-stage adjoints
(e.g. `build_rotation` backward) will mask or amplify in later ones, so don't
skip ahead.

### Phase 4 — rasterizer

12. `rasterize_to_pixels` forward — the pixel-loop / tile loop is the single
    biggest perf prize and also the largest kernel surface. Treat it as its
    own milestone, not a continuation of Phase 2.
13. `rasterize_to_pixels` backward — separate milestone again. The accumulation
    pattern (front-to-back alpha compositing has a back-to-front gradient) is
    where most CUDA implementations spend their complexity budget; expect the
    same on NPU.

### Phase 5 — orchestration

14. `fully_fused_projection_batch` — keep this one in Python for as long as
    possible. Its job is to call the per-op kernels in the right order and
    apply masks; once the underlying ops are on the NPU, the orchestrator
    benefits "for free." Only replace it with a fused NPU kernel after the
    individual ops are stable, so we have a known-good reference to diff
    against.

### Rules of thumb while porting

* **One flag per landing.** Each PR flips exactly one `use_kernel_<op>` and
  ships a numerical-equivalence test against the torch path. This is the
  whole reason the dispatcher exists — keep the granularity.
* **Forward before backward, always.** A kernel that produces wrong forward
  outputs will produce confidently-wrong gradients; landing forward first
  gives a stable reference to diff the backward against.
* **Don't fuse across stage boundaries yet.** It's tempting to fuse, e.g.,
  `build_covariance_3d` + `build_covariance_2d`, but doing so before both
  forward and backward kernels exist independently makes regressions much
  harder to localize. Fusion is a Phase-6+ optimization.
* **Watch the dtype/device round-trips.** The current adapter pattern (see
  `kernel_build_covariance_3d` in the example above) goes
  `cuda → cpu → kernel → cpu → cuda`. That's fine for correctness during
  porting but will dominate runtime once kernels are fast — plan to remove
  the round-trips once the simulator is replaced by the real NPU path.

## Non-goals

* This dispatcher does **not** rewrite `rendering.py` or any other call site.
  Migrating call sites from direct imports to `from dispatch import ops` is
  intentionally left to the caller, one op at a time, so you can roll each
  kernel in with a targeted change.
* No autograd gluing — kernel adapters currently return detached tensors. If
  you need grad through a kernel adapter, wrap it in a `torch.autograd.Function`
  in the adapter itself.
