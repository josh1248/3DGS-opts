"""Quaternion -> 3x3 rotation matrix on a2: R = quat_to_mat(normalize(r)).

Contract (on a2, with a layout transpose at the boundary):
    r : float32 [4, N]   raw quaternion components in channel-major layout
                         (row 0 = all w's, row 1 = all x's, row 2 = y's, row 3 = z's)
    R : float32 [9, N]   rotation-matrix entries in channel-major layout
                         (rows 0..8 = R00, R01, R02, R10, R11, R12, R20, R21, R22)

Why the layout differs from the a5 version:
    a2 has no @vf / Reg scalar primitives. Vec math must happen on whole UB
    tensors. If we kept the original [N, 4] / [N, 9] layout, each per-component
    column would need a strided sub-C0 GM load, which fails UB memory
    validation. Transposing once at the caller boundary makes every component
    a contiguous 1-D strip in GM and aligns to the "[1, TILE]" a2 vec pattern.

Topology: vec-only. Device: a2 (b3).
Tail-safety: N may be arbitrary.
"""

from easyasc.a2 import *


TILE = 64  # elements per vec tile (C0-aligned; matches the validated a2 scalar column width)


@kernel()
def build_rotation_kernel(r: GMTensor, R: GMTensor, N: Var):
    # One UB buffer per quaternion component (each [1, TILE] = one C0-aligned row).
    ub_w = Tensor(DT.float, [1, TILE], Position.UB)
    ub_x = Tensor(DT.float, [1, TILE], Position.UB)
    ub_y = Tensor(DT.float, [1, TILE], Position.UB)
    ub_z = Tensor(DT.float, [1, TILE], Position.UB)

    # Scratch tensors reused across the 9 output computations.
    ub_a = Tensor(DT.float, [1, TILE], Position.UB)
    ub_b = Tensor(DT.float, [1, TILE], Position.UB)
    ub_norm = Tensor(DT.float, [1, TILE], Position.UB)

    # 9 output channels (R00..R22 in row-major 3x3 flattening order).
    ub_R00 = Tensor(DT.float, [1, TILE], Position.UB)
    ub_R01 = Tensor(DT.float, [1, TILE], Position.UB)
    ub_R02 = Tensor(DT.float, [1, TILE], Position.UB)
    ub_R10 = Tensor(DT.float, [1, TILE], Position.UB)
    ub_R11 = Tensor(DT.float, [1, TILE], Position.UB)
    ub_R12 = Tensor(DT.float, [1, TILE], Position.UB)
    ub_R20 = Tensor(DT.float, [1, TILE], Position.UB)
    ub_R21 = Tensor(DT.float, [1, TILE], Position.UB)
    ub_R22 = Tensor(DT.float, [1, TILE], Position.UB)

    # Split tiles across vec sub-blocks (20 cube cores * 2 sub-blocks = 40 vec lanes on a2).
    n_tiles = CeilDiv(N, TILE)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    t_begin = Var(tiles_per_core * GetVecIdx())
    t_end = Min(t_begin + tiles_per_core, n_tiles)

    with auto_sync():
        for ti in range(t_begin, t_end):
            col0 = Var(ti * TILE)
            valid = Min(TILE, N - col0)

            # Contiguous loads: one row of r per component.
            ub_w <<= r[0:1, col0:col0 + valid]
            ub_x <<= r[1:2, col0:col0 + valid]
            ub_y <<= r[2:3, col0:col0 + valid]
            ub_z <<= r[3:4, col0:col0 + valid]

            # norm = sqrt(w² + x² + y² + z²)
            mul(ub_a, ub_w, ub_w)
            mul(ub_b, ub_x, ub_x)
            add(ub_a, ub_a, ub_b)
            mul(ub_b, ub_y, ub_y)
            add(ub_a, ub_a, ub_b)
            mul(ub_b, ub_z, ub_z)
            add(ub_a, ub_a, ub_b)
            sqrt(ub_norm, ub_a)

            # Normalize quaternion in place.
            div(ub_w, ub_w, ub_norm)
            div(ub_x, ub_x, ub_norm)
            div(ub_y, ub_y, ub_norm)
            div(ub_z, ub_z, ub_norm)

            # R00 = 1 - 2*(y² + z²)
            mul(ub_a, ub_y, ub_y)
            mul(ub_b, ub_z, ub_z)
            add(ub_a, ub_a, ub_b)
            muls(ub_a, ub_a, -2.0)
            adds(ub_R00, ub_a, 1.0)

            # R01 = 2*(x*y - w*z)
            mul(ub_a, ub_x, ub_y)
            mul(ub_b, ub_w, ub_z)
            sub(ub_a, ub_a, ub_b)
            muls(ub_R01, ub_a, 2.0)

            # R02 = 2*(x*z + w*y)
            mul(ub_a, ub_x, ub_z)
            mul(ub_b, ub_w, ub_y)
            add(ub_a, ub_a, ub_b)
            muls(ub_R02, ub_a, 2.0)

            # R10 = 2*(x*y + w*z)
            mul(ub_a, ub_x, ub_y)
            mul(ub_b, ub_w, ub_z)
            add(ub_a, ub_a, ub_b)
            muls(ub_R10, ub_a, 2.0)

            # R11 = 1 - 2*(x² + z²)
            mul(ub_a, ub_x, ub_x)
            mul(ub_b, ub_z, ub_z)
            add(ub_a, ub_a, ub_b)
            muls(ub_a, ub_a, -2.0)
            adds(ub_R11, ub_a, 1.0)

            # R12 = 2*(y*z - w*x)
            mul(ub_a, ub_y, ub_z)
            mul(ub_b, ub_w, ub_x)
            sub(ub_a, ub_a, ub_b)
            muls(ub_R12, ub_a, 2.0)

            # R20 = 2*(x*z - w*y)
            mul(ub_a, ub_x, ub_z)
            mul(ub_b, ub_w, ub_y)
            sub(ub_a, ub_a, ub_b)
            muls(ub_R20, ub_a, 2.0)

            # R21 = 2*(y*z + w*x)
            mul(ub_a, ub_y, ub_z)
            mul(ub_b, ub_w, ub_x)
            add(ub_a, ub_a, ub_b)
            muls(ub_R21, ub_a, 2.0)

            # R22 = 1 - 2*(x² + y²)
            mul(ub_a, ub_x, ub_x)
            mul(ub_b, ub_y, ub_y)
            add(ub_a, ub_a, ub_b)
            muls(ub_a, ub_a, -2.0)
            adds(ub_R22, ub_a, 1.0)

            # Contiguous stores: one row of R per output channel.
            R[0:1, col0:col0 + valid] <<= ub_R00
            R[1:2, col0:col0 + valid] <<= ub_R01
            R[2:3, col0:col0 + valid] <<= ub_R02
            R[3:4, col0:col0 + valid] <<= ub_R10
            R[4:5, col0:col0 + valid] <<= ub_R11
            R[5:6, col0:col0 + valid] <<= ub_R12
            R[6:7, col0:col0 + valid] <<= ub_R20
            R[7:8, col0:col0 + valid] <<= ub_R21
            R[8:9, col0:col0 + valid] <<= ub_R22

    return R


if __name__ == "__main__":
    import torch

    def build_rotation_torch(r_nx4: torch.Tensor) -> torch.Tensor:
        # CPU reference in the original [N, 4] convention, mirroring
        # 3DGS-opts/rasterizer_torchsplat.py::build_rotation with `device='cuda'` dropped.
        norm = torch.sqrt(r_nx4[:, 0] * r_nx4[:, 0] + r_nx4[:, 1] * r_nx4[:, 1]
                          + r_nx4[:, 2] * r_nx4[:, 2] + r_nx4[:, 3] * r_nx4[:, 3])
        q = r_nx4 / norm[:, None]
        R = torch.zeros((q.size(0), 3, 3), dtype=r_nx4.dtype)
        w = q[:, 0]; x = q[:, 1]; y = q[:, 2]; z = q[:, 3]
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R

    torch.manual_seed(0)

    for N in [64, 17, 90]:
        # Caller-side layout transpose (one-time; see module docstring for rationale).
        r_nx4 = torch.randn((N, 4), dtype=torch.float32)
        r_in = r_nx4.t().contiguous()                      # [N, 4] -> [4, N]
        R_in = torch.zeros((9, N), dtype=torch.float32)

        R_ref = build_rotation_torch(r_nx4).reshape(N, 9)  # [N, 9] reference
        R_out_9xN = OpExec(build_rotation_kernel, simulator=True)(r_in, R_in, N)
        R_out = R_out_9xN.t().contiguous()                 # [9, N] -> [N, 9]

        torch.testing.assert_close(R_out, R_ref, rtol=1e-4, atol=1e-4)
        diff = torch.abs(R_out - R_ref).max().item()
        print(f"N={N:>4}  max_abs_diff={diff:.3e}")
