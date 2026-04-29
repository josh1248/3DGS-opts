"""3D covariance on a5: cov3d = R(q) @ diag(s)^2 @ R(q).T, expressed via the
identity   cov[u, v] = sum_j R[u, j] * R[v, j] * s[j]^2.

Contract (mirrors `3DGS-opts/pytorch/EWA_fully_fused_proj_packed.py::build_covariance_3d`,
but with the symmetric 3x3 output flattened to [N, 9] so the kernel only uses
shape-preserving 2D ops; the test wrapper reshapes to [N, 3, 3] outside):
    s   : float32 [N, 3]   per-axis scale
    r   : float32 [N, 4]   raw quaternion (w, x, y, z), unnormalized
    cov : float32 [N, 9]   row-major 3x3 per row (symmetric); reshape to [N, 3, 3] outside

Topology: vec-only (no matmul).
Tail-safety: N may be arbitrary. The second dim is fixed at 3 / 4 / 9.
Device: a5 (950). Work split across vec cores via GetVecNum / GetVecIdx.

Structure mirrors `3DGS-opts/A5/build_rotation.py`. The 9 R entries are computed
inline with the same formulas, then held in scalar Regs while the 6 unique cov
entries are accumulated and the 3 mirrored entries are emitted.
"""

from easyasc.a5 import *


CHUNK = 32          # rows per vf call (also the UB chunk tile)
S_COLS = 3          # per-row scale width
R_COLS = 4          # per-row raw quaternion width
COV_COLS = 9        # per-row flattened 3x3

# UB column padding. float32 C0 = 8 elements. Each row must be a multiple of C0.
# S_PAD   = ceil(3 / 8) * 8 = 8   (3 real + 5 junk)
# R_PAD   = ceil(4 / 8) * 8 = 8   (4 real + 4 junk)
# COV_PAD = ceil(9 / 8) * 8 = 16  (9 real + 7 junk)
S_PAD = 8
R_PAD = 8
COV_PAD = 16


@vf()
def build_covariance_3d_vf(sbuf: Tensor, rbuf: Tensor, Cbuf: Tensor, rows: Var):
    # Quaternion components and norm.
    w = Reg(DT.float)
    x = Reg(DT.float)
    y = Reg(DT.float)
    z = Reg(DT.float)
    norm = Reg(DT.float)

    # Nine R entries, held simultaneously so the cov stage can index any (u, j).
    R00 = Reg(DT.float); R01 = Reg(DT.float); R02 = Reg(DT.float)
    R10 = Reg(DT.float); R11 = Reg(DT.float); R12 = Reg(DT.float)
    R20 = Reg(DT.float); R21 = Reg(DT.float); R22 = Reg(DT.float)

    # Squared per-axis scales.
    sx2 = Reg(DT.float)
    sy2 = Reg(DT.float)
    sz2 = Reg(DT.float)

    # Temps and the output accumulator.
    t = Reg(DT.float)
    tmp = Reg(DT.float)
    acc = Reg(DT.float)
    one_r = Reg(DT.float)

    one_r.fill(1.0)

    for i in range(rows):
        # ---- Load + normalize quaternion (verbatim from build_rotation_vf) ----
        w <<= rbuf[i:i + 1, 0:1].single()
        x <<= rbuf[i:i + 1, 1:2].single()
        y <<= rbuf[i:i + 1, 2:3].single()
        z <<= rbuf[i:i + 1, 3:4].single()

        t <<= w * w
        tmp <<= x * x
        t <<= t + tmp
        tmp <<= y * y
        t <<= t + tmp
        tmp <<= z * z
        t <<= t + tmp
        norm <<= t.sqrt()

        w <<= w / norm
        x <<= x / norm
        y <<= y / norm
        z <<= z / norm

        # ---- Build all 9 R entries (same nine formulas as build_rotation_vf) ----
        # R00 = 1 - 2*(y^2 + z^2)
        t <<= y * y
        tmp <<= z * z
        t <<= t + tmp
        t <<= t * 2.0
        R00 <<= one_r - t

        # R01 = 2*(x*y - w*z)
        t <<= x * y
        tmp <<= w * z
        t <<= t - tmp
        R01 <<= t * 2.0

        # R02 = 2*(x*z + w*y)
        t <<= x * z
        tmp <<= w * y
        t <<= t + tmp
        R02 <<= t * 2.0

        # R10 = 2*(x*y + w*z)
        t <<= x * y
        tmp <<= w * z
        t <<= t + tmp
        R10 <<= t * 2.0

        # R11 = 1 - 2*(x^2 + z^2)
        t <<= x * x
        tmp <<= z * z
        t <<= t + tmp
        t <<= t * 2.0
        R11 <<= one_r - t

        # R12 = 2*(y*z - w*x)
        t <<= y * z
        tmp <<= w * x
        t <<= t - tmp
        R12 <<= t * 2.0

        # R20 = 2*(x*z - w*y)
        t <<= x * z
        tmp <<= w * y
        t <<= t - tmp
        R20 <<= t * 2.0

        # R21 = 2*(y*z + w*x)
        t <<= y * z
        tmp <<= w * x
        t <<= t + tmp
        R21 <<= t * 2.0

        # R22 = 1 - 2*(x^2 + y^2)
        t <<= x * x
        tmp <<= y * y
        t <<= t + tmp
        t <<= t * 2.0
        R22 <<= one_r - t

        # ---- Load scales and pre-square ----
        sx2 <<= sbuf[i:i + 1, 0:1].single()
        sy2 <<= sbuf[i:i + 1, 1:2].single()
        sz2 <<= sbuf[i:i + 1, 2:3].single()
        sx2 <<= sx2 * sx2
        sy2 <<= sy2 * sy2
        sz2 <<= sz2 * sz2

        # ---- Compute the 6 unique cov entries; mirror the 3 off-diagonals ----
        # cov[u, v] = R[u, 0] * R[v, 0] * sx2
        #          + R[u, 1] * R[v, 1] * sy2
        #          + R[u, 2] * R[v, 2] * sz2
        # Output layout (row-major 3x3 flattened to 9):
        #   [c00, c01, c02, c10, c11, c12, c20, c21, c22]
        # with c10 = c01, c20 = c02, c21 = c12.

        # cov[0, 0]
        t <<= R00 * R00
        acc <<= t * sx2
        t <<= R01 * R01
        tmp <<= t * sy2
        acc <<= acc + tmp
        t <<= R02 * R02
        tmp <<= t * sz2
        acc <<= acc + tmp
        Cbuf[i:i + 1, 0:1] <<= acc.single_value()

        # cov[0, 1]   (also written into cov[1, 0] mirror slot below)
        t <<= R00 * R10
        acc <<= t * sx2
        t <<= R01 * R11
        tmp <<= t * sy2
        acc <<= acc + tmp
        t <<= R02 * R12
        tmp <<= t * sz2
        acc <<= acc + tmp
        Cbuf[i:i + 1, 1:2] <<= acc.single_value()
        Cbuf[i:i + 1, 3:4] <<= acc.single_value()  # mirror -> cov[1, 0]

        # cov[0, 2]   (also -> cov[2, 0])
        t <<= R00 * R20
        acc <<= t * sx2
        t <<= R01 * R21
        tmp <<= t * sy2
        acc <<= acc + tmp
        t <<= R02 * R22
        tmp <<= t * sz2
        acc <<= acc + tmp
        Cbuf[i:i + 1, 2:3] <<= acc.single_value()
        Cbuf[i:i + 1, 6:7] <<= acc.single_value()  # mirror -> cov[2, 0]

        # cov[1, 1]
        t <<= R10 * R10
        acc <<= t * sx2
        t <<= R11 * R11
        tmp <<= t * sy2
        acc <<= acc + tmp
        t <<= R12 * R12
        tmp <<= t * sz2
        acc <<= acc + tmp
        Cbuf[i:i + 1, 4:5] <<= acc.single_value()

        # cov[1, 2]   (also -> cov[2, 1])
        t <<= R10 * R20
        acc <<= t * sx2
        t <<= R11 * R21
        tmp <<= t * sy2
        acc <<= acc + tmp
        t <<= R12 * R22
        tmp <<= t * sz2
        acc <<= acc + tmp
        Cbuf[i:i + 1, 5:6] <<= acc.single_value()
        Cbuf[i:i + 1, 7:8] <<= acc.single_value()  # mirror -> cov[2, 1]

        # cov[2, 2]
        t <<= R20 * R20
        acc <<= t * sx2
        t <<= R21 * R21
        tmp <<= t * sy2
        acc <<= acc + tmp
        t <<= R22 * R22
        tmp <<= t * sz2
        acc <<= acc + tmp
        Cbuf[i:i + 1, 8:9] <<= acc.single_value()


@kernel()
def build_covariance_3d_kernel(s: GMTensor, r: GMTensor, cov: GMTensor, N: Var):
    sbuf = DBuff(DT.float, [CHUNK, S_PAD], Position.UB)
    rbuf = DBuff(DT.float, [CHUNK, R_PAD], Position.UB)
    Cbuf = DBuff(DT.float, [CHUNK, COV_PAD], Position.UB)

    buf_cnt = Var(0)

    total_chunks = CeilDiv(N, CHUNK)
    chunks_per_core = CeilDiv(total_chunks, GetVecNum())
    chunk_begin = Var(chunks_per_core * GetVecIdx())
    chunk_end = Min(chunk_begin + chunks_per_core, total_chunks)

    with auto_sync():
        for chunk_idx in range(chunk_begin, chunk_end):
            row0 = Var(chunk_idx * CHUNK)
            valid_rows = Min(CHUNK, N - row0)

            # GM [valid_rows, 3] -> UB [valid_rows, 8] (3 real + 5 junk per row).
            sbuf[buf_cnt] <<= s[row0:row0 + valid_rows, 0:S_COLS]
            # GM [valid_rows, 4] -> UB [valid_rows, 8] (4 real + 4 junk per row).
            rbuf[buf_cnt] <<= r[row0:row0 + valid_rows, 0:R_COLS]

            build_covariance_3d_vf(sbuf[buf_cnt], rbuf[buf_cnt], Cbuf[buf_cnt], valid_rows)

            # UB [valid_rows, 16] -> GM [valid_rows, 9] (drop the 7 junk cols per row).
            cov[row0:row0 + valid_rows, 0:COV_COLS] <<= Cbuf[buf_cnt][0:valid_rows, 0:COV_COLS]

            buf_cnt += 1

    return cov


if __name__ == "__main__":
    import torch

    def build_covariance_3d_torch(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # Inlined CPU reference; mirrors
        #   3DGS-opts/pytorch/EWA_fully_fused_proj_packed.py::build_covariance_3d
        # composed with build_scaling_rotation + build_rotation, with the
        # `device='cuda'` lines dropped so it runs without a GPU.
        N = r.shape[0]
        norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1]
                          + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
        q = r / norm[:, None]
        w = q[:, 0]; x = q[:, 1]; y = q[:, 2]; z = q[:, 3]

        R = torch.zeros((N, 3, 3), dtype=r.dtype)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)

        L = torch.zeros((N, 3, 3), dtype=r.dtype)
        L[:, 0, 0] = s[:, 0]
        L[:, 1, 1] = s[:, 1]
        L[:, 2, 2] = s[:, 2]
        L = R @ L

        return L @ L.transpose(1, 2)

    torch.manual_seed(0)

    # Per session budget: N <= 60, at most 2 instances. User runs larger N.
    # N = 17 -> tail-only path (one partial chunk).
    # N = 60 -> one full CHUNK=32 + 28-row tail.
    for N in [17, 60]:
        s = torch.randn((N, S_COLS), dtype=torch.float32)
        r = torch.randn((N, R_COLS), dtype=torch.float32)
        cov_3d = torch.zeros((N, 3, 3), dtype=torch.float32)

        ref = build_covariance_3d_torch(s, r)

        out_2d = OpExec(build_covariance_3d_kernel, simulator=True)(
            s, r, cov_3d.view(N, COV_COLS), N,
        )
        out_3d = out_2d.reshape(N, 3, 3)

        torch.testing.assert_close(out_3d, ref, rtol=1e-4, atol=1e-4)
        diff = torch.abs(out_3d - ref).max().item()
        print(f"N={N:>5}  max_abs_diff={diff:.3e}")
