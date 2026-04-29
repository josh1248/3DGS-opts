"""Quaternion -> 3x3 rotation matrix on a5: R = quat_to_mat(normalize(r)).

Contract (mirrors `3DGS-opts/rasterizer_torchsplat.py::build_rotation`,
but with R flattened to [N, 9] so the kernel only uses shape-preserving ops):
    r : float32 [N, 4]   raw quaternion (w, x, y, z)
    R : float32 [N, 9]   row-major 3x3 per row; reshape to [N, 3, 3] outside

Topology: vec-only (no matmul).
Tail-safety: N may be arbitrary. The second dim is fixed at 4 / 9.
Device: a5 (950). Work split across vec cores via GetVecNum / GetVecIdx.
"""

from easyasc.a5 import *


CHUNK = 32          # rows per vf call (also the UB chunk tile)
IN_COLS = 4         # raw quaternion width
OUT_COLS = 9        # flattened 3x3 width

# UB column padding. float32 C0 = 8 elements. Each row must be a multiple of C0.
# IN_PAD  = ceil(IN_COLS  / 8) * 8 = 8   (one C0 block per row)
# OUT_PAD = ceil(OUT_COLS / 8) * 8 = 16  (two C0 blocks per row)
IN_PAD = 8
OUT_PAD = 16


@vf()
def build_rotation_vf(qbuf: Tensor, Rbuf: Tensor, rows: Var):
    w = Reg(DT.float)
    x = Reg(DT.float)
    y = Reg(DT.float)
    z = Reg(DT.float)
    norm = Reg(DT.float)
    t = Reg(DT.float)
    s = Reg(DT.float)
    one_r = Reg(DT.float)
    out_r = Reg(DT.float)

    one_r.fill(1.0)

    for i in range(rows):
        # Load one raw quaternion (w, x, y, z).
        w <<= qbuf[i:i + 1, 0:1].single()
        x <<= qbuf[i:i + 1, 1:2].single()
        y <<= qbuf[i:i + 1, 2:3].single()
        z <<= qbuf[i:i + 1, 3:4].single()

        # norm = sqrt(w^2 + x^2 + y^2 + z^2)
        t <<= w * w
        s <<= x * x
        t <<= t + s
        s <<= y * y
        t <<= t + s
        s <<= z * z
        t <<= t + s
        norm <<= t.sqrt()

        # Normalize quaternion in place (scalar registers).
        w <<= w / norm
        x <<= x / norm
        y <<= y / norm
        z <<= z / norm

        # R[0,0] = 1 - 2*(y^2 + z^2)
        t <<= y * y
        s <<= z * z
        t <<= t + s
        t <<= t * 2.0
        out_r <<= one_r - t
        Rbuf[i:i + 1, 0:1] <<= out_r.single_value()

        # R[0,1] = 2*(x*y - w*z)
        t <<= x * y
        s <<= w * z
        t <<= t - s
        out_r <<= t * 2.0
        Rbuf[i:i + 1, 1:2] <<= out_r.single_value()

        # R[0,2] = 2*(x*z + w*y)
        t <<= x * z
        s <<= w * y
        t <<= t + s
        out_r <<= t * 2.0
        Rbuf[i:i + 1, 2:3] <<= out_r.single_value()

        # R[1,0] = 2*(x*y + w*z)
        t <<= x * y
        s <<= w * z
        t <<= t + s
        out_r <<= t * 2.0
        Rbuf[i:i + 1, 3:4] <<= out_r.single_value()

        # R[1,1] = 1 - 2*(x^2 + z^2)
        t <<= x * x
        s <<= z * z
        t <<= t + s
        t <<= t * 2.0
        out_r <<= one_r - t
        Rbuf[i:i + 1, 4:5] <<= out_r.single_value()

        # R[1,2] = 2*(y*z - w*x)
        t <<= y * z
        s <<= w * x
        t <<= t - s
        out_r <<= t * 2.0
        Rbuf[i:i + 1, 5:6] <<= out_r.single_value()

        # R[2,0] = 2*(x*z - w*y)
        t <<= x * z
        s <<= w * y
        t <<= t - s
        out_r <<= t * 2.0
        Rbuf[i:i + 1, 6:7] <<= out_r.single_value()

        # R[2,1] = 2*(y*z + w*x)
        t <<= y * z
        s <<= w * x
        t <<= t + s
        out_r <<= t * 2.0
        Rbuf[i:i + 1, 7:8] <<= out_r.single_value()

        # R[2,2] = 1 - 2*(x^2 + y^2)
        t <<= x * x
        s <<= y * y
        t <<= t + s
        t <<= t * 2.0
        out_r <<= one_r - t
        Rbuf[i:i + 1, 8:9] <<= out_r.single_value()


@kernel()
def build_rotation_kernel(r: GMTensor, R: GMTensor, N: Var):
    qbuf = DBuff(DT.float, [CHUNK, IN_PAD], Position.UB)
    Rbuf = DBuff(DT.float, [CHUNK, OUT_PAD], Position.UB)

    buf_cnt = Var(0)

    total_chunks = CeilDiv(N, CHUNK)
    chunks_per_core = CeilDiv(total_chunks, GetVecNum())
    chunk_begin = Var(chunks_per_core * GetVecIdx())
    chunk_end = Min(chunk_begin + chunks_per_core, total_chunks)

    with auto_sync():
        for chunk_idx in range(chunk_begin, chunk_end):
            row0 = Var(chunk_idx * CHUNK)
            valid_rows = Min(CHUNK, N - row0)

            # GM [valid_rows, 4] -> UB [valid_rows, 8] (4 real + 4 junk per row).
            qbuf[buf_cnt] <<= r[row0:row0 + valid_rows, 0:IN_COLS]

            build_rotation_vf(qbuf[buf_cnt], Rbuf[buf_cnt], valid_rows)

            # UB [valid_rows, 16] -> GM [valid_rows, 9] (drop the 7 junk cols per row).
            R[row0:row0 + valid_rows, 0:OUT_COLS] <<= Rbuf[buf_cnt][0:valid_rows, 0:OUT_COLS]

            buf_cnt += 1

    return R


if __name__ == "__main__":
    import torch

    def build_rotation_torch(r: torch.Tensor) -> torch.Tensor:
        # CPU-side reference; mirrors 3DGS-opts/rasterizer_torchsplat.py::build_rotation
        # with the `device='cuda'` line dropped so it runs without a GPU.
        norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1]
                          + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
        q = r / norm[:, None]
        R = torch.zeros((q.size(0), 3, 3), dtype=r.dtype)
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

    # Mix aligned, tail, and tiny shapes.
    for N in [64, 128, 100, 17, 1025]:
        r = torch.randn((N, 4), dtype=torch.float32)
        R = torch.zeros((N, OUT_COLS), dtype=torch.float32)

        R_ref = build_rotation_torch(r).reshape(N, OUT_COLS)
        R_kernel = OpExec(build_rotation_kernel, simulator=True)(r, R, N)

        torch.testing.assert_close(R_kernel, R_ref, rtol=1e-4, atol=1e-4)
        diff = torch.abs(R_kernel - R_ref).max().item()
        print(f"N={N:>5}  max_abs_diff={diff:.3e}")
