"""Bounding rectangle of a projected 2D Gaussian on a5: mirrors
3DGS-opts/pytorch/EWA_fully_fused_proj_packed.py::get_rect.

Contract:
    pix_coord : float32 [N, 2]    per-Gaussian projected pixel center (x, y)
    radii     : float32 [N]       per-Gaussian bounding-circle radius
    width     : int               image width  (used as scalar clip bound W-1)
    height    : int               image height (used as scalar clip bound H-1)
    rect_min  : float32 [N, 2]    clip(pix - r, 0, [W-1, H-1])
    rect_max  : float32 [N, 2]    clip(pix + r, 0, [W-1, H-1])

The clip upper bounds (W-1, H-1) enter `vmins(...)` as Python floats, the same
way `0.1` enters `vmaxs(0.1)` in get_radius.py. Because the @kernel and @vf
decorators AST-recompile their function with `exec(..., func.__globals__, ns)`
and discard closures, the bounds live as module-level globals (_MAX_X, _MAX_Y)
that the public `get_rect` wrapper rebinds before each OpExec call. Each
trace reads the live values, so a single compiled @kernel handles any
(width, height) pair without rebuilding.

Topology: vec-only.
Tail-safety: arbitrary N. Inner dim 2 is fixed at parse time.
Device: a5 (950).
"""

from easyasc.a5 import *


CHUNK = 32       # rows per @vf call; UB tile height
PIX_COLS = 2     # pix_coord columns (x, y)
RAD_COLS = 1     # radii columns after [N] -> [N, 1] unsqueeze
RECT_COLS = 2    # rect_min / rect_max columns
PAD = 8          # float32 C0; one C0 block holds 2 real cols + 6 junk


# Live clip upper bounds rebound by `get_rect` before each OpExec call.
# Read at trace time via vmins(_MAX_X) / vmins(_MAX_Y) inside the @vf body.
_MAX_X = 0.0
_MAX_Y = 0.0


@vf()
def get_rect_vf(pbuf: Tensor, rbuf: Tensor,
                mnbuf: Tensor, mxbuf: Tensor, rows: Var):
    px = Reg(DT.float)
    py = Reg(DT.float)
    r = Reg(DT.float)
    out = Reg(DT.float)

    for i in range(rows):
        px <<= pbuf[i:i + 1, 0:1].single()
        py <<= pbuf[i:i + 1, 1:2].single()
        r <<= rbuf[i:i + 1, 0:1].single()

        # rect_min[..., 0] = clip(px - r, 0, _MAX_X)
        out <<= px - r
        out <<= out.vmaxs(0.0)
        out <<= out.vmins(_MAX_X)
        mnbuf[i:i + 1, 0:1] <<= out.single_value()

        # rect_min[..., 1] = clip(py - r, 0, _MAX_Y)
        out <<= py - r
        out <<= out.vmaxs(0.0)
        out <<= out.vmins(_MAX_Y)
        mnbuf[i:i + 1, 1:2] <<= out.single_value()

        # rect_max[..., 0] = clip(px + r, 0, _MAX_X)
        out <<= px + r
        out <<= out.vmaxs(0.0)
        out <<= out.vmins(_MAX_X)
        mxbuf[i:i + 1, 0:1] <<= out.single_value()

        # rect_max[..., 1] = clip(py + r, 0, _MAX_Y)
        out <<= py + r
        out <<= out.vmaxs(0.0)
        out <<= out.vmins(_MAX_Y)
        mxbuf[i:i + 1, 1:2] <<= out.single_value()


@kernel()
def get_rect_kernel(pix_coord: GMTensor, radii: GMTensor,
                    rect_min: GMTensor, rect_max: GMTensor, N: Var):
    pbuf = DBuff(DT.float, [CHUNK, PAD], Position.UB)
    rbuf = DBuff(DT.float, [CHUNK, PAD], Position.UB)
    mnbuf = DBuff(DT.float, [CHUNK, PAD], Position.UB)
    mxbuf = DBuff(DT.float, [CHUNK, PAD], Position.UB)

    buf_cnt = Var(0)

    total_chunks = CeilDiv(N, CHUNK)
    chunks_per_core = CeilDiv(total_chunks, GetVecNum())
    chunk_begin = Var(chunks_per_core * GetVecIdx())
    chunk_end = Min(chunk_begin + chunks_per_core, total_chunks)

    with auto_sync():
        for chunk_idx in range(chunk_begin, chunk_end):
            row0 = Var(chunk_idx * CHUNK)
            valid_rows = Min(CHUNK, N - row0)

            # GM [valid_rows, 2] -> UB [valid_rows, 8] (2 real + 6 junk).
            pbuf[buf_cnt] <<= pix_coord[row0:row0 + valid_rows, 0:PIX_COLS]
            # GM [valid_rows, 1] -> UB [valid_rows, 8] (1 real + 7 junk).
            rbuf[buf_cnt] <<= radii[row0:row0 + valid_rows, 0:RAD_COLS]

            get_rect_vf(pbuf[buf_cnt], rbuf[buf_cnt],
                        mnbuf[buf_cnt], mxbuf[buf_cnt], valid_rows)

            # UB [valid_rows, 8] -> GM [valid_rows, 2] (drop the 6 junk cols).
            rect_min[row0:row0 + valid_rows, 0:RECT_COLS] <<= \
                mnbuf[buf_cnt][0:valid_rows, 0:RECT_COLS]
            rect_max[row0:row0 + valid_rows, 0:RECT_COLS] <<= \
                mxbuf[buf_cnt][0:valid_rows, 0:RECT_COLS]

            buf_cnt += 1

    return rect_min, rect_max


def get_rect(pix_coord, radii, width, height):
    """Public wrapper: takes float32 [N, 2] pix_coord and float32 [N] radii,
    returns (rect_min, rect_max), each float32 [N, 2].

    Mirrors 3DGS-opts/pytorch/EWA_fully_fused_proj_packed.py::get_rect. Uses a
    shape-only unsqueeze ([N] -> [N, 1]) on radii to bridge to the kernel
    layout. Sets module-level _MAX_X / _MAX_Y to (W-1, H-1) so the next
    @kernel trace reads the right clip bounds inside its @vf calls.
    """
    import torch

    assert pix_coord.dim() == 2 and pix_coord.shape[1] == 2, \
        "expected pix_coord shape [N, 2]"
    assert pix_coord.dtype == torch.float32, "expected float32 pix_coord"
    assert radii.dim() == 1 and radii.shape[0] == pix_coord.shape[0], \
        "expected radii shape [N] matching pix_coord[0]"
    assert radii.dtype == torch.float32, "expected float32 radii"

    global _MAX_X, _MAX_Y
    _MAX_X = float(width - 1.0)
    _MAX_Y = float(height - 1.0)

    N = pix_coord.shape[0]
    pix_2d = pix_coord.contiguous()
    rad_2d = radii.unsqueeze(-1).contiguous()
    rect_min = torch.zeros((N, 2), dtype=torch.float32)
    rect_max = torch.zeros((N, 2), dtype=torch.float32)

    out_min, out_max = OpExec(get_rect_kernel, simulator=True)(
        pix_2d, rad_2d, rect_min, rect_max, N,
    )
    return out_min, out_max


if __name__ == "__main__":
    import torch

    def get_rect_torch(pix_coord, radii, width, height):
        # Mirrors 3DGS-opts/pytorch/EWA_fully_fused_proj_packed.py::get_rect.
        rect_min = (pix_coord - radii[:, None]).clone()
        rect_max = (pix_coord + radii[:, None]).clone()
        rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
        rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
        rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
        rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
        return rect_min, rect_max

    torch.manual_seed(0)

    width, height = 640, 480

    # Test budget: N <= 60, at most 2 cases. User runs larger N themselves.
    # N=17 stays inside one chunk; N=60 spans two chunks (32 + 28 tail).
    for N in [17, 60]:
        # pix_coord spread over [-50, 690] x [-50, 530] hits both clip bounds.
        pix_coord = torch.empty((N, 2), dtype=torch.float32)
        pix_coord[:, 0] = torch.rand(N) * 740.0 - 50.0
        pix_coord[:, 1] = torch.rand(N) * 580.0 - 50.0
        radii = torch.rand(N, dtype=torch.float32) * 100.0

        ref_min, ref_max = get_rect_torch(pix_coord, radii, width, height)
        out_min, out_max = get_rect(pix_coord, radii, width, height)

        torch.testing.assert_close(out_min, ref_min, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out_max, ref_max, rtol=1e-4, atol=1e-4)

        d_min = torch.abs(out_min - ref_min).max().item()
        d_max = torch.abs(out_max - ref_max).max().item()
        print(f"N={N:>3}  max_abs_diff: rect_min={d_min:.3e}  rect_max={d_max:.3e}")
