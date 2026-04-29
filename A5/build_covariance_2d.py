"""2D screen-space covariance on a5: mirrors
3DGS-opts/pytorch/EWA_fully_fused_proj_packed.py::build_covariance_2d.

Per Gaussian, the kernel computes
    cov_c   = R @ cov3d @ R.T                   (R is viewmatrix[:3, :3])
    J       = [[fx*rz, 0, -fx*tx*rz^2],
               [0, fy*rz, -fy*ty*rz^2]]         (fx, fy from K)
    cov2d   = J @ cov_c @ J.T
    cov2d_blur = cov2d + 0.3 * I
    det_blur   = det(cov2d_blur)
    compensation = sqrt(max(det(cov2d) / det_blur, 0))

Contract:
    mean_c       : float32 [N, 3]   per-row camera-space (x, y, z)
    cov3d        : float32 [N, 9]   per-row 3x3 row-major flattened
    vm_R         : float32 [1, 9]   viewmatrix[:3, :3] flattened, shared across rows
    params       : float32 [1, 8]   [fx, fy, lim_x_pos, lim_x_neg, lim_y_pos, lim_y_neg, 0, 0]
    cov2d_blur   : float32 [N, 4]   row-major 2x2 with +0.3 already on the diagonal
    det_blur     : float32 [N, 1]
    compensation : float32 [N, 1]

Topology: vec-only (3x3 / 2x3 / 2x2 matmuls fold into scalar mul-adds).
Tail-safety: arbitrary N. Inner widths are fixed at parse time.
Device: a5 (950).

mean3d is intentionally absent: the PyTorch reference only used it to read
`torch.finfo(mean3d.dtype).eps`. The wrapper validates dtype=float32 and bakes
eps_z = 1.19209e-7 as a literal. eps2d is also baked as the only call-site
default 0.3.

Style mirrors 3DGS-opts/A5/build_covariance_3d.py for chunking and scalar-loop
structure. Per-row work: load constants once at vf-top (9 R + 6 params), then
per row compute J's 4 nonzero entries and accumulate the 6 unique cov_c entries
the same way build_covariance_3d.py does.
"""

from easyasc.a5 import *


CHUNK = 32          # rows per @vf call; UB tile height
MEAN_COLS = 3       # mean_c per-row width
COV3_COLS = 9       # cov3d per-row width (3x3 flat)
R_COLS = 9          # vm_R width (3x3 flat)
P_COLS = 6          # 6 real param slots; padded to one C0 block
COV2_COLS = 4       # cov2d_blur per-row width (2x2 flat)
ONE_COLS = 1        # det_blur, compensation per-row width

# UB column padding. float32 C0 = 8 elements -> row width must be a multiple of 8.
MEAN_PAD = 8        # 3 + 5 junk
COV3_PAD = 16       # 9 + 7 junk
R_PAD = 16          # 9 + 7 junk
P_PAD = 8           # 6 + 2 junk
COV2_PAD = 8        # 4 + 4 junk
ONE_PAD = 8         # 1 + 7 junk

EPS_Z = 1.19209e-7  # finfo(float32).eps -- guards 1/z when z is near 0
EPS2D = 0.3         # diagonal blur term, baked from the only call-site default


@vf()
def build_covariance_2d_vf(mc_buf: Tensor, c3_buf: Tensor,
                            R_buf: Tensor, P_buf: Tensor,
                            cb_buf: Tensor, db_buf: Tensor, cp_buf: Tensor,
                            rows: Var):
    # -------- per-call constants: 9 viewmatrix entries + 6 derived params --------
    R00 = Reg(DT.float); R01 = Reg(DT.float); R02 = Reg(DT.float)
    R10 = Reg(DT.float); R11 = Reg(DT.float); R12 = Reg(DT.float)
    R20 = Reg(DT.float); R21 = Reg(DT.float); R22 = Reg(DT.float)

    fx_r = Reg(DT.float); fy_r = Reg(DT.float)
    lim_xp = Reg(DT.float); lim_xn = Reg(DT.float)
    lim_yp = Reg(DT.float); lim_yn = Reg(DT.float)

    R00 <<= R_buf[0:1, 0:1].single()
    R01 <<= R_buf[0:1, 1:2].single()
    R02 <<= R_buf[0:1, 2:3].single()
    R10 <<= R_buf[0:1, 3:4].single()
    R11 <<= R_buf[0:1, 4:5].single()
    R12 <<= R_buf[0:1, 5:6].single()
    R20 <<= R_buf[0:1, 6:7].single()
    R21 <<= R_buf[0:1, 7:8].single()
    R22 <<= R_buf[0:1, 8:9].single()

    fx_r   <<= P_buf[0:1, 0:1].single()
    fy_r   <<= P_buf[0:1, 1:2].single()
    lim_xp <<= P_buf[0:1, 2:3].single()
    lim_xn <<= P_buf[0:1, 3:4].single()
    lim_yp <<= P_buf[0:1, 4:5].single()
    lim_yn <<= P_buf[0:1, 5:6].single()

    one_r = Reg(DT.float); one_r.fill(1.0)
    eps_z = Reg(DT.float); eps_z.fill(EPS_Z)

    # max(xrz, -lim_xn) needs a Reg holding -lim_xn; both lim_xn and lim_yn are
    # constant for the whole call, so negate once outside the per-row loop.
    neg_lim_xn = Reg(DT.float); neg_lim_xn <<= lim_xn * -1.0
    neg_lim_yn = Reg(DT.float); neg_lim_yn <<= lim_yn * -1.0

    # -------- per-row scratch --------
    x = Reg(DT.float); y = Reg(DT.float); z = Reg(DT.float)
    z_safe = Reg(DT.float); rz = Reg(DT.float); rz2 = Reg(DT.float)
    xrz = Reg(DT.float); yrz = Reg(DT.float)
    tx = Reg(DT.float); ty = Reg(DT.float); tmpr = Reg(DT.float)

    J00 = Reg(DT.float); J02 = Reg(DT.float)
    J11 = Reg(DT.float); J12 = Reg(DT.float)

    c00 = Reg(DT.float); c01 = Reg(DT.float); c02 = Reg(DT.float)
    c10 = Reg(DT.float); c11 = Reg(DT.float); c12 = Reg(DT.float)
    c20 = Reg(DT.float); c21 = Reg(DT.float); c22 = Reg(DT.float)

    cc00 = Reg(DT.float); cc01 = Reg(DT.float); cc02 = Reg(DT.float)
    cc11 = Reg(DT.float); cc12 = Reg(DT.float); cc22 = Reg(DT.float)

    A00 = Reg(DT.float); A01 = Reg(DT.float); A02 = Reg(DT.float)
    A10 = Reg(DT.float); A11 = Reg(DT.float); A12 = Reg(DT.float)

    B00 = Reg(DT.float); B01 = Reg(DT.float); B11 = Reg(DT.float)

    t = Reg(DT.float); s = Reg(DT.float); acc = Reg(DT.float)
    det_orig = Reg(DT.float); det_blur_v = Reg(DT.float)
    c2d00 = Reg(DT.float); c2d11 = Reg(DT.float)
    ratio = Reg(DT.float); comp = Reg(DT.float)

    for i in range(rows):
        # ---- load mean_c (x, y, z) ----
        x <<= mc_buf[i:i + 1, 0:1].single()
        y <<= mc_buf[i:i + 1, 1:2].single()
        z <<= mc_buf[i:i + 1, 2:3].single()

        # ---- load 9 cov3d entries (row-major) ----
        c00 <<= c3_buf[i:i + 1, 0:1].single()
        c01 <<= c3_buf[i:i + 1, 1:2].single()
        c02 <<= c3_buf[i:i + 1, 2:3].single()
        c10 <<= c3_buf[i:i + 1, 3:4].single()
        c11 <<= c3_buf[i:i + 1, 4:5].single()
        c12 <<= c3_buf[i:i + 1, 5:6].single()
        c20 <<= c3_buf[i:i + 1, 6:7].single()
        c21 <<= c3_buf[i:i + 1, 7:8].single()
        c22 <<= c3_buf[i:i + 1, 8:9].single()

        # ---- z guard + reciprocals ----
        z_safe <<= z.vmax(eps_z)
        rz <<= one_r / z_safe
        rz2 <<= rz * rz

        # ---- truncated tx, ty (frustum guard) ----
        # tx = z * min(lim_xp, max(-lim_xn, x*rz))
        xrz <<= x * rz
        tmpr <<= xrz.vmax(neg_lim_xn)
        tmpr <<= tmpr.vmin(lim_xp)
        tx <<= z * tmpr

        yrz <<= y * rz
        tmpr <<= yrz.vmax(neg_lim_yn)
        tmpr <<= tmpr.vmin(lim_yp)
        ty <<= z * tmpr

        # ---- J's 4 nonzero entries ----
        J00 <<= fx_r * rz
        t   <<= fx_r * tx
        t   <<= t * rz2
        J02 <<= t * -1.0

        J11 <<= fy_r * rz
        t   <<= fy_r * ty
        t   <<= t * rz2
        J12 <<= t * -1.0

        # ---- cov_c = R @ cov3d @ R.T (nested matmul, mirrors PyTorch op order) ----
        # M[u,k] = sum_i R[u,i] * cov3d[i,k]    (3-term sums, 9 entries)
        # cc[u,v] = sum_k M[u,k] * R[v,k]       (3-term sums, 6 unique entries; cc is symmetric)
        # Mirroring PyTorch's `(R @ cov3d) @ R.T` keeps the rounding error to the
        # same chained-matmul depth instead of collapsing to a flat 9-term sum.

        m00 = Reg(DT.float); m01 = Reg(DT.float); m02 = Reg(DT.float)
        m10 = Reg(DT.float); m11 = Reg(DT.float); m12 = Reg(DT.float)
        m20 = Reg(DT.float); m21 = Reg(DT.float); m22 = Reg(DT.float)

        # M = R @ cov3d
        m00 <<= R00 * c00; t <<= R01 * c10; m00 <<= m00 + t; t <<= R02 * c20; m00 <<= m00 + t
        m01 <<= R00 * c01; t <<= R01 * c11; m01 <<= m01 + t; t <<= R02 * c21; m01 <<= m01 + t
        m02 <<= R00 * c02; t <<= R01 * c12; m02 <<= m02 + t; t <<= R02 * c22; m02 <<= m02 + t
        m10 <<= R10 * c00; t <<= R11 * c10; m10 <<= m10 + t; t <<= R12 * c20; m10 <<= m10 + t
        m11 <<= R10 * c01; t <<= R11 * c11; m11 <<= m11 + t; t <<= R12 * c21; m11 <<= m11 + t
        m12 <<= R10 * c02; t <<= R11 * c12; m12 <<= m12 + t; t <<= R12 * c22; m12 <<= m12 + t
        m20 <<= R20 * c00; t <<= R21 * c10; m20 <<= m20 + t; t <<= R22 * c20; m20 <<= m20 + t
        m21 <<= R20 * c01; t <<= R21 * c11; m21 <<= m21 + t; t <<= R22 * c21; m21 <<= m21 + t
        m22 <<= R20 * c02; t <<= R21 * c12; m22 <<= m22 + t; t <<= R22 * c22; m22 <<= m22 + t

        # cc = M @ R.T  (compute the 6 unique entries; cc[1,0]=cc01, cc[2,0]=cc02, cc[2,1]=cc12)
        cc00 <<= m00 * R00; t <<= m01 * R01; cc00 <<= cc00 + t; t <<= m02 * R02; cc00 <<= cc00 + t
        cc01 <<= m00 * R10; t <<= m01 * R11; cc01 <<= cc01 + t; t <<= m02 * R12; cc01 <<= cc01 + t
        cc02 <<= m00 * R20; t <<= m01 * R21; cc02 <<= cc02 + t; t <<= m02 * R22; cc02 <<= cc02 + t
        cc11 <<= m10 * R10; t <<= m11 * R11; cc11 <<= cc11 + t; t <<= m12 * R12; cc11 <<= cc11 + t
        cc12 <<= m10 * R20; t <<= m11 * R21; cc12 <<= cc12 + t; t <<= m12 * R22; cc12 <<= cc12 + t
        cc22 <<= m20 * R20; t <<= m21 * R21; cc22 <<= cc22 + t; t <<= m22 * R22; cc22 <<= cc22 + t

        # ---- A = J @ cov_c (sparse J: only J00, J02 in row 0; J11, J12 in row 1) ----
        # cc is symmetric -> cc[1,0]=cc01, cc[2,0]=cc02, cc[2,1]=cc12.
        t   <<= J00 * cc00; s   <<= J02 * cc02; A00 <<= t + s
        t   <<= J00 * cc01; s   <<= J02 * cc12; A01 <<= t + s
        t   <<= J00 * cc02; s   <<= J02 * cc22; A02 <<= t + s
        t   <<= J11 * cc01; s   <<= J12 * cc02; A10 <<= t + s
        t   <<= J11 * cc11; s   <<= J12 * cc12; A11 <<= t + s
        t   <<= J11 * cc12; s   <<= J12 * cc22; A12 <<= t + s

        # ---- B = A @ J.T (B is symmetric, so B[0,1] == B[1,0]) ----
        t   <<= A00 * J00; s   <<= A02 * J02; B00 <<= t + s
        t   <<= A01 * J11; s   <<= A02 * J12; B01 <<= t + s
        t   <<= A11 * J11; s   <<= A12 * J12; B11 <<= t + s

        # ---- determinants and compensation ----
        # det_orig = B00*B11 - B01*B01     (cov2d before adding the blur)
        # det_blur = (B00+0.3)*(B11+0.3) - B01*B01
        det_orig <<= B00 * B11
        t        <<= B01 * B01
        det_orig <<= det_orig - t

        c2d00 <<= B00 + 0.3
        c2d11 <<= B11 + 0.3
        det_blur_v <<= c2d00 * c2d11
        det_blur_v <<= det_blur_v - t   # t still holds B01 * B01

        ratio <<= det_orig / det_blur_v
        ratio <<= ratio.vmaxs(0.0)
        comp  <<= ratio.sqrt()

        # ---- stores ----
        cb_buf[i:i + 1, 0:1] <<= c2d00.single_value()
        cb_buf[i:i + 1, 1:2] <<= B01.single_value()
        cb_buf[i:i + 1, 2:3] <<= B01.single_value()  # mirror -> cov2d_blur[1, 0]
        cb_buf[i:i + 1, 3:4] <<= c2d11.single_value()
        db_buf[i:i + 1, 0:1] <<= det_blur_v.single_value()
        cp_buf[i:i + 1, 0:1] <<= comp.single_value()


@kernel()
def build_covariance_2d_kernel(mean_c: GMTensor, cov3d: GMTensor,
                                vm_R: GMTensor, params: GMTensor,
                                cov2d_blur: GMTensor, det_blur: GMTensor,
                                compensation: GMTensor, N: Var):
    mc_ub = DBuff(DT.float, [CHUNK, MEAN_PAD], Position.UB)
    c3_ub = DBuff(DT.float, [CHUNK, COV3_PAD], Position.UB)
    R_ub  = DBuff(DT.float, [1,     R_PAD],    Position.UB)
    P_ub  = DBuff(DT.float, [1,     P_PAD],    Position.UB)
    cb_ub = DBuff(DT.float, [CHUNK, COV2_PAD], Position.UB)
    db_ub = DBuff(DT.float, [CHUNK, ONE_PAD],  Position.UB)
    cp_ub = DBuff(DT.float, [CHUNK, ONE_PAD],  Position.UB)

    buf_cnt = Var(0)

    total_chunks = CeilDiv(N, CHUNK)
    chunks_per_core = CeilDiv(total_chunks, GetVecNum())
    chunk_begin = Var(chunks_per_core * GetVecIdx())
    chunk_end = Min(chunk_begin + chunks_per_core, total_chunks)

    with auto_sync():
        for chunk_idx in range(chunk_begin, chunk_end):
            row0 = Var(chunk_idx * CHUNK)
            valid_rows = Min(CHUNK, N - row0)

            mc_ub[buf_cnt] <<= mean_c[row0:row0 + valid_rows, 0:MEAN_COLS]
            c3_ub[buf_cnt] <<= cov3d[row0:row0 + valid_rows, 0:COV3_COLS]
            # Per-call constants: tiny (24 floats); reload per-chunk against
            # buf_cnt so the auto_sync write/read events stay balanced. The DMA
            # cost (one C0 block each) is negligible vs the per-row math.
            R_ub[buf_cnt] <<= vm_R[0:1, 0:R_COLS]
            P_ub[buf_cnt] <<= params[0:1, 0:P_COLS]

            build_covariance_2d_vf(
                mc_ub[buf_cnt], c3_ub[buf_cnt],
                R_ub[buf_cnt], P_ub[buf_cnt],
                cb_ub[buf_cnt], db_ub[buf_cnt], cp_ub[buf_cnt],
                valid_rows,
            )

            cov2d_blur  [row0:row0 + valid_rows, 0:COV2_COLS] <<= cb_ub[buf_cnt][0:valid_rows, 0:COV2_COLS]
            det_blur    [row0:row0 + valid_rows, 0:ONE_COLS]  <<= db_ub[buf_cnt][0:valid_rows, 0:ONE_COLS]
            compensation[row0:row0 + valid_rows, 0:ONE_COLS]  <<= cp_ub[buf_cnt][0:valid_rows, 0:ONE_COLS]

            buf_cnt += 1

    return cov2d_blur, det_blur, compensation


def build_covariance_2d(mean3d, cov3d, mean_c, viewmatrix, K, width, height, eps2d=0.3):
    """Public wrapper: matches the EWA_fully_fused_proj_packed.build_covariance_2d
    signature. mean3d is accepted for signature compatibility but unused — the
    reference only used it to read `torch.finfo(dtype).eps`, which we bake.

    Returns (cov2d_blur [N, 2, 2], det_blur [N], compensation [N]).
    """
    import torch

    assert eps2d == 0.3, "kernel bakes eps2d=0.3"
    assert mean_c.dtype == torch.float32, "mean_c must be float32"
    assert mean_c.dim() == 2 and mean_c.shape[1] == 3, "mean_c must be [N, 3]"
    assert cov3d.dtype == torch.float32, "cov3d must be float32"
    assert viewmatrix.dtype == torch.float32 and viewmatrix.shape[-2:] == (4, 4)
    assert K.dtype == torch.float32 and K.shape[-2:] == (3, 3)

    N = mean_c.shape[0]
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])

    tan_fovx = 0.5 * float(width) / fx
    tan_fovy = 0.5 * float(height) / fy
    lim_x_pos = (float(width)  - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx                 / fx + 0.3 * tan_fovx
    lim_y_pos = (float(height) - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy                 / fy + 0.3 * tan_fovy

    vm_R = viewmatrix[:3, :3].reshape(1, R_COLS).contiguous()
    params = torch.zeros((1, P_PAD), dtype=torch.float32)
    params[0, 0] = fx; params[0, 1] = fy
    params[0, 2] = lim_x_pos; params[0, 3] = lim_x_neg
    params[0, 4] = lim_y_pos; params[0, 5] = lim_y_neg

    mc_2d = mean_c.contiguous()
    c3_2d = cov3d.reshape(N, COV3_COLS).contiguous()

    cov2d_flat = torch.zeros((N, COV2_COLS), dtype=torch.float32)
    det_blur_2d = torch.zeros((N, ONE_COLS), dtype=torch.float32)
    compensation_2d = torch.zeros((N, ONE_COLS), dtype=torch.float32)

    cov_out, det_out, comp_out = OpExec(build_covariance_2d_kernel, simulator=True)(
        mc_2d, c3_2d, vm_R, params,
        cov2d_flat, det_blur_2d, compensation_2d, N,
    )
    return cov_out.reshape(N, 2, 2), det_out.squeeze(-1), comp_out.squeeze(-1)


if __name__ == "__main__":
    import torch
    import math

    def build_covariance_2d_torch(mean_c, cov3d, viewmatrix, K, width, height, eps2d=0.3):
        # Inlined CPU reference; mirrors
        #   3DGS-opts/pytorch/EWA_fully_fused_proj_packed.py::build_covariance_2d
        # with `device='cuda'` lines dropped and the unused mean3d argument
        # removed (only used there to read finfo(dtype).eps).
        R = viewmatrix[:3, :3]
        z = mean_c[:, 2]
        x = mean_c[:, 0]; y = mean_c[:, 1]

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        eps = torch.finfo(mean_c.dtype).eps
        rz = 1.0 / torch.clamp(z, min=eps)

        tan_fovx = 0.5 * width / fx
        tan_fovy = 0.5 * height / fy
        lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
        lim_x_neg = cx / fx + 0.3 * tan_fovx
        lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
        lim_y_neg = cy / fy + 0.3 * tan_fovy

        lim_x_pos_t = torch.as_tensor(lim_x_pos, dtype=z.dtype)
        lim_x_neg_t = torch.as_tensor(lim_x_neg, dtype=z.dtype)
        lim_y_pos_t = torch.as_tensor(lim_y_pos, dtype=z.dtype)
        lim_y_neg_t = torch.as_tensor(lim_y_neg, dtype=z.dtype)

        xrz = x * rz; yrz = y * rz
        tx = z * torch.minimum(lim_x_pos_t, torch.maximum(-lim_x_neg_t, xrz))
        ty = z * torch.minimum(lim_y_pos_t, torch.maximum(-lim_y_neg_t, yrz))
        rz2 = rz * rz

        N = mean_c.shape[0]
        J = torch.zeros((N, 2, 3), dtype=mean_c.dtype)
        J[:, 0, 0] = fx * rz
        J[:, 0, 2] = -fx * tx * rz2
        J[:, 1, 1] = fy * rz
        J[:, 1, 2] = -fy * ty * rz2

        cov_c = R @ cov3d @ R.T
        cov2d = J @ cov_c @ J.transpose(1, 2)

        filt = torch.eye(2, 2, dtype=cov2d.dtype) * eps2d
        det_orig = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] * cov2d[..., 1, 0]
        cov2d_blur = cov2d + filt[None]
        det_blur = cov2d_blur[..., 0, 0] * cov2d_blur[..., 1, 1] \
                   - cov2d_blur[..., 0, 1] * cov2d_blur[..., 1, 0]
        compensation = torch.sqrt(torch.clamp(det_orig / det_blur, min=0))

        return cov2d_blur, det_blur, compensation

    def random_rotation_matrix(seed_offset: int = 0) -> torch.Tensor:
        # Build an SO(3) matrix from a random unit quaternion, no GPU dependency.
        torch.manual_seed(123 + seed_offset)
        q = torch.randn(4, dtype=torch.float32)
        q = q / q.norm()
        w, xq, yq, zq = q[0], q[1], q[2], q[3]
        R = torch.tensor([
            [1 - 2 * (yq * yq + zq * zq), 2 * (xq * yq - w * zq), 2 * (xq * zq + w * yq)],
            [2 * (xq * yq + w * zq), 1 - 2 * (xq * xq + zq * zq), 2 * (yq * zq - w * xq)],
            [2 * (xq * zq - w * yq), 2 * (yq * zq + w * xq), 1 - 2 * (xq * xq + yq * yq)],
        ], dtype=torch.float32)
        return R

    width, height = 640, 480
    fx_v = fy_v = 300.0
    cx_v, cy_v = 320.0, 240.0
    K = torch.tensor([[fx_v, 0.0, cx_v],
                      [0.0, fy_v, cy_v],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

    # Test budget: N <= 60, at most 2 cases. User runs larger N.
    # N = 17 -> tail-only path (one partial chunk).
    # N = 60 -> one full CHUNK=32 + 28-row tail.
    for trial, N in enumerate([17, 60]):
        torch.manual_seed(2024 + trial)

        Rmat = random_rotation_matrix(seed_offset=trial)
        viewmatrix = torch.eye(4, dtype=torch.float32)
        viewmatrix[:3, :3] = Rmat
        viewmatrix[:3, 3] = torch.tensor([0.1, -0.2, 0.5])  # arbitrary translation

        # Camera-space coords: z in [1.0, 5.0] (positive, well above eps);
        # x,y spread wide enough to exercise both the +lim and -lim clamp branches.
        mean_c = torch.empty((N, 3), dtype=torch.float32)
        mean_c[:, 0] = (torch.rand(N) * 6.0 - 3.0)  # [-3, 3]
        mean_c[:, 1] = (torch.rand(N) * 6.0 - 3.0)
        mean_c[:, 2] = (torch.rand(N) * 4.0 + 1.0)  # [1, 5]

        # PSD cov3d via L @ L.T with a small ridge for stability.
        L = torch.randn(N, 3, 3, dtype=torch.float32) * 0.1
        cov3d_full = L @ L.transpose(1, 2) + 0.01 * torch.eye(3, dtype=torch.float32)[None]

        ref_cov, ref_det, ref_comp = build_covariance_2d_torch(
            mean_c, cov3d_full, viewmatrix, K, width, height, eps2d=0.3,
        )

        out_cov, out_det, out_comp = build_covariance_2d(
            mean3d=None,         # unused, kept only for wrapper signature parity
            cov3d=cov3d_full,
            mean_c=mean_c,
            viewmatrix=viewmatrix,
            K=K,
            width=width,
            height=height,
            eps2d=0.3,
        )

        torch.testing.assert_close(out_cov, ref_cov, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out_det, ref_det, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out_comp, ref_comp, rtol=1e-4, atol=1e-4)

        def _stats(out, ref):
            abs_diff = (out - ref).abs()
            denom = ref.abs().clamp(min=1e-12)
            rel_diff = abs_diff / denom
            return abs_diff.max().item(), rel_diff.max().item(), ref.abs().max().item()

        c_abs, c_rel, c_mag = _stats(out_cov,  ref_cov)
        d_abs, d_rel, d_mag = _stats(out_det,  ref_det)
        p_abs, p_rel, p_mag = _stats(out_comp, ref_comp)

        # Float32 has ~7 decimal digits of mantissa precision (eps = 1.19e-7).
        # A clean 3-term inner sum hits ~3*eps ≈ 4e-7 relative; chained 6-term
        # matmul depth ~6*eps ≈ 7e-7. Anything in the 1e-6..1e-5 range is normal
        # for these compositions; > 1e-4 would be a real concern.
        print(f"N={N:>3}")
        print(f"   cov2d_blur   max|ref|={c_mag:.3e}  max_abs_diff={c_abs:.3e}  max_rel_diff={c_rel:.3e}")
        print(f"   det_blur     max|ref|={d_mag:.3e}  max_abs_diff={d_abs:.3e}  max_rel_diff={d_rel:.3e}")
        print(f"   compensation max|ref|={p_mag:.3e}  max_abs_diff={p_abs:.3e}  max_rel_diff={p_rel:.3e}")
