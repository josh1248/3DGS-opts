"""
Simple OpenCV viewer for a 3D Gaussian Splatting .ply scene.

Usage:
    python view_scene.py --ply /path/to/point_cloud.ply

Controls (OpenCV window must be focused):
    W / S  : move camera forward / backward
    A / D  : strafe left / right
    Q / E  : move down / up
    J / L  : yaw left / right
    I / K  : pitch up / down
    R      : reset camera
    ESC    : quit

Expects the standard Inria 3DGS PLY schema:
    x, y, z,
    f_dc_0, f_dc_1, f_dc_2,
    f_rest_0 ... f_rest_{3*((sh_degree+1)**2 - 1) - 1},
    opacity,
    scale_0, scale_1, scale_2,
    rot_0, rot_1, rot_2, rot_3    (w, x, y, z quaternion)

Stored scales are log-scales; stored opacities are pre-sigmoid logits;
quaternions are un-normalized. All of that is handled on load.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import path: rendering.py imports from a package called `torch_splat`, but
# the four files live next to this script. Expose them as that package name.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import types

_pkg = types.ModuleType("torch_splat")
_pkg.__path__ = [HERE]
sys.modules.setdefault("torch_splat", _pkg)
for _mod in ("rasterization_utils", "EWA_fully_fused_proj_packed", "sh_utils"):
    try:
        __import__(_mod)
        sys.modules[f"torch_splat.{_mod}"] = sys.modules[_mod]
    except Exception as e:
        print(f"[warn] failed to alias torch_splat.{_mod}: {e}")

try:
    from rendering import torch_rasterization
    HAS_PIPELINE = True
except Exception as e:
    print(f"[warn] could not import torch_rasterization: {e}")
    print("[warn] falling back to a minimal self-contained rasterizer.")
    HAS_PIPELINE = False

# ---------------------------------------------------------------------------
# PLY loading
# ---------------------------------------------------------------------------
def load_3dgs_ply(path: str, device: torch.device) -> dict:
    from plyfile import PlyData
    ply = PlyData.read(path)
    v = ply["vertex"].data

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    # quaternions (w, x, y, z) — normalize
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)
    rot /= (np.linalg.norm(rot, axis=-1, keepdims=True) + 1e-8)

    # scales: stored as log(scale). take exp() to get actual scales
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32))

    # opacity: stored pre-sigmoid
    opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))

    # SH coefficients
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)  # [N,3]

    rest_names = sorted(
        [n for n in v.dtype.names if n.startswith("f_rest_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    if rest_names:
        f_rest = np.stack([v[n] for n in rest_names], axis=-1).astype(np.float32)  # [N, 3*(K-1)]
        K_minus_1 = f_rest.shape[1] // 3
        # Inria format stores as [3, K-1]: channels-major, flatten. Reshape accordingly.
        f_rest = f_rest.reshape(-1, 3, K_minus_1).transpose(0, 2, 1)  # [N, K-1, 3]
        sh = np.concatenate([f_dc[:, None, :], f_rest], axis=1)  # [N, K, 3]
    else:
        sh = f_dc[:, None, :]  # [N, 1, 3]

    K = sh.shape[1]
    sh_degree = int(round(math.sqrt(K))) - 1
    assert (sh_degree + 1) ** 2 == K, f"odd SH band count K={K}"

    print(f"[info] loaded {xyz.shape[0]} gaussians, sh_degree={sh_degree}")
    return {
        "means":     torch.from_numpy(xyz).to(device),
        "quats":     torch.from_numpy(rot).to(device),
        "scales":    torch.from_numpy(scales).to(device),
        "opacities": torch.from_numpy(opacity).to(device),
        "sh":        torch.from_numpy(sh).to(device),
        "sh_degree": sh_degree,
    }

# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------
def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return a 4x4 world-to-camera matrix (OpenCV convention: +Z forward)."""
    f = target - eye
    f /= np.linalg.norm(f) + 1e-8
    r = np.cross(f, up)
    r /= np.linalg.norm(r) + 1e-8
    u = np.cross(r, f)
    R = np.stack([r, -u, f], axis=0)  # rows = camera axes in world
    t = -R @ eye
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


class OrbitCam:
    """Basic fly-camera in world space."""
    def __init__(self, center: np.ndarray, radius: float):
        self.center = center.astype(np.float32)
        self.radius = float(radius)
        self.yaw = 0.0
        self.pitch = 0.0
        self.eye = self.center + np.array([0, 0, -self.radius], dtype=np.float32)

    def view(self) -> np.ndarray:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        fwd = np.array([sy * cp, -sp, cy * cp], dtype=np.float32)
        return look_at(self.eye, self.eye + fwd, np.array([0, 1, 0], dtype=np.float32))

    def translate_local(self, dx: float, dy: float, dz: float):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        right = np.array([cy, 0, -sy], dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)
        fwd = np.array([sy, 0, cy], dtype=np.float32)
        self.eye += right * dx + up * dy + fwd * dz

# ---------------------------------------------------------------------------
# Fallback rasterizer (tiny, slow, no SH degree > 0) — just to confirm pixels.
# ---------------------------------------------------------------------------
@torch.no_grad()
def fallback_render(splats: dict, w2c: torch.Tensor, K: torch.Tensor,
                    W: int, H: int) -> torch.Tensor:
    """Project centers only, splat a disk of color per gaussian. No EWA."""
    means = splats["means"]
    opac = splats["opacities"]
    # DC term only → RGB
    C0 = 0.28209479177387814
    rgb = (splats["sh"][:, 0, :] * C0 + 0.5).clamp(0, 1)

    ones = torch.ones(means.shape[0], 1, device=means.device)
    cam = (torch.cat([means, ones], dim=-1) @ w2c.T)[..., :3]
    front = cam[:, 2] > 0.01
    cam = cam[front]; rgb = rgb[front]; opac = opac[front]
    uv = (cam @ K.T)
    uv = uv[:, :2] / uv[:, 2:3]

    order = torch.argsort(cam[:, 2], descending=True)  # back-to-front
    uv = uv[order]; rgb = rgb[order]; opac = opac[order]

    img = torch.ones(H, W, 3, device=means.device)
    u = uv[:, 0].long(); v = uv[:, 1].long()
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid]; v = v[valid]; c = rgb[valid]; a = opac[valid, None]
    img[v, u] = img[v, u] * (1 - a) + c * a
    return img

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="path to a 3DGS point_cloud.ply")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0, help="vertical FoV in deg")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default=None, help="optional: save a single frame to this path and exit")
    args = ap.parse_args()

    import cv2
    device = torch.device(args.device)
    splats = load_3dgs_ply(args.ply, device)

    # Initial camera: orbit around the scene centroid
    centroid = splats["means"].mean(dim=0).cpu().numpy()
    extent = float((splats["means"].amax(0) - splats["means"].amin(0)).norm().item())
    cam = OrbitCam(centroid, radius=0.7 * extent)

    # Intrinsics (OpenCV-style)
    fy = 0.5 * args.height / math.tan(math.radians(args.fov) / 2)
    fx = fy
    K = torch.tensor([[fx, 0, args.width / 2],
                      [0, fy, args.height / 2],
                      [0, 0, 1]], dtype=torch.float32, device=device)

    def render_once() -> np.ndarray:
        w2c = torch.from_numpy(cam.view()).to(device)

        if HAS_PIPELINE:
            try:
                colors, _alphas, _meta = torch_rasterization(
                    means=splats["means"],
                    quats=splats["quats"],
                    scales=splats["scales"],
                    opacities=splats["opacities"],
                    colors=splats["sh"],
                    viewmats=w2c[None],                  # [C=1, 4, 4]
                    Ks=K[None],                          # [C=1, 3, 3]
                    width=args.width,
                    height=args.height,
                    sh_degree=splats["sh_degree"],
                    packed=True,
                    render_mode="RGB",
                )
                img = colors[0].clamp(0, 1).cpu().numpy()  # [H,W,3]
                return img
            except Exception as e:
                print(f"[warn] main pipeline failed ({e}); using fallback")

        return fallback_render(splats, w2c, K, args.width, args.height).cpu().numpy()

    if args.save is not None:
        img = render_once()
        bgr = (img[..., ::-1] * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(args.save, bgr)
        print(f"[info] wrote {args.save}")
        return

    cv2.namedWindow("3DGS", cv2.WINDOW_NORMAL)
    step = 0.05 * extent
    ang = math.radians(3.0)

    while True:
        img = render_once()
        bgr = (img[..., ::-1] * 255).clip(0, 255).astype(np.uint8)
        cv2.imshow("3DGS", bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:                       # ESC
            break
        elif key == ord('w'): cam.translate_local(0, 0, +step)
        elif key == ord('s'): cam.translate_local(0, 0, -step)
        elif key == ord('a'): cam.translate_local(-step, 0, 0)
        elif key == ord('d'): cam.translate_local(+step, 0, 0)
        elif key == ord('q'): cam.translate_local(0, -step, 0)
        elif key == ord('e'): cam.translate_local(0, +step, 0)
        elif key == ord('j'): cam.yaw -= ang
        elif key == ord('l'): cam.yaw += ang
        elif key == ord('i'): cam.pitch -= ang
        elif key == ord('k'): cam.pitch += ang
        elif key == ord('r'):
            cam = OrbitCam(centroid, radius=0.7 * extent)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
