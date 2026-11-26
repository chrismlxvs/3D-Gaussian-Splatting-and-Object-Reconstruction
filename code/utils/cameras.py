"""Camera JSON parsing and explicit OpenCV/renderer pose conversions.

All transforms use column vectors. The supplied JSON convention is
world-to-camera with OpenCV axes (right, down, forward). Common NeRF/3DGS
renderers use camera-to-world with OpenGL axes (right, up, backward).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def scaled(self, width: int, height: int) -> "Intrinsics":
        """Scale pixel calibration to a resized image, including principal point."""
        if width <= 0 or height <= 0:
            raise ValueError("target dimensions must be positive")
        sx, sy = width / self.width, height / self.height
        return Intrinsics(width, height, self.fx * sx, self.fy * sy,
                          self.cx * sx, self.cy * sy)


@dataclass(frozen=True)
class CameraView:
    name: str
    world_to_camera: np.ndarray

    @property
    def camera_to_world_opencv(self) -> np.ndarray:
        return np.linalg.inv(_validated_rigid_transform(
            self.world_to_camera, self.name))

    @property
    def camera_to_world_opengl(self) -> np.ndarray:
        # OpenCV camera coordinates are right/down/forward; OpenGL are
        # right/up/backward. This changes only the camera basis, not world axes.
        cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
        return self.camera_to_world_opencv @ cv_to_gl

    @property
    def world_to_camera_gsplat(self) -> np.ndarray:
        """Return the supplied w2c transform for gsplat's rasterizer API.

        gsplat's 3DGS rasterizer uses a world-to-camera view matrix and treats
        positive camera Z as forward, matching these supplied OpenCV poses.
        Do not apply the NeRF/OpenGL Y/Z camera-axis flip here.
        """
        return _validated_rigid_transform(self.world_to_camera, self.name)


@dataclass(frozen=True)
class CameraSet:
    intrinsics: Intrinsics
    views: tuple[CameraView, ...]
    extras: dict[str, Any]


def _validated_rigid_transform(matrix: Any, label: str) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(m[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{label} has an invalid homogeneous bottom row")
    r = m[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=2e-3):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(r), 1.0, atol=2e-3):
        raise ValueError(f"{label} rotation determinant must be +1")
    return m


def load_camera_json(path: str | Path) -> CameraSet:
    """Load a supplied camera JSON without changing its view order."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    intr = Intrinsics(*(int(data[k]) for k in ("width", "height")),
                      *(float(data[k]) for k in ("fx", "fy", "cx", "cy")))
    if intr.width <= 0 or intr.height <= 0 or intr.fx <= 0 or intr.fy <= 0:
        raise ValueError("camera dimensions and focal lengths must be positive")
    views = tuple(
        CameraView(str(row["name"]),
                   _validated_rigid_transform(row["world_to_camera"], row["name"]))
        for row in data["views"]
    )
    if not views:
        raise ValueError("camera JSON contains no views")
    return CameraSet(intr, views, {k: v for k, v in data.items()
                                  if k not in {"width", "height", "fx", "fy", "cx", "cy", "views"}})


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to Nx3 points represented as columns."""
    m = _validated_rigid_transform(matrix, "transform")
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return (m[:3, :3] @ p.T + m[:3, 3:4]).T
