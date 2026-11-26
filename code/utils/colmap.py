"""Minimal COLMAP binary readers needed by the project adapters."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SparsePoints:
    xyz: np.ndarray
    rgb: np.ndarray
    point_ids: np.ndarray
    reprojection_error: np.ndarray


def load_points3d_binary(path: str | Path) -> SparsePoints:
    """Read COLMAP points3D.bin without requiring pycolmap.

    COLMAP's binary record is point id (uint64), XYZ (3×float64), RGB
    (3×uint8), reprojection error (float64), and a variable image/2D-point
    track. Tracks are skipped after validating all record boundaries.
    """
    xyz: list[tuple[float, float, float]] = []
    rgb: list[tuple[int, int, int]] = []
    ids: list[int] = []
    errors: list[float] = []
    with Path(path).open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("truncated COLMAP points3D header")
        (count,) = struct.unpack("<Q", raw)
        for _ in range(count):
            header = f.read(43)
            if len(header) != 43:
                raise ValueError("truncated COLMAP points3D record")
            point_id = struct.unpack_from("<Q", header, 0)[0]
            point_xyz = struct.unpack_from("<3d", header, 8)
            point_rgb = struct.unpack_from("<3B", header, 32)
            error = struct.unpack_from("<d", header, 35)[0]
            track_raw = f.read(8)
            if len(track_raw) != 8:
                raise ValueError("truncated COLMAP point track length")
            (track_length,) = struct.unpack("<Q", track_raw)
            track_bytes = f.read(track_length * 8)  # image_id:uint32, point2D_idx:uint32
            if len(track_bytes) != track_length * 8:
                raise ValueError("truncated COLMAP point track")
            ids.append(point_id)
            xyz.append(point_xyz)
            rgb.append(point_rgb)
            errors.append(error)
        if f.read(1):
            raise ValueError("unexpected trailing bytes in COLMAP points3D file")
    return SparsePoints(np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
                        np.asarray(rgb, dtype=np.uint8).reshape(-1, 3),
                        np.asarray(ids, dtype=np.uint64),
                        np.asarray(errors, dtype=np.float64))
