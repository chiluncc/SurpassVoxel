import math
import numpy as np

from surpassvoxel.structures.typing import Vector3Df, Vector3Di
from surpassvoxel.structures.voxel.base import Voxel


# ================================


def translation(
        voxel: Voxel,
        offset: Vector3Di,
        *,
        strict: bool = True,
        ) -> Voxel | str:
    bounds = voxel.aabb()
    if bounds is None:
        return voxel.clone()
    low, high = bounds

    if strict:
        for axis, shift in enumerate(offset):
            if low[axis] + shift < 0 or high[axis] + shift >= voxel._shape[axis]:
                return "moved voxels cross the voxel border"

    source_slices, target_slices = [], []
    after = voxel.shape_like()
    for axis, shift in enumerate(offset):
        shift = int(shift)
        first = max(0, -shift)
        last = min(voxel._shape[axis], voxel._shape[axis] - shift)
        if first >= last:
            return after
        source_slices.append(slice(first, last))
        target_slices.append(slice(first + shift, last + shift))

    after._data[tuple(target_slices)] = voxel._data[tuple(source_slices)]
    return after


def rotation(
        voxel: Voxel,
        axis: Vector3Df,
        degrees: float,
        point: Vector3Df,
        *,
        strict: bool = True,
        ) -> Voxel | str:
    if degrees % 360 == 0:
        return voxel.clone()

    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm == 0:
        return "rotation axis must be non-zero"
    ax, ay, az = ax / norm, ay / norm, az / norm

    angle = math.radians(degrees)
    sine, cosine = math.sin(angle), math.cos(angle)
    complement = 1 - cosine
    matrix = np.array([
        [cosine + ax * ax * complement, ax * ay * complement - az * sine, ax * az * complement + ay * sine],
        [ay * ax * complement + az * sine, cosine + ay * ay * complement, ay * az * complement - ax * sine],
        [az * ax * complement - ay * sine, az * ay * complement + ax * sine, cosine + az * az * complement],
    ], dtype=np.float64)

    grid = voxel._data
    occupied = (grid[..., 0] & 0x80) != 0
    if not occupied.any():
        return voxel.clone()

    nx, ny, nz = voxel._shape
    limit = np.array(voxel._shape, dtype=np.int64)[:, None]
    pivot = point

    if strict:
        for x0 in range(0, nx, 16):
            x1 = min(x0 + 16, nx)
            slice_occupied = occupied[x0:x1]
            if not slice_occupied.any():
                continue
            ix, iy, iz = np.nonzero(slice_occupied)
            points = np.stack((ix + x0 - pivot[0], iy - pivot[1], iz - pivot[2]), axis=0)
            landed = np.floor(matrix @ points + pivot[:, None] + 0.5).astype(np.int64)
            if ((landed < 0) | (landed >= limit)).any():
                return "rotated voxels cross the voxel border"

    inverse = matrix.T
    ys = np.arange(ny, dtype=np.float64) - pivot[1]
    zs = np.arange(nz, dtype=np.float64) - pivot[2]
    after = voxel.shape_like()
    for x0 in range(0, nx, 16):
        x1 = min(x0 + 16, nx)
        xs = np.arange(x0, x1, dtype=np.float64) - pivot[0]
        source_x = np.floor(inverse[0, 0] * xs[:, None, None] + inverse[0, 1] * ys[None, :, None] + inverse[0, 2] * zs[None, None, :] + pivot[0] + 0.5).astype(np.int64)
        source_y = np.floor(inverse[1, 0] * xs[:, None, None] + inverse[1, 1] * ys[None, :, None] + inverse[1, 2] * zs[None, None, :] + pivot[1] + 0.5).astype(np.int64)
        source_z = np.floor(inverse[2, 0] * xs[:, None, None] + inverse[2, 1] * ys[None, :, None] + inverse[2, 2] * zs[None, None, :] + pivot[2] + 0.5).astype(np.int64)
        inside = (0 <= source_x) & (source_x < nx) & (0 <= source_y) & (source_y < ny) & (0 <= source_z) & (source_z < nz)
        block = np.zeros((x1 - x0, ny, nz, 4), dtype=np.uint8)
        if inside.any():
            block[inside] = grid[source_x[inside], source_y[inside], source_z[inside]]
        after._data[x0:x1] = block
    return after


def scaling(
        voxel: Voxel,
        factor: Vector3Df,
        point: Vector3Df,
        *,
        strict: bool = True,
        ) -> Voxel | str:
    if not all(value > 0 for value in factor):
        return "scale factors must be positive"

    grid = voxel._data
    bounds = voxel.aabb()
    if bounds is None or all(value == 1.0 for value in factor):
        return voxel.clone()

    low, high = [int(value) for value in bounds[0]], [int(value) for value in bounds[1]]
    pivot = point

    cells = []
    for axis, value in enumerate(factor):
        extent = high[axis] - low[axis] + 1
        size = max(1, int(math.floor(extent * value + 0.5)))
        image = pivot[axis] + (low[axis] - pivot[axis]) * value
        first = int(math.floor(image + (extent * value - size) / 2.0 + 0.5))
        last = first + size - 1
        if strict and (first < 0 or last >= voxel._shape[axis]):
            return "scaled voxels cross the voxel border"
        cells.append([max(first, 0), min(last, voxel._shape[axis] - 1)])
    if any(first > last for first, last in cells):
        return voxel.shape_like()

    source = grid[low[0]:high[0] + 1, low[1]:high[1] + 1, low[2]:high[2] + 1]

    weights, cores, nearest = [], [], []
    for axis, value in enumerate(factor):
        index = np.arange(cells[axis][0], cells[axis][1] + 1, dtype=np.float64)
        before = pivot[axis] + (index - pivot[axis]) / value
        after = pivot[axis] + (index + 1 - pivot[axis]) / value
        edges = np.arange(low[axis], high[axis] + 2, dtype=np.float64)
        overlap = np.minimum(after[:, None], edges[None, 1:]) - np.maximum(before[:, None], edges[None, :-1])
        overlap = np.clip(overlap, 0.0, None)
        weights.append(overlap.astype(np.float32))
        cores.append((overlap >= 0.5 * min(1.0, 1.0 / value)).astype(np.float32))
        nearest.append(np.clip(
            np.floor(pivot[axis] + (index + 0.5 - pivot[axis]) / value),
            low[axis], high[axis],
        ).astype(np.int64))

    def weave(block: np.ndarray, weight: np.ndarray, axis: int) -> np.ndarray:
        woven = np.einsum("ij,j...->i...", weight, np.moveaxis(block, axis, 0))
        return np.moveaxis(woven, 0, axis)

    total = np.ones([high[axis] - low[axis] + 1 for axis in range(3)], dtype=np.float32)
    filled = ((source[..., 0] & 0x80) != 0).astype(np.float32)
    core = filled.copy()
    for axis in range(3):
        total = weave(total, weights[axis], axis)
        filled = weave(filled, weights[axis], axis)
        core = weave(core, cores[axis], axis)
    occupied = ((total > 0.0) & (filled >= 0.5 * total)) | (core >= 1.0 - 1e-6)

    def gather(shift: tuple[int, int, int]) -> np.ndarray:
        return source[
            np.clip(nearest[0] + shift[0] - low[0], 0, high[0] - low[0])[:, None, None],
            np.clip(nearest[1] + shift[1] - low[1], 0, high[1] - low[1])[None, :, None],
            np.clip(nearest[2] + shift[2] - low[2], 0, high[2] - low[2])[None, None, :],
        ]

    picked = gather((0, 0, 0))
    empty = occupied & ((picked[..., 0] & 0x80) == 0)
    reach = int(math.ceil(max(1.0 / value for value in factor))) + 1
    offsets = []
    for distance in range(1, reach + 1):
        for axis in range(3):
            for side in (-1, 1):
                shift = [0, 0, 0]
                shift[axis] = side * distance
                offsets.append(tuple(shift))
    for shift in offsets:
        if not empty.any():
            break
        shifted = gather(shift)
        take = empty & ((shifted[..., 0] & 0x80) != 0)
        picked = np.where(take[..., None], shifted, picked)
        empty &= ~take
    occupied &= ~empty
    block = np.where(occupied[..., None], picked, np.zeros(4, dtype=np.uint8)).astype(np.uint8)

    after = voxel.shape_like()
    after._data[cells[0][0]:cells[0][1] + 1, cells[1][0]:cells[1][1] + 1, cells[2][0]:cells[2][1] + 1] = block
    return after
