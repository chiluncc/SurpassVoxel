import numpy as np
import math
from typing import Callable
from abc import ABC, abstractmethod

from surpassvoxel.structures.typing import GridBool, Vector3Di, Vector3Dfs, VoxelItemType, VoxelItemsType


_LINE_SAMPLES_PER_UNIT = 4


class BaseShape3D(ABC):

    def __init__(self, *, mask: VoxelItemType | None = None, condition: VoxelItemType | None = None) -> None:
        super().__init__()
        if (mask is None) ^ (condition is None):
            raise ValueError("mask and condition must be given together")
        self._mask = np.zeros(4, dtype=np.uint8) if mask is None else mask
        self._condition = np.zeros(4, dtype=np.uint8) if condition is None else condition

    @property
    def mask(self) -> VoxelItemType:
        return self._mask

    @property
    def condition(self) -> VoxelItemType:
        return self._condition

    @abstractmethod
    def get_masked(self, data: VoxelItemsType) -> GridBool:
        raise NotImplementedError

    @abstractmethod
    def is_cross_border(self, data: VoxelItemsType) -> bool:
        raise NotImplementedError


def _masked_grid(
        data: VoxelItemsType,
        mask_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        mask: VoxelItemType,
        condition: VoxelItemType,
        v1: Vector3Di,
        v2: Vector3Di,
        ) -> GridBool:
    nx, ny, nz = data.shape[:3]
    grid = np.zeros((nx, ny, nz), dtype=bool)

    x0, y0, z0 = (max(component, 0) for component in v1)
    x1, y1, z1 = (min(component, size - 1) for component, size in zip(v2, (nx, ny, nz)))
    if x0 > x1 or y0 > y1 or z0 > z1:
        return grid

    xs = np.arange(x0, x1 + 1, dtype=np.int64)
    ys = np.arange(y0, y1 + 1, dtype=np.int64)
    zs = np.arange(z0, z1 + 1, dtype=np.int64)
    inside = mask_fn(xs, ys, zs)

    if mask.any():
        item = data[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        inside = inside & (((item ^ condition) & mask) == 0).all(axis=-1)

    grid[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = inside
    return grid


def _bezier_curve(points: Vector3Dfs) -> tuple[Vector3Dfs, Vector3Dfs]:
    degree = len(points) - 1
    step = float(np.linalg.norm(np.diff(points, axis=0), axis=1).max())
    count = max(2, math.ceil(_LINE_SAMPLES_PER_UNIT * degree * step) + 1)
    t = np.linspace(0.0, 1.0, count)
    complement = 1.0 - t

    curve = np.zeros((count, 3), dtype=np.float64)
    tangent = np.zeros((count, 3), dtype=np.float64)
    for index in range(degree + 1):
        weight = math.comb(degree, index) * complement ** (degree - index) * t ** index
        curve += weight[:, None] * points[index]
        if index < degree:
            slope = math.comb(degree - 1, index) * complement ** (degree - 1 - index) * t ** index
            tangent += slope[:, None] * (points[index + 1] - points[index])
    tangent *= degree

    length = np.linalg.norm(tangent, axis=1)
    tangent = tangent / np.where(length > 0.0, length, 1.0)[:, None]
    return curve, tangent


# ================================


class PointShape(BaseShape3D):

    def __init__(
            self,
            position: Vector3Di,
            *,
            mask: VoxelItemType | None = None,
            condition: VoxelItemType | None = None,
            ) -> None:
        super().__init__(mask=mask, condition=condition)
        self._position = position

    @property
    def position(self) -> Vector3Di:
        return self._position

    def get_masked(self, data: VoxelItemsType) -> GridBool:
        return _masked_grid(
            data,
            lambda xs, ys, zs: np.ones((xs.size, ys.size, zs.size), dtype=bool),
            self._mask,
            self._condition,
            self._position,
            self._position,
        )

    def is_cross_border(self, data: VoxelItemsType) -> bool:
        return not all(0 <= component < size for component, size in zip(self._position, data.shape))


class BoxShape(BaseShape3D):

    def __init__(
            self,
            position: Vector3Di,
            size: Vector3Di,
            *,
            mask: VoxelItemType | None = None,
            condition: VoxelItemType | None = None,
            ) -> None:
        super().__init__(mask=mask, condition=condition)
        self._position = position
        self._size = size

    @property
    def position(self) -> Vector3Di:
        return self._position

    @property
    def size(self) -> Vector3Di:
        return self._size

    def get_masked(self, data: VoxelItemsType) -> GridBool:
        return _masked_grid(
            data,
            lambda xs, ys, zs: np.ones((xs.size, ys.size, zs.size), dtype=bool),
            self._mask,
            self._condition,
            self._position,
            self._position + self._size - 1,
        )

    def is_cross_border(self, data: VoxelItemsType) -> bool:
        return any(
            position < 0 or position + size > limit
            for position, size, limit in zip(self._position, self._size, data.shape)
        )


class SphereShape(BaseShape3D):

    def __init__(
            self,
            position: Vector3Di,
            radius: int,
            *,
            mask: VoxelItemType | None = None,
            condition: VoxelItemType | None = None,
            ) -> None:
        super().__init__(mask=mask, condition=condition)
        self._position = position
        self._radius = radius

    @property
    def position(self) -> Vector3Di:
        return self._position

    @property
    def radius(self) -> int:
        return self._radius

    def _mask_fn(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        x, y, z = self._position
        distance = (xs[:, None, None] - x) ** 2
        distance = distance + (ys[None, :, None] - y) ** 2
        distance = distance + (zs[None, None, :] - z) ** 2
        return distance <= self._radius ** 2

    def get_masked(self, data: VoxelItemsType) -> GridBool:
        radius = np.full(3, self._radius, dtype=np.int32)
        return _masked_grid(
            data,
            self._mask_fn,
            self._mask,
            self._condition,
            self._position - radius,
            self._position + radius,
        )

    def is_cross_border(self, data: VoxelItemsType) -> bool:
        return any(
            position - self._radius < 0 or position + self._radius >= limit
            for position, limit in zip(self._position, data.shape)
        )


class LineShape(BaseShape3D):

    def __init__(
            self,
            points: Vector3Dfs,
            *,
            mask: VoxelItemType | None = None,
            condition: VoxelItemType | None = None,
            ) -> None:
        super().__init__(mask=mask, condition=condition)
        if not 2 <= len(points) <= 5:
            raise ValueError("line needs 2 to 5 control points")

        controls = np.array(points, dtype=np.float64)
        controls[0] = np.rint(controls[0])
        controls[-1] = np.rint(controls[-1])
        self._points = controls
        self._curve, self._tangents = _bezier_curve(controls)
        self._cells = np.rint(self._curve).astype(np.int32)

    @property
    def points(self) -> Vector3Dfs:
        return self._points

    def get_points_and_tangents(self) -> tuple[Vector3Dfs, Vector3Dfs]:
        return self._curve, self._tangents

    def _mask_fn(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        inside = np.zeros((xs.size, ys.size, zs.size), dtype=bool)
        low = xs[0], ys[0], zs[0]
        high = xs[-1], ys[-1], zs[-1]

        keep = np.ones(len(self._cells), dtype=bool)
        for axis in range(3):
            keep &= (self._cells[:, axis] >= low[axis]) & (self._cells[:, axis] <= high[axis])
        cells = self._cells[keep]
        inside[cells[:, 0] - low[0], cells[:, 1] - low[1], cells[:, 2] - low[2]] = True
        return inside

    def get_masked(self, data: VoxelItemsType) -> GridBool:
        return _masked_grid(
            data,
            self._mask_fn,
            self._mask,
            self._condition,
            np.floor(self._points.min(axis=0)).astype(np.int32),
            np.ceil(self._points.max(axis=0)).astype(np.int32),
        )

    def is_cross_border(self, data: VoxelItemsType) -> bool:
        return bool(((self._cells < 0) | (self._cells >= data.shape[:3])).any())
