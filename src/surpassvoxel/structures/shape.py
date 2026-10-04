import numpy as np
from typing import Callable
from abc import ABC, abstractmethod

from surpassvoxel.structures.vector import Vector


type _ItemType = np.ndarray[tuple[int], np.dtype[np.uint8]]
type _InputType = np.ndarray[tuple[int, int, int, int], np.dtype[np.uint8]]
type _OutputType = np.ndarray[tuple[int], np.dtype[np.uint32]]


class BaseShape(ABC):

    def __init__(self, *, mask: _ItemType | None = None, condition: _ItemType | None = None):
        super().__init__()
        if (mask is None) ^ (condition is None):
            raise ValueError("mask and condition must be given together")
        self._mask = np.zeros(0, dtype=np.uint8) if mask is None else mask
        self._condition = np.zeros(0, dtype=np.uint8) if condition is None else condition

    @property
    def mask(self) -> _ItemType:
        return self._mask

    @property
    def condition(self) -> _ItemType:
        return self._condition

    @abstractmethod
    def get_masked(self, data: _InputType) -> _OutputType:
        raise NotImplementedError

    @abstractmethod
    def is_cross_border(self, data: _InputType) -> bool:
        raise NotImplementedError


def _masked_indices(
        data: _InputType,
        mask_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        mask: _ItemType,
        condition: _ItemType,
        v1: Vector,
        v2: Vector,
        ) -> _OutputType:
    nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]

    x0, y0, z0 = (max(component, 0) for component in v1.value)
    x1, y1, z1 = (min(component, size - 1) for component, size in zip(v2.value, (nx, ny, nz)))
    if x0 > x1 or y0 > y1 or z0 > z1:
        return np.empty(0, dtype=np.uint32)

    xs = np.arange(x0, x1 + 1, dtype=np.int64)
    ys = np.arange(y0, y1 + 1, dtype=np.int64)
    zs = np.arange(z0, z1 + 1, dtype=np.int64)
    inside = mask_fn(xs, ys, zs)

    if mask.any():
        item = data[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        inside = inside & (((item ^ condition) & mask) == 0).all(axis=-1)

    local = np.flatnonzero(inside)
    x_offset, remainder = np.divmod(local, (y1 - y0 + 1) * (z1 - z0 + 1))
    y_offset, z_offset = np.divmod(remainder, z1 - z0 + 1)
    return ((x0 + x_offset) * ny * nz + (y0 + y_offset) * nz + (z0 + z_offset)).astype(np.uint32)


# ================================


class PointShape(BaseShape):

    def __init__(
            self,
            position: Vector,
            *,
            mask: _ItemType | None = None,
            condition: _ItemType | None = None,
            ):
        super().__init__(mask=mask, condition=condition)
        self._position = position

    @property
    def position(self) -> Vector:
        return self._position

    def get_masked(self, data: _InputType) -> _OutputType:
        return _masked_indices(
            data,
            lambda xs, ys, zs: np.ones((xs.size, ys.size, zs.size), dtype=bool),
            self._mask,
            self._condition,
            self._position,
            self._position,
        )

    def is_cross_border(self, data: _InputType) -> bool:
        return not all(0 <= component < size for component, size in zip(self._position.value, data.shape))


class BoxShape(BaseShape):

    def __init__(
            self,
            position: Vector,
            size: Vector,
            *,
            mask: _ItemType | None = None,
            condition: _ItemType | None = None,
            ):
        super().__init__(mask=mask, condition=condition)
        self._position = position
        self._size = size

    @property
    def position(self) -> Vector:
        return self._position

    @property
    def size(self) -> Vector:
        return self._size

    def get_masked(self, data: _InputType) -> _OutputType:
        return _masked_indices(
            data,
            lambda xs, ys, zs: np.ones((xs.size, ys.size, zs.size), dtype=bool),
            self._mask,
            self._condition,
            self._position,
            self._position + self._size - Vector(1, 1, 1),
        )

    def is_cross_border(self, data: _InputType) -> bool:
        return any(
            position < 0 or position + size > limit
            for position, size, limit in zip(self._position.value, self._size.value, data.shape)
        )


class SphereShape(BaseShape):

    def __init__(
            self,
            position: Vector,
            radius: int,
            *,
            mask: _ItemType | None = None,
            condition: _ItemType | None = None,
            ):
        super().__init__(mask=mask, condition=condition)
        self._position = position
        self._radius = radius

    @property
    def position(self) -> Vector:
        return self._position

    @property
    def radius(self) -> int:
        return self._radius

    def _mask_fn(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        x, y, z = self._position.value
        distance = (xs[:, None, None] - x) ** 2
        distance = distance + (ys[None, :, None] - y) ** 2
        distance = distance + (zs[None, None, :] - z) ** 2
        return distance <= self._radius ** 2

    def get_masked(self, data: _InputType) -> _OutputType:
        radius = Vector(self._radius, self._radius, self._radius)
        return _masked_indices(
            data,
            self._mask_fn,
            self._mask,
            self._condition,
            self._position - radius,
            self._position + radius,
        )

    def is_cross_border(self, data: _InputType) -> bool:
        return any(
            position - self._radius < 0 or position + self._radius >= limit
            for position, limit in zip(self._position.value, data.shape)
        )


class CylinderShape(BaseShape):

    def __init__(
            self,
            position: Vector,
            radius: int,
            y_size: int,
            *,
            mask: _ItemType | None = None,
            condition: _ItemType | None = None,
            ):
        super().__init__(mask=mask, condition=condition)
        self._position = position
        self._radius = radius
        self._y_size = y_size

    @property
    def position(self) -> Vector:
        return self._position

    @property
    def radius(self) -> int:
        return self._radius

    @property
    def y_size(self) -> int:
        return self._y_size

    def _mask_fn(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        x, _, z = self._position.value
        distance = (xs[:, None, None] - x) ** 2 + (zs[None, None, :] - z) ** 2
        return np.broadcast_to(distance <= self._radius ** 2, (xs.size, ys.size, zs.size))

    def get_masked(self, data: _InputType) -> _OutputType:
        radius = Vector(self._radius, 0, self._radius)
        return _masked_indices(
            data,
            self._mask_fn,
            self._mask,
            self._condition,
            self._position - radius,
            self._position + radius + Vector(0, self._y_size - 1, 0),
        )

    def is_cross_border(self, data: _InputType) -> bool:
        x, y, z = self._position.value
        nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]
        return (
            x - self._radius < 0 or x + self._radius >= nx
            or y < 0 or y + self._y_size > ny
            or z - self._radius < 0 or z + self._radius >= nz
        )


class PrismShape(BaseShape):

    def __init__(
            self,
            position: Vector,
            v0: Vector,
            v1: Vector,
            v2: Vector,
            height: int,
            *,
            mask: _ItemType | None = None,
            condition: _ItemType | None = None,
            ):
        super().__init__(mask=mask, condition=condition)
        self._position = position
        self._vertices = (v0, v1, v2,)
        self._height = height

    @property
    def position(self) -> Vector:
        return self._position

    @property
    def vertices(self) -> tuple[Vector, Vector, Vector]:
        return self._vertices

    @property
    def height(self) -> int:
        return self._height

    def _points(self) -> tuple[Vector, ...]:
        return tuple(self._position + vertex for vertex in self._vertices)

    def _bounds(self) -> tuple[Vector, Vector]:
        points = self._points()
        return Vector.min(*points), Vector.max(*points)

    @staticmethod
    def _outside(v1: Vector, v2: Vector, data: _InputType) -> bool:
        return any(low < 0 or high >= limit for low, high, limit in zip(v1.value, v2.value, data.shape))

    @staticmethod
    def _half_space(
            xs: np.ndarray,
            ys: np.ndarray,
            zs: np.ndarray,
            normal: Vector,
            point: Vector,
            ) -> np.ndarray:
        vx, vy, vz = normal.value
        px, py, pz = point.value
        return (
            vx * (xs[:, None, None] - px)
            + vy * (ys[None, :, None] - py)
            + vz * (zs[None, None, :] - pz)
        )

    def _mask_fn(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        points = self._points()
        inside = np.ones((xs.size, ys.size, zs.size), dtype=bool)

        y_axis = Vector(0, 1, 0)
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            normal = (points[j] - points[i]).cross(y_axis)
            reference = normal.dot(points[k] - points[i])
            inside &= (self._half_space(xs, ys, zs, normal, points[i]) * reference) >= 0

        normal = (points[1] - points[0]).cross(points[2] - points[0])
        depth = normal.y
        cap = self._half_space(xs, ys, zs, normal, points[0])
        inside &= (cap * depth) >= 0
        inside &= ((cap - depth * (self._height - 1)) * depth) <= 0
        return inside

    def get_masked(self, data: _InputType) -> _OutputType:
        v1, v2 = self._bounds()
        return _masked_indices(
            data,
            self._mask_fn,
            self._mask,
            self._condition,
            v1,
            v2 + Vector(0, self._height - 1, 0),
        )

    def is_cross_border(self, data: _InputType) -> bool:
        v1, v2 = self._bounds()
        return self._outside(v1, v2 + Vector(0, self._height - 1, 0), data)


class PyramidShape(BaseShape):

    def __init__(
            self,
            position: Vector,
            v0: Vector,
            v1: Vector,
            v2: Vector,
            v3: Vector,
            *,
            mask: _ItemType | None = None,
            condition: _ItemType | None = None,
            ):
        super().__init__(mask=mask, condition=condition)
        self._position = position
        self._vertices = (v0, v1, v2, v3)

    @property
    def position(self) -> Vector:
        return self._position

    @property
    def vertices(self) -> tuple[Vector, Vector, Vector, Vector]:
        return self._vertices

    def _points(self) -> tuple[Vector, ...]:
        return tuple(self._position + vertex for vertex in self._vertices)

    def _bounds(self) -> tuple[Vector, Vector]:
        points = self._points()
        return Vector.min(*points), Vector.max(*points)

    @staticmethod
    def _outside(v1: Vector, v2: Vector, data: _InputType) -> bool:
        return any(low < 0 or high >= limit for low, high, limit in zip(v1.value, v2.value, data.shape))

    @staticmethod
    def _half_space(
            xs: np.ndarray,
            ys: np.ndarray,
            zs: np.ndarray,
            normal: Vector,
            point: Vector,
            ) -> np.ndarray:
        vx, vy, vz = normal.value
        px, py, pz = point.value
        return (
            vx * (xs[:, None, None] - px)
            + vy * (ys[None, :, None] - py)
            + vz * (zs[None, None, :] - pz)
        )

    def _mask_fn(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        points = self._points()
        inside = np.ones((xs.size, ys.size, zs.size), dtype=bool)
        for i, j, k, opposite in ((0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0)):
            normal = (points[j] - points[i]).cross(points[k] - points[i])
            reference = normal.dot(points[opposite] - points[i])
            inside &= (self._half_space(xs, ys, zs, normal, points[i]) * reference) >= 0
        return inside

    def get_masked(self, data: _InputType) -> _OutputType:
        return _masked_indices(
            data,
            self._mask_fn,
            self._mask,
            self._condition,
            *self._bounds(),
        )

    def is_cross_border(self, data: _InputType) -> bool:
        return self._outside(*self._bounds(), data)
