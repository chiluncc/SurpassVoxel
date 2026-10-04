import numpy as np
from typing import Callable
from abc import ABC, abstractmethod

from surpassvoxel.structures.vector import Vector


type _ItemType = np.ndarray[tuple[int], np.dtype[np.uint8]]
type _InputType = np.ndarray[tuple[int, int, int, int], np.dtype[np.uint8]]
type _OutputType = np.ndarray[tuple[int], np.dtype[np.uint32]]


class BaseShape(ABC):

    def __init__(self, *, mask: _ItemType | None = None, value: _ItemType | None = None):
        super().__init__()
        if (mask is None) ^ (value is None):
            raise ValueError("mask and value must be given together")
        self._mask = np.zeros(0, dtype=np.uint8) if mask is None else mask
        self._value = np.zeros(0, dtype=np.uint8) if value is None else value

    @property
    def mask(self) -> _ItemType:
        return self._mask

    @property
    def value(self) -> _ItemType:
        return self._value

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
        value: _ItemType,
        v1: Vector,
        v2: Vector,
        ) -> _OutputType:
    nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]

    x0, x1 = max(v1.x, 0), min(v2.x, nx - 1)
    y0, y1 = max(v1.y, 0), min(v2.y, ny - 1)
    z0, z1 = max(v1.z, 0), min(v2.z, nz - 1)
    if x0 > x1 or y0 > y1 or z0 > z1:
        return np.empty(0, dtype=np.uint32)

    xs = np.arange(x0, x1 + 1, dtype=np.int64)
    ys = np.arange(y0, y1 + 1, dtype=np.int64)
    zs = np.arange(z0, z1 + 1, dtype=np.int64)
    inside = mask_fn(xs, ys, zs)

    if mask.any():
        item = data[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        inside = inside & (((item ^ value) & mask) == 0).all(axis=-1)

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
            value: _ItemType | None = None,
            ):
        super().__init__(mask=mask, value=value)
        self._position = Vector(int(position.x), int(position.y), int(position.z))

    @property
    def position(self) -> Vector:
        return self._position

    def get_masked(self, data: _InputType) -> _OutputType:
        return _masked_indices(
            data,
            lambda xs, ys, zs: np.ones((xs.size, ys.size, zs.size), dtype=bool),
            self._mask,
            self._value,
            self._position,
            self._position,
        )

    def is_cross_border(self, data: _InputType) -> bool:
        nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]
        return not (
            0 <= self._position.x < nx
            and 0 <= self._position.y < ny
            and 0 <= self._position.z < nz
        )


class BoxShape(BaseShape):

    def __init__(
            self,
            position: Vector,
            size: Vector,
            *,
            mask: _ItemType | None = None,
            value: _ItemType | None = None,
            ):
        super().__init__(mask=mask, value=value)
        self._position = Vector(int(position.x), int(position.y), int(position.z))
        self._size = Vector(int(size.x), int(size.y), int(size.z))

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
            self._value,
            self._position,
            self._position + self._size - Vector(1, 1, 1),
        )

    def is_cross_border(self, data: _InputType) -> bool:
        nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]
        return (
            self._position.x < 0 or self._position.x + self._size.x > nx
            or self._position.y < 0 or self._position.y + self._size.y > ny
            or self._position.z < 0 or self._position.z + self._size.z > nz
        )


class SphereShape(BaseShape):

    def __init__(
            self,
            position: Vector,
            radius: int,
            *,
            mask: _ItemType | None = None,
            value: _ItemType | None = None,
            ):
        super().__init__(mask=mask, value=value)
        self._position = Vector(int(position.x), int(position.y), int(position.z))
        self._radius = int(radius)

    @property
    def position(self) -> Vector:
        return self._position

    @property
    def radius(self) -> int:
        return self._radius

    def _mask_fn(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        distance = (xs[:, None, None] - self._position.x) ** 2
        distance = distance + (ys[None, :, None] - self._position.y) ** 2
        distance = distance + (zs[None, None, :] - self._position.z) ** 2
        return distance <= self._radius ** 2

    def get_masked(self, data: _InputType) -> _OutputType:
        radius = Vector(self._radius, self._radius, self._radius)
        return _masked_indices(
            data,
            self._mask_fn,
            self._mask,
            self._value,
            self._position - radius,
            self._position + radius,
        )

    def is_cross_border(self, data: _InputType) -> bool:
        nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]
        return (
            self._position.x - self._radius < 0 or self._position.x + self._radius >= nx
            or self._position.y - self._radius < 0 or self._position.y + self._radius >= ny
            or self._position.z - self._radius < 0 or self._position.z + self._radius >= nz
        )


class CylinderShape(BaseShape):

    def __init__(
            self,
            position: Vector,
            radius: int,
            y_size: int,
            *,
            mask: _ItemType | None = None,
            value: _ItemType | None = None,
            ):
        super().__init__(mask=mask, value=value)
        self._position = Vector(int(position.x), int(position.y), int(position.z))
        self._radius = int(radius)
        self._y_size = int(y_size)

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
        distance = (xs[:, None, None] - self._position.x) ** 2 + (zs[None, None, :] - self._position.z) ** 2
        return np.broadcast_to(distance <= self._radius ** 2, (xs.size, ys.size, zs.size))

    def get_masked(self, data: _InputType) -> _OutputType:
        radius = Vector(self._radius, 0, self._radius)
        return _masked_indices(
            data,
            self._mask_fn,
            self._mask,
            self._value,
            self._position - radius,
            self._position + radius + Vector(0, self._y_size - 1, 0),
        )

    def is_cross_border(self, data: _InputType) -> bool:
        nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]
        return (
            self._position.x - self._radius < 0 or self._position.x + self._radius >= nx
            or self._position.y < 0 or self._position.y + self._y_size > ny
            or self._position.z - self._radius < 0 or self._position.z + self._radius >= nz
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
            value: _ItemType | None = None,
            ):
        super().__init__(mask=mask, value=value)
        self._position = Vector(int(position.x), int(position.y), int(position.z))
        self._vertices = (
            Vector(int(v0.x), int(v0.y), int(v0.z)),
            Vector(int(v1.x), int(v1.y), int(v1.z)),
            Vector(int(v2.x), int(v2.y), int(v2.z)),
        )
        self._height = int(height)

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
        nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]
        return (
            v1.x < 0 or v2.x >= nx
            or v1.y < 0 or v2.y >= ny
            or v1.z < 0 or v2.z >= nz
        )

    @staticmethod
    def _half_space(
            xs: np.ndarray,
            ys: np.ndarray,
            zs: np.ndarray,
            normal: Vector,
            point: Vector,
            ) -> np.ndarray:
        return (
            normal.x * (xs[:, None, None] - point.x)
            + normal.y * (ys[None, :, None] - point.y)
            + normal.z * (zs[None, None, :] - point.z)
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
            self._value,
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
            value: _ItemType | None = None,
            ):
        super().__init__(mask=mask, value=value)
        self._position = Vector(int(position.x), int(position.y), int(position.z))
        self._vertices = (
            Vector(int(v0.x), int(v0.y), int(v0.z)),
            Vector(int(v1.x), int(v1.y), int(v1.z)),
            Vector(int(v2.x), int(v2.y), int(v2.z)),
            Vector(int(v3.x), int(v3.y), int(v3.z)),
        )

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
        nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]
        return (
            v1.x < 0 or v2.x >= nx
            or v1.y < 0 or v2.y >= ny
            or v1.z < 0 or v2.z >= nz
        )

    @staticmethod
    def _half_space(
            xs: np.ndarray,
            ys: np.ndarray,
            zs: np.ndarray,
            normal: Vector,
            point: Vector,
            ) -> np.ndarray:
        return (
            normal.x * (xs[:, None, None] - point.x)
            + normal.y * (ys[None, :, None] - point.y)
            + normal.z * (zs[None, None, :] - point.z)
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
            self._value,
            *self._bounds(),
        )

    def is_cross_border(self, data: _InputType) -> bool:
        return self._outside(*self._bounds(), data)
