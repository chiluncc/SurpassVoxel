import numpy as np
from typing import Literal
from abc import ABC, abstractmethod
from collections.abc import Callable


type _InputDataType = np.ndarray[tuple[int, int, int, Literal[4]], np.dtype[np.uint8]]
type _OutputIndexType = np.ndarray[tuple[int], np.dtype[np.uint32]]
type _ItemType = np.ndarray[Literal[4], np.dtype[np.uint8]]


class BaseShape(ABC):

    def __init__(self, *, mask: _ItemType | None = None, value: _ItemType | None = None):
        super().__init__()
        if (mask is None) ^ (value is None):
            raise ValueError("mask and value must be given together")
        self._mask = np.zeros(4, dtype=np.uint8) if mask is None else mask
        self._value = np.zeros(4, dtype=np.uint8) if value is None else value

    @property
    def mask(self) -> _ItemType:
        return self._mask

    @property
    def value(self) -> _ItemType:
        return self._value

    @abstractmethod
    def get_masked(self, data: _InputDataType) -> _OutputIndexType:
        raise NotImplementedError

    @abstractmethod
    def is_cross_border(self, data: _InputDataType) -> bool:
        raise NotImplementedError


# ================================


def _masked_indices(
        data: _InputDataType,
        mask_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        mask: _ItemType,
        value: _ItemType,
        x: tuple[int, int],
        y: tuple[int, int],
        z: tuple[int, int],
        ) -> _OutputIndexType:
    nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]

    x0, x1 = max(x[0], 0), min(x[1], nx - 1)
    y0, y1 = max(y[0], 0), min(y[1], ny - 1)
    z0, z1 = max(z[0], 0), min(z[1], nz - 1)
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


class PointShape(BaseShape):

    def __init__(
            self,
            x: int,
            y: int,
            z: int,
            *,
            mask: _ItemType | None = None,
            value: _ItemType | None = None,
            ):
        super().__init__(mask=mask, value=value)
        self._x, self._y, self._z = int(x), int(y), int(z)

    @property
    def position(self) -> tuple[int, int, int]:
        return self._x, self._y, self._z

    def get_masked(self, data: _InputDataType) -> _OutputIndexType:
        return _masked_indices(
            data,
            lambda xs, ys, zs: np.ones((xs.size, ys.size, zs.size), dtype=bool),
            self._mask,
            self._value,
            (self._x, self._x),
            (self._y, self._y),
            (self._z, self._z),
        )

    def is_cross_border(self, data: _InputDataType) -> bool:
        nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]
        return not (0 <= self._x < nx and 0 <= self._y < ny and 0 <= self._z < nz)


class BoxShape(BaseShape):

    def __init__(
            self,
            x: int,
            y: int,
            z: int,
            x_size: int,
            y_size: int,
            z_size: int,
            *,
            mask: _ItemType | None = None,
            value: _ItemType | None = None,
            ):
        super().__init__(mask=mask, value=value)
        self._x, self._y, self._z = int(x), int(y), int(z)
        self._x_size, self._y_size, self._z_size = int(x_size), int(y_size), int(z_size)

    @property
    def position(self) -> tuple[int, int, int]:
        return self._x, self._y, self._z

    @property
    def size(self) -> tuple[int, int, int]:
        return self._x_size, self._y_size, self._z_size

    def get_masked(self, data: _InputDataType) -> _OutputIndexType:
        return _masked_indices(
            data,
            lambda xs, ys, zs: np.ones((xs.size, ys.size, zs.size), dtype=bool),
            self._mask,
            self._value,
            (self._x, self._x + self._x_size - 1),
            (self._y, self._y + self._y_size - 1),
            (self._z, self._z + self._z_size - 1),
        )

    def is_cross_border(self, data: _InputDataType) -> bool:
        nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]
        return (
            self._x < 0 or self._x + self._x_size > nx
            or self._y < 0 or self._y + self._y_size > ny
            or self._z < 0 or self._z + self._z_size > nz
        )


class SphereShape(BaseShape):

    def __init__(
            self,
            x: int,
            y: int,
            z: int,
            radius: int,
            *,
            mask: _ItemType | None = None,
            value: _ItemType | None = None,
            ):
        super().__init__(mask=mask, value=value)
        self._x, self._y, self._z = int(x), int(y), int(z)
        self._radius = int(radius)

    @property
    def position(self) -> tuple[int, int, int]:
        return self._x, self._y, self._z

    @property
    def radius(self) -> int:
        return self._radius

    def _mask_fn(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        distance = (xs[:, None, None] - self._x) ** 2
        distance = distance + (ys[None, :, None] - self._y) ** 2
        distance = distance + (zs[None, None, :] - self._z) ** 2
        return distance <= self._radius ** 2

    def get_masked(self, data: _InputDataType) -> _OutputIndexType:
        return _masked_indices(
            data,
            self._mask_fn,
            self._mask,
            self._value,
            (self._x - self._radius, self._x + self._radius),
            (self._y - self._radius, self._y + self._radius),
            (self._z - self._radius, self._z + self._radius),
        )

    def is_cross_border(self, data: _InputDataType) -> bool:
        nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]
        return (
            self._x - self._radius < 0 or self._x + self._radius >= nx
            or self._y - self._radius < 0 or self._y + self._radius >= ny
            or self._z - self._radius < 0 or self._z + self._radius >= nz
        )


class CylinderShape(BaseShape):

    def __init__(
            self,
            x: int,
            y: int,
            z: int,
            radius: int,
            y_size: int,
            *,
            mask: _ItemType | None = None,
            value: _ItemType | None = None,
            ):
        super().__init__(mask=mask, value=value)
        self._x, self._y, self._z = int(x), int(y), int(z)
        self._radius = int(radius)
        self._y_size = int(y_size)

    @property
    def position(self) -> tuple[int, int, int]:
        return self._x, self._y, self._z

    @property
    def radius(self) -> int:
        return self._radius

    @property
    def y_size(self) -> int:
        return self._y_size

    def _mask_fn(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        distance = (xs[:, None, None] - self._x) ** 2 + (zs[None, None, :] - self._z) ** 2
        return np.broadcast_to(distance <= self._radius ** 2, (xs.size, ys.size, zs.size))

    def get_masked(self, data: _InputDataType) -> _OutputIndexType:
        return _masked_indices(
            data,
            self._mask_fn,
            self._mask,
            self._value,
            (self._x - self._radius, self._x + self._radius),
            (self._y, self._y + self._y_size - 1),
            (self._z - self._radius, self._z + self._radius),
        )

    def is_cross_border(self, data: _InputDataType) -> bool:
        nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]
        return (
            self._x - self._radius < 0 or self._x + self._radius >= nx
            or self._y < 0 or self._y + self._y_size > ny
            or self._z - self._radius < 0 or self._z + self._radius >= nz
        )
