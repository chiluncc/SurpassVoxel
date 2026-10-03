import numpy as np
import numpy.typing as npt
from typing import Literal, Iterator, Self

from surpassvoxel.structures.color import PALETTE_TYPE, get_palette, ColorSpace, ColorType
from surpassvoxel.structures.shape import BaseShape


type VoxelMiddleDataType = np.ndarray[tuple[int, Literal[4]], np.dtype[np.uint8]]

class Voxel:
    """
    Coordinates are right handed, +x right, +y up, +z out of the screen, origin at
    the (0, 0, 0) corner, item (x, y, z) with 0 <= x < x_size and so on.

    Item layout, four bytes per voxel::

        byte0     byte1     byte2     byte3
        10000000  22222222  00000000  00000000

    byte0 bit7 (0x80) : voxel exist, 1 = occupied, 0 = empty
    byte0 bit6..bit0  : reserved
    byte1 bit7..bit0  : color index, 0..255
    byte2             : reserved
    byte3             : reserved

    An empty voxel is four 0x00 bytes. Read as a little endian uint32, an
    occupied voxel with color index c is 0x80 | (c << 8).
    """
    
    def __init__(self, x_size: int, y_size: int, z_size: int, *, color_space: PALETTE_TYPE = "MAGICVOXEL_255"):
        self._shape = (x_size, y_size, z_size)
        self._color_space_type = color_space
        
        if min(self._shape) <= 0:
            raise ValueError("All size need greater than zero")
        if max(self._shape) > 256:
            raise ValueError("Too large in size")

        self._data = np.zeros((*self._shape, 4), dtype=np.uint8)
        self._color_space = ColorSpace(get_palette(color_space), name=color_space)

    def __iter__(self) -> Iterator[npt.NDArray[np.uint8]]:
        return iter(self._data.copy().reshape(-1, 4))

    @property
    def color_space_type(self) -> PALETTE_TYPE | None:
        return self._color_space_type

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def shape(self) -> tuple[int]:
        return self._data.shape

    @property
    def data(self) -> VoxelMiddleDataType:
        return self._data.copy().reshape(-1, 4)

    @data.setter
    def data(self, data: VoxelMiddleDataType):
        self._data = data.copy().reshape(*self._data.shape)

    # ================================

    def change_color_space(self, color_space: ColorSpace) -> Self:
        source = self._color_space.get_color(np.arange(len(self._color_space)))
        lut = color_space.get_index(source).astype(np.uint8)

        data = self._data.copy()
        items = data.reshape(-1, 4)
        occupied = (items[:, 0] & 0x80) != 0
        items[occupied, 1] = lut[items[occupied, 1]]

        new = type(self).__new__(type(self))
        new._shape = self._shape
        new._color_space_type = color_space.name
        new._color_space = color_space
        new._data = data
        return new

    def clean(self) -> None:
        self._data.fill(0)

    def touch(self, shape: BaseShape) -> list[tuple[int, ...]]:
        _, ny, nz = self._shape
        indices = shape.get_masked(self._data)
        items = self._data.reshape(-1, 4)[indices]
        keep = (items[:, 0] & 0x80) != 0
        indices, items = indices[keep], items[keep]
        if indices.size == 0:
            return []

        xyz = np.stack([indices // (ny * nz), (indices // nz) % ny, indices % nz], axis=1)
        rows = np.concatenate((xyz, self._color_space.get_color(items[:, 1])), axis=1)
        return [tuple(map(int, row)) for row in rows]

    def modify(self, shape: BaseShape, value: np.ndarray[tuple[Literal[4]], np.dtype[np.uint8]], *, strict: bool = True) -> str | None:
        if strict and shape.is_cross_border(self._data):
            return f"{type(shape).__name__} crosses the voxel border"

        self._data.reshape(-1, 4)[shape.get_masked(self._data)] = value
        return None

    def mirror(self, *, source: Literal["low", "high"], axis: Literal["YZ", "XZ", "XY"] = "YZ") -> None:
        axis_no = {"YZ": 0, "XZ": 1, "XY": 2}[axis]
        size = self._shape[axis_no]
        low = np.arange(size // 2)
        high = size - 1 - low
        src, dst = (low, high) if source == "low" else (high, low)

        src_sel, dst_sel = [slice(None)] * 4, [slice(None)] * 4
        src_sel[axis_no], dst_sel[axis_no] = src, dst
        self._data[tuple(dst_sel)] = self._data[tuple(src_sel)]
