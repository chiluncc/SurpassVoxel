import math
import struct
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Literal, Iterator, Self

from surpassvoxel.structures.color import PaletteType, ColorType, get_palette, ColorSpace
from surpassvoxel.structures.shape import BaseShape
from surpassvoxel.structures.vector import Vector


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
    
    _MAGIC = b"SSVOX"
    _VERSION = 1
    _HEADER_SIZE = 20
    _NAME_TERMINATOR = b"\x00\x00\x00\x00"

    def __init__(self, size: Vector, *, color_space: PaletteType = "MAGICVOXEL_255"):
        self._shape = size.value
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
    def color_space_type(self) -> PaletteType | None:
        return self._color_space_type

    @property
    def size(self) -> int:
        return self._shape[0] * self._shape[1] * self._shape[2]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def data(self) -> np.ndarray[tuple[int, Literal[4]], np.dtype[np.uint8]]:
        return self._data.copy().reshape(-1, 4)

    @data.setter
    def data(self, data: np.ndarray[tuple[int, Literal[4]], np.dtype[np.uint8]]):
        self._data = data.copy().reshape(*self._data.shape)

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

    # ================================

    def touch(self, shape: BaseShape) -> list[tuple[Vector, ColorType]] | None:
        _, ny, nz = self._shape
        indices = shape.get_masked(self._data)
        items = self._data.reshape(-1, 4)[indices]
        keep = (items[:, 0] & 0x80) != 0
        indices, items = indices[keep], items[keep]
        if indices.size == 0:
            return None

        xyz = np.stack([indices // (ny * nz), (indices // nz) % ny, indices % nz], axis=1)
        colors = self._color_space.get_color(items[:, 1])
        return [(Vector(*map(int, point)), color) for point, color in zip(xyz, colors)]

    def aabb(self) -> tuple[Vector, Vector] | None:
        occupied = (self._data[..., 0] & 0x80) != 0
        if not occupied.any():
            return None

        low, high = [], []
        for axis in range(3):
            touched = np.flatnonzero(occupied.any(axis=tuple(other for other in range(3) if other != axis)))
            low.append(int(touched[0]))
            high.append(int(touched[-1]))
        return Vector(*low), Vector(*high)

    # ================================

    def save(self, path: Path) -> None:
        """
        magic | version u16 | size u16 x3 | mode u8 | count u32 | entries u16
        | ascii name + 4x 0x00 | payload
        """
        name = self._color_space_type
        if name is None:
            raise ValueError("voxel has no colour space name")
        try:
            encoded = name.encode("ascii")
        except UnicodeEncodeError:
            raise ValueError(f"colour space name must be ascii: {name!r}") from None

        nx, ny, nz = self._shape
        flat = self._data.reshape(-1, 4)
        count = int((flat[:, 0] & 0x80).astype(bool).sum())
        dense = count * 7 >= nx * ny * nz * 4

        if dense:
            payload = self._data.tobytes()
        else:
            index = np.flatnonzero(flat[:, 0] & 0x80)
            records = np.empty((count, 7), dtype=np.uint8)
            records[:, 0] = index // (ny * nz)
            records[:, 1] = (index // nz) % ny
            records[:, 2] = index % nz
            records[:, 3:] = flat[index]
            payload = records.tobytes()

        header = (
            self._MAGIC
            + struct.pack("<H", self._VERSION)
            + struct.pack("<HHH", nx, ny, nz)
            + struct.pack("<B", 1 if dense else 0)
            + struct.pack("<I", count)
            + struct.pack("<H", len(self._color_space))
        )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(header + encoded + self._NAME_TERMINATOR + payload)

    @classmethod
    def load(cls, path: Path) -> Self:
        blob = Path(path).read_bytes()
        if len(blob) < cls._HEADER_SIZE:
            raise ValueError(f"SSVOX file is truncated: {len(blob)} bytes")
        if blob[:len(cls._MAGIC)] != cls._MAGIC:
            raise ValueError(f"not a SSVOX file, magic is {blob[:len(cls._MAGIC)]!r}")

        version = struct.unpack_from("<H", blob, 5)[0]
        if version != cls._VERSION:
            raise ValueError(f"unsupported SSVOX version {version}")

        nx, ny, nz = struct.unpack_from("<HHH", blob, 7)
        if min(nx, ny, nz) < 1 or max(nx, ny, nz) > 256:
            raise ValueError(f"SSVOX size out of range: {nx} x {ny} x {nz}")

        mode = blob[13]
        if mode not in (0, 1):
            raise ValueError(f"unknown SSVOX storage mode {mode}")
        count = struct.unpack_from("<I", blob, 14)[0]
        total = nx * ny * nz
        if count > total:
            raise ValueError(f"SSVOX count {count} exceeds the {total} voxels of the grid")
        entries = struct.unpack_from("<H", blob, 18)[0]

        split = blob.find(cls._NAME_TERMINATOR, cls._HEADER_SIZE)
        if split < 0:
            raise ValueError("SSVOX colour space name has no terminator")
        if split == cls._HEADER_SIZE:
            raise ValueError("SSVOX colour space name is empty")
        try:
            name = blob[cls._HEADER_SIZE:split].decode("ascii")
        except UnicodeDecodeError:
            raise ValueError("SSVOX colour space name must be ascii") from None

        palette = get_palette(name)
        if len(palette) < entries:
            raise ValueError(
                f"colour space {name!r} now has {len(palette)} entries, "
                f"but the file was written with {entries}"
            )

        payload = blob[split + len(cls._NAME_TERMINATOR):]
        if mode == 1:
            expected = total * 4
            if len(payload) != expected:
                raise ValueError(f"dense payload size {len(payload)} != {expected}")
            data = np.frombuffer(payload, dtype=np.uint8).reshape(nx, ny, nz, 4)
        else:
            expected = count * 7
            if len(payload) != expected:
                raise ValueError(f"sparse payload size {len(payload)} != {expected}")
            records = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 7)
            x, y, z = (records[:, axis].astype(np.int64) for axis in range(3))
            if x.size and (x.max() >= nx or y.max() >= ny or z.max() >= nz):
                raise ValueError("sparse voxel coordinates are out of range")
            data = np.zeros((nx, ny, nz, 4), dtype=np.uint8)
            data.reshape(-1, 4)[x * (ny * nz) + y * nz + z] = records[:, 3:]

        if int((data[..., 0] & 0x80).astype(bool).sum()) != count:
            raise ValueError(f"SSVOX payload holds a different voxel count than the header's {count}")

        voxel = cls(Vector(nx, ny, nz), color_space=name)
        voxel.data = data.reshape(-1, 4)
        return voxel

    # ================================

    def clean(self) -> None:
        self._data.fill(0)

    def mirror(self, *, source: Literal["low", "high"], axis: Literal["YZ", "XZ", "XY"] = "YZ") -> None:
        axis_no = {"YZ": 0, "XZ": 1, "XY": 2}[axis]
        size = self._shape[axis_no]
        low = np.arange(size // 2)
        high = size - 1 - low
        src, dst = (low, high) if source == "low" else (high, low)

        src_sel, dst_sel = [slice(None)] * 4, [slice(None)] * 4
        src_sel[axis_no], dst_sel[axis_no] = src, dst
        self._data[tuple(dst_sel)] = self._data[tuple(src_sel)]

    def modify(self, shape: BaseShape, value: np.ndarray[tuple[Literal[4]], np.dtype[np.uint8]], *, strict: bool = True) -> str | None:
        if strict and shape.is_cross_border(self._data):
            return f"{type(shape).__name__} crosses the voxel border"

        self._data.reshape(-1, 4)[shape.get_masked(self._data)] = value
        return None

    def move(self, offset: Vector, *, strict: bool = True) -> str | None:
        box = self.aabb()
        if box is None:
            return None

        low, high = box[0].value, box[1].value
        shifts = offset.value
        if strict:
            for axis, shift in enumerate(shifts):
                if low[axis] + shift < 0 or high[axis] + shift >= self._shape[axis]:
                    return "moved voxels cross the voxel border"

        new = np.zeros_like(self._data)
        source_slices, target_slices = [], []
        for axis, shift in enumerate(shifts):
            low, high = max(0, -shift), min(self._shape[axis], self._shape[axis] - shift)
            if low >= high:
                self._data = new
                return None
            source_slices.append(slice(low, high))
            target_slices.append(slice(low + shift, high + shift))
        new[tuple(target_slices)] = self._data[tuple(source_slices)]
        self._data = new
        return None

    def rotate(self, axis: Vector, degrees: float, *, strict: bool = True) -> str | None:
        if degrees % 360 == 0:
            return None

        ax, ay, az = map(float, axis.value)
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm == 0:
            return "rotation axis must be non-zero"
        ax, ay, az = ax / norm, ay / norm, az / norm

        angle = math.radians(degrees)
        sine, cosine = math.sin(angle), math.cos(angle)
        complement = 1 - cosine
        rotation = np.array([
            [cosine + ax * ax * complement, ax * ay * complement - az * sine, ax * az * complement + ay * sine],
            [ay * ax * complement + az * sine, cosine + ay * ay * complement, ay * az * complement - ax * sine],
            [az * ax * complement - ay * sine, az * ay * complement + ax * sine, cosine + az * az * complement],
        ], dtype=np.float64)

        occupied = (self._data[..., 0] & 0x80) != 0
        if not occupied.any():
            return None

        nx, ny, nz = self._shape
        center = np.array([(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2], dtype=np.float64)

        if strict:
            for x0 in range(0, nx, 16):
                x1 = min(x0 + 16, nx)
                slice_occupied = occupied[x0:x1]
                if not slice_occupied.any():
                    continue
                ix, iy, iz = np.nonzero(slice_occupied)
                points = np.stack((ix + x0 - center[0], iy - center[1], iz - center[2]), axis=0)
                landed = np.floor(rotation @ points + center[:, None] + 0.5).astype(np.int64)
                if ((landed < 0) | (landed >= np.array(self._shape)[:, None])).any():
                    return "rotated voxels cross the voxel border"

        inverse = rotation.T
        ys = np.arange(ny, dtype=np.float64) - center[1]
        zs = np.arange(nz, dtype=np.float64) - center[2]
        new = np.zeros_like(self._data)
        for x0 in range(0, nx, 16):
            x1 = min(x0 + 16, nx)
            xs = np.arange(x0, x1, dtype=np.float64) - center[0]
            source_x = np.floor(inverse[0, 0] * xs[:, None, None] + inverse[0, 1] * ys[None, :, None] + inverse[0, 2] * zs[None, None, :] + center[0] + 0.5).astype(np.int64)
            source_y = np.floor(inverse[1, 0] * xs[:, None, None] + inverse[1, 1] * ys[None, :, None] + inverse[1, 2] * zs[None, None, :] + center[1] + 0.5).astype(np.int64)
            source_z = np.floor(inverse[2, 0] * xs[:, None, None] + inverse[2, 1] * ys[None, :, None] + inverse[2, 2] * zs[None, None, :] + center[2] + 0.5).astype(np.int64)
            inside = (0 <= source_x) & (source_x < nx) & (0 <= source_y) & (source_y < ny) & (0 <= source_z) & (source_z < nz)
            block = np.zeros((x1 - x0, ny, nz, 4), dtype=np.uint8)
            if inside.any():
                block[inside] = self._data[source_x[inside], source_y[inside], source_z[inside]]
            new[x0:x1] = block
        self._data = new
        return None

    def scale(self, factor: tuple[float, float, float], *, strict: bool = True) -> str | None:
        fx, fy, fz = factor
        if not all(math.isfinite(value) and value > 0 for value in (fx, fy, fz)):
            return "scale factors must be positive"

        box = self.aabb()
        if box is None:
            return None

        low, high = list(box[0].value), list(box[1].value)

        source_axes = []
        for axis, value in enumerate((fx, fy, fz)):
            extent = high[axis] - low[axis] + 1
            size = max(1, math.ceil(extent * value))
            if strict and low[axis] + size > self._shape[axis]:
                return "scaled voxels cross the voxel border"
            size = min(size, self._shape[axis] - low[axis])
            source_axes.append(low[axis] + np.floor(np.arange(size) / value).astype(np.int64))

        new = np.zeros_like(self._data)
        sx, sy, sz = source_axes
        new[low[0]:low[0] + sx.size, low[1]:low[1] + sy.size, low[2]:low[2] + sz.size] = self._data[np.ix_(sx, sy, sz)]
        self._data = new
        return None

    def merge(
            self,
            merged: Self,
            src_pos: Vector,
            dst_pos: Vector,
            *,
            mode: Literal["fill_all", "fill_empty"] = "fill_all",
            strict: bool = False,
            ) -> str | None:
        offset = Vector(*(dst - src for dst, src in zip(dst_pos.value, src_pos.value)))
        box = merged.aabb()

        if strict and box is not None:
            for axis, (shift, limit) in enumerate(zip(offset.value, self._shape)):
                if box[0].value[axis] + shift < 0 or box[1].value[axis] + shift >= limit:
                    return "merged voxels cross the voxel border"

        source_slices, target_slices = [], []
        for shift, self_size, merged_size in zip(offset.value, self._shape, merged._shape):
            low, high = max(0, -shift), min(merged_size, self_size - shift)
            if low >= high:
                return None
            source_slices.append(slice(low, high))
            target_slices.append(slice(low + shift, high + shift))

        source = merged._data[tuple(source_slices)]
        target = self._data[tuple(target_slices)]
        selected = (source[..., 0] & 0x80) != 0
        if mode == "fill_empty":
            selected &= (target[..., 0] & 0x80) == 0

        rows = source[selected]
        source_colors = merged._color_space.get_color(np.arange(len(merged._color_space)))
        target_colors = self._color_space.get_color(np.arange(len(self._color_space)))
        if not np.array_equal(source_colors, target_colors):
            lut = self._color_space.get_index(source_colors).astype(np.uint8)
            rows[:, 1] = lut[rows[:, 1]]
        target[selected] = rows
        return None
