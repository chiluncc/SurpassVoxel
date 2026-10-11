import struct
import numpy as np
from pathlib import Path
from typing import Self

from surpassvoxel.structures.color import ColorSpace, get_palette
from surpassvoxel.structures.typing import PaletteType, Vector3Di, VoxelItemsType


class Voxel:
    """
    Coordinates are right handed, +x right, +y up, +z out of the screen, origin at
    the (0, 0, 0) corner, item (x, y, z) with 0 <= x < x_size and so on.

    Item layout, four bytes per voxel::

    byte0 bit7 (0x80) : voxel exist, 1 = true, 0 = false
    byte0 bit6 (0x40) : voxel assigned material, 1 = true, 0 = false
    byte0 bit5 (0x20) : voxel already grouped, 1 = true, 0 = false
    byte1 bit7..bit0 : color index
    byte2 bit7..bit0 : material index
    byte3 bit7..bit0 : group index

    An empty voxel is four 0x00 bytes. Read as a little endian uint32, an
    occupied voxel with color index c is 0x80 | (c << 8).
    """

    _MAGIC = b"SSVOX"
    _VERSION = 2
    _HEADER_SIZE = 21
    _NAME_TERMINATOR = b"\x00\x00\x00\x00"

    def __init__(
            self,
            shape: tuple[int, int, int],
            *,
            color_space_id: PaletteType = "MAGICVOXEL_255",
            color_mask: tuple[bool, bool, bool, bool],
            ) -> None:
        
        if max(shape) > 256:
            raise ValueError("Too large in size")

        self._shape = shape
        self._data = np.zeros((*shape, 4), dtype=np.uint8)
        self._color_space = ColorSpace(color_space_id, mask=color_mask)

    @property
    def color_space(self) -> ColorSpace:
        return self._color_space

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def voxel_items(self) -> VoxelItemsType:
        return self._data.copy()

    @voxel_items.setter
    def voxel_items(self, data: VoxelItemsType) -> None:
        self._data = np.array(data, dtype=np.uint8)

    def convert_color_space(self, color_space: ColorSpace) -> Self:
        source = self._color_space.get_color(np.arange(len(self._color_space)))
        lut = color_space.get_index(source).astype(np.uint8)

        new = self.clone()
        occupied = (new._data[..., 0] & 0x80) != 0
        new._data[occupied, 1] = lut[new._data[occupied, 1]]
        new._color_space = color_space
        return new

    def clean(self) -> None:
        self._data.fill(0)

    def clone(self) -> Self:
        new = type(self).__new__(type(self))
        new._shape = self._shape
        new._color_space = self._color_space
        new._data = self._data.copy()
        return new

    def shape_like(self, *, color_space_id: PaletteType | None = None) -> Self:
        new = type(self).__new__(type(self))
        new._shape = self._shape
        new._data = np.zeros((*self._shape, 4), dtype=np.uint8)
        if color_space_id is None:
            new._color_space = self._color_space
        else:
            new._color_space = ColorSpace(color_space_id)
        return new

    def aabb(self) -> tuple[Vector3Di, Vector3Di] | None:
        occupied = (self._data[..., 0] & 0x80) != 0
        if not occupied.any():
            return None

        low, high = [], []
        for axis in range(3):
            touched = np.flatnonzero(occupied.any(axis=tuple(other for other in range(3) if other != axis)))
            low.append(int(touched[0]))
            high.append(int(touched[-1]))
        return np.array(low, dtype=np.int32), np.array(high, dtype=np.int32)

    def save(self, path: Path) -> None:
        """
        magic | version u16 | size u16 x3 | mode u8 | count u32 | entries u16 | mask u8 | ascii name + 4x 0x00 | payload
        """
        name = self._color_space.name
        try:
            encoded = name.encode("ascii")
        except UnicodeEncodeError:
            raise ValueError(f"colour space name must be ascii: {name!r}") from None

        nx, ny, nz = self._shape
        occupied = (self._data[..., 0] & 0x80) != 0
        count = int(occupied.sum())
        dense = count * 7 >= nx * ny * nz * 4

        if dense:
            payload = self._data.tobytes()
        else:
            xyz = np.argwhere(occupied)
            records = np.empty((count, 7), dtype=np.uint8)
            records[:, :3] = xyz
            records[:, 3:] = self._data[occupied]
            payload = records.tobytes()

        header = (
            self._MAGIC
            + struct.pack("<H", self._VERSION)
            + struct.pack("<HHH", nx, ny, nz)
            + struct.pack("<B", 1 if dense else 0)
            + struct.pack("<I", count)
            + struct.pack("<H", len(self._color_space))
            + struct.pack("<B", sum(1 << index for index, valid in enumerate(self._color_space.mask) if valid))
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
        mask = blob[20]
        color_mask = tuple(bool(mask >> index & 1) for index in range(4))
        if not any(color_mask):
            raise ValueError("SSVOX colour space mask has no valid channel")

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
            data[x, y, z] = records[:, 3:]

        if int((data[..., 0] & 0x80).astype(bool).sum()) != count:
            raise ValueError(f"SSVOX payload holds a different voxel count than the header's {count}")

        voxel = cls((nx, ny, nz), color_space_id=name, color_mask=color_mask)
        voxel.voxel_items = data
        return voxel
