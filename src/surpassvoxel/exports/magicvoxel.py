import struct
import numpy as np
from pathlib import Path

from surpassvoxel.structures.voxel import Voxel
from surpassvoxel.structures.color import ColorSpace


def transform_to_magicvoxel(data: Voxel, path: Path) -> None:
    fixed = (Path(__file__).resolve().parents[1] / "resources" / "exports" / "magicvoxel_fixed.bin").read_bytes()

    colors = data.color_space.get_color(np.arange(len(data.color_space)))
    if len(colors) > 255:
        data = data.convert_color_space(ColorSpace("MAGICVOXEL_255"))
        colors = data.color_space.get_color(np.arange(len(data.color_space)))

    palette = np.zeros((256, 4), dtype=np.uint8)
    palette[:len(colors)] = colors

    nx, ny, nz = data.shape
    items = data.voxel_items
    occupied = (items[..., 0] & 0x80) != 0
    xyz = np.argwhere(occupied)
    count = xyz.shape[0]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    order = np.lexsort((x, z, y))

    entries = np.empty((count, 4), dtype=np.uint8)
    entries[:, 0] = x[order]
    entries[:, 1] = z[order]
    entries[:, 2] = y[order]
    entries[:, 3] = items[..., 1][occupied][order] + 1

    size = b"SIZE" + struct.pack("<I", 12) + struct.pack("<I", 0) + struct.pack("<III", nx, nz, ny)
    xyzi = (
        b"XYZI" + struct.pack("<I", 4 + 4 * count) + struct.pack("<I", 0)
        + struct.pack("<I", count) + entries.tobytes()
    )
    blob = (
        b"VOX " + struct.pack("<I", 200)
        + b"MAIN" + struct.pack("<I", 0) + struct.pack("<I", len(size) + len(xyzi) + len(fixed))
        + size + xyzi
        + fixed[:944] + palette.tobytes() + fixed[1968:]
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
