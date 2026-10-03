import struct
import numpy as np
from pathlib import Path

from surpassvoxel.structures.voxel import Voxel
from surpassvoxel.structures.color import ColorSpace, get_palette


def transform_to_magicvoxel(data: Voxel, path: Path) -> None:
    fixed = (Path(__file__).resolve().parents[1] / "resources" / "exports" / "magicvoxel_fixed.bin").read_bytes()

    colors = get_palette(data.color_space_type)
    if len(colors) > 255:
        data = data.change_color_space(ColorSpace(get_palette("MAGICVOXEL_255"), name="MAGICVOXEL_255"))
        colors = get_palette(data.color_space_type)

    palette = np.zeros((256, 4), dtype=np.uint8)
    palette[:len(colors)] = colors

    nx, ny, nz = data.shape[:3]
    flat = data.data
    idx = np.flatnonzero(flat[:, 0] & 0x80)

    x = idx // (ny * nz)
    y = (idx // nz) % ny
    z = idx % nz
    order = np.lexsort((x, z, y))

    entries = np.empty((idx.size, 4), dtype=np.uint8)
    entries[:, 0] = x[order]
    entries[:, 1] = z[order]
    entries[:, 2] = y[order]
    entries[:, 3] = flat[idx[order], 1] + 1

    size = b"SIZE" + struct.pack("<I", 12) + struct.pack("<I", 0) + struct.pack("<III", nx, nz, ny)
    xyzi = (
        b"XYZI" + struct.pack("<I", 4 + 4 * idx.size) + struct.pack("<I", 0)
        + struct.pack("<I", idx.size) + entries.tobytes()
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
