"""Colour palettes, loaded from ``resources/colors`` as raw channel bytes.
Every ``*.bin`` file there is self describing: an ascii key, four 0x00 bytes, then the entries, four bytes each, R, G, B, A
"""

import threading
import numpy as np
from pathlib import Path
from typing import Literal

from surpassvoxel.structures.typing import ColorType, ColorsType


_palettes: dict[str, ColorsType] | None = None
_palette_lock = threading.Lock()


def _load_palette(path: Path) -> tuple[str, ColorsType]:
    PALETTE_TERMINATOR = b"\x00\x00\x00\x00"

    blob = path.read_bytes()

    split = blob.find(PALETTE_TERMINATOR)
    if split < 0:
        raise ValueError(f"palette file has no key terminator: {path}")
    if split == 0:
        raise ValueError(f"palette file has an empty key: {path}")

    raw = blob[split + len(PALETTE_TERMINATOR) :]
    if not raw:
        raise ValueError(f"palette file has no colour data: {path}")
    if len(raw) % 4:
        raise ValueError(f"palette data must be whole RGBA entries, {path} has {len(raw)} bytes")

    try:
        name = blob[:split].decode("ascii")
    except UnicodeDecodeError:
        raise ValueError(f"palette key must be ascii: {path}") from None

    palette = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4)
    palette.flags.writeable = False
    return name, palette


def get_palette(name: str) -> ColorsType:
    PALETTE_DIR = Path(__file__).resolve().parents[1] / "resources" / "colors"

    global _palettes
    if _palettes is None:
        with _palette_lock:
            if _palettes is None:
                if not PALETTE_DIR.is_dir():
                    raise ValueError(f"palette directory is missing: {PALETTE_DIR}")
                loaded: dict[str, np.ndarray[tuple[int, Literal[4]], np.dtype[np.uint8]]] = {}
                for path in sorted(PALETTE_DIR.glob("*.bin")):
                    key, palette = _load_palette(path)
                    if key in loaded:
                        raise ValueError(f"duplicate palette key {key!r} in {path}")
                    loaded[key] = palette
                _palettes = loaded

    try:
        return _palettes[name].copy()
    except KeyError:
        raise ValueError(f"unknown palette {name!r}, available: {sorted(_palettes)}") from None


class ColorSpace:
    
    def __init__(
            self,
            color_space_id: str,
            *,
            mask: tuple[bool, bool, bool, bool] | None = None,
            ) -> None:
        palette = get_palette(color_space_id)
        
        if mask is None:
            mask = (True, True, True, True)
        if not any(mask):
            raise ValueError("at least one channel must be valid")

        self._palette = palette
        self._mask = np.array(mask, dtype=bool)
        self._masked_palette = palette[:, self._mask].astype(np.int32)
        self._name = color_space_id

    def __len__(self) -> int:
        return self._palette.shape[0]

    @property
    def mask(self) -> tuple[bool, bool, bool, bool]:
        return tuple(self._mask.tolist())

    @property
    def name(self) -> str:
        return self._name

    def get_color(
            self,
            index: int | np.ndarray[tuple[int], np.dtype[int]],
            ) -> ColorType | ColorsType:
        
        if isinstance(index, np.ndarray):
            return self._palette[index].copy()
        return self._palette[index].copy()

    def get_index(
            self,
            rgba: ColorType | ColorsType
            ) -> int | np.ndarray[tuple[int], np.dtype[int]]:
        
        if rgba.ndim == 2:
            querys = rgba.astype(np.int32)[:, self._mask]
            diff = self._masked_palette[None] - querys[:, None]
            return (diff ** 2).sum(axis=2).argmin(axis=1)

        query = rgba.astype(np.int32)[self._mask]
        diff = self._masked_palette - query
        return int((diff ** 2).sum(axis=1).argmin())
