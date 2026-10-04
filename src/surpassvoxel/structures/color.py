"""Colour palettes, loaded from ``resources/colors`` as raw channel bytes.
Every ``*.bin`` file there is self describing: an ascii key, four 0x00 bytes, then the entries, four bytes each, R, G, B, A
"""

import threading
import numpy as np
from pathlib import Path
from typing import Literal


_palettes: dict[str, np.ndarray[tuple[int, Literal[4]], np.dtype[np.uint8]]] | None = None
_palette_lock = threading.Lock()


def _load_palette(path: Path) -> tuple[str, np.ndarray[tuple[int, Literal[4]], np.dtype[np.uint8]]]:
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


type PaletteType = Literal["VGA_16", "MAGICVOXEL_255"]


def get_palette(name: str) -> np.ndarray[tuple[int, Literal[4]], np.dtype[np.uint8]]:
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
        return _palettes[name]
    except KeyError:
        raise ValueError(f"unknown palette {name!r}, available: {sorted(_palettes)}") from None


type ColorType = np.ndarray[Literal[4], np.dtype[np.uint8]]
type ColorTypes = np.ndarray[tuple[int, Literal[4]], np.dtype[np.uint8]]


class ColorSpace:
    
    def __init__(
            self,
            palette: np.ndarray[tuple[int, Literal[4]], np.dtype[np.uint8]],
            *,
            name: PaletteType | None = None,
            mask: tuple[bool, ...] | None = None,
            ):
        palette = np.array(palette)
        if palette.ndim != 2 or palette.shape[1] != 4:
            raise ValueError(f"palette must have shape (entries, 4), got {palette.shape}")
        if palette.shape[0] == 0:
            raise ValueError("palette is empty")

        if mask is None:
            mask = (True, True, True, True)
        else:
            mask = (*(mask), True, True, True, True)[:4]
        if not any(mask):
            raise ValueError("at least one channel must be valid")

        self._palette = palette
        self._mask = np.array(mask, dtype=bool)
        self._masked_palette = palette[:, self._mask].astype(np.int32)
        self._name = name

    def __len__(self) -> int:
        return self._palette.shape[0]

    @property
    def mask(self) -> tuple[bool, bool, bool, bool]:
        return tuple(self._mask.tolist())

    @property
    def name(self) -> PaletteType | None:
        return self._name

    def get_color(
            self,
            index: int | np.ndarray[tuple[int], np.dtype[int]],
            ) -> ColorType | ColorTypes | None:
        if isinstance(index, np.ndarray):
            if index.size and (index.min() < 0 or index.max() >= len(self)):
                return None
            return self._palette[index]
        if not 0 <= index < len(self):
            return None
        return self._palette[index].copy()

    def get_index(self, rgba: ColorType | ColorTypes) -> int | np.ndarray[tuple[int], np.dtype[int]]:
        if rgba.ndim == 2:
            colors = rgba.astype(np.int32)[:, self._mask]
            diff = self._masked_palette[None] - colors[:, None]
            return (diff ** 2).sum(axis=2).argmin(axis=1)

        query = rgba.reshape(-1).astype(np.int32)
        if query.size != 4:
            raise ValueError(f"expected four channels, got {query.size}")

        diff = self._masked_palette - query[self._mask]
        return int((diff ** 2).sum(axis=1).argmin())
