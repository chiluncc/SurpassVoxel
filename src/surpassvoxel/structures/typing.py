import numpy as np
from typing import Literal


# color
type PaletteType = Literal["VGA_16", "MAGICVOXEL_255"]
type ColorType = np.ndarray[tuple[Literal[4]], np.dtype[np.uint8]]
type ColorsType = np.ndarray[tuple[int, Literal[4]], np.dtype[np.uint8]]


# voxel
type VoxelItemType = np.ndarray[tuple[Literal[4]], np.dtype[np.uint8]]
type VoxelItemsType = np.ndarray[tuple[int, int, int, Literal[4]], np.dtype[np.uint8]]


# grid
type GridBool = np.ndarray[tuple[int, int, int], np.dtype[np.bool_]]


# vector
type Vector2Df = np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]
type Vector2Dfs = np.ndarray[tuple[int, Literal[2]], np.dtype[np.float64]]
type Vector3Df = np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]
type Vector3Dfs = np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]]
type Vector2Di = np.ndarray[tuple[Literal[2]], np.dtype[np.int32]]
type Vector2Dis = np.ndarray[tuple[int, Literal[2]], np.dtype[np.int32]]
type Vector3Di = np.ndarray[tuple[Literal[3]], np.dtype[np.int32]]
type Vector3Dis = np.ndarray[tuple[int, Literal[3]], np.dtype[np.int32]]