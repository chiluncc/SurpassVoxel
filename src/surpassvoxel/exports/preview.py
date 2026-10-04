import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from surpassvoxel.structures.voxel import Voxel


def _trace(
        origin: np.ndarray,
        direction: np.ndarray,
        occupied: np.ndarray,
        mask: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = np.array(occupied.shape, dtype=np.float64)
    limits = np.array(occupied.shape, dtype=np.int64)
    count = origin.shape[0]

    parallel = direction == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 / direction
        low = (0.0 - origin) * inverse
        high = (shape - origin) * inverse
    inside = parallel & (origin >= 0.0) & (origin <= shape)
    outside = parallel & ~inside
    low = np.where(inside, -np.inf, np.where(outside, np.inf, low))
    high = np.where(inside, np.inf, np.where(outside, -np.inf, high))
    t_low = np.minimum(low, high)
    t_high = np.maximum(low, high)
    enter = np.max(t_low, axis=1)
    leave = np.min(t_high, axis=1)
    active = (enter <= leave) & (leave > 0.0)
    enter = np.maximum(enter, 0.0) + 1e-9

    step = np.sign(direction).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(step != 0, np.abs(1.0 / np.where(direction == 0.0, 1.0, direction)), np.inf)
        cell = np.floor(origin + enter[:, None] * direction).astype(np.int64)
        plane = np.where(step > 0, cell + 1.0, cell.astype(np.float64))
        t_max = (plane - origin) / np.where(direction == 0.0, 1.0, direction)
    t_max = np.where(step != 0, t_max, np.inf)
    t = enter
    rows = np.arange(count)
    entry = np.argmax(t_low, axis=1)
    normal = np.zeros((count, 3), dtype=np.int64)
    normal[rows, entry] = -step[rows, entry]

    hit = np.zeros(count, dtype=bool)
    hit_cell = np.zeros((count, 3), dtype=np.int64)
    hit_normal = np.zeros((count, 3), dtype=np.int64)
    hit_t = np.zeros(count, dtype=np.float64)

    for _ in range(int(limits.sum()) + 3):
        if not active.any():
            break
        index = np.flatnonzero(active)
        cells = cell[index]
        inside_grid = np.all((cells >= 0) & (cells < limits), axis=1)
        if inside_grid.any():
            live = index[inside_grid]
            points = cells[inside_grid]
            probe = occupied[points[:, 0], points[:, 1], points[:, 2]]
            if mask is not None:
                probe = probe & mask[points[:, 0], points[:, 1], points[:, 2]]
            found = live[probe]
            if found.size:
                hit[found] = True
                hit_cell[found] = cell[found]
                hit_normal[found] = normal[found]
                hit_t[found] = t[found]
                active[found] = False

        index = np.flatnonzero(active)
        if index.size == 0:
            break
        rows = np.arange(index.size)
        axis = np.argmin(t_max[index], axis=1)
        walked = step[index[rows], axis]
        cell[index[rows], axis] += walked
        normal[index[rows]] = 0
        normal[index[rows], axis] = -walked
        t[index[rows]] = t_max[index[rows], axis]
        t_max[index[rows], axis] += delta[index[rows], axis]
        active[index[t[index] > leave[index]]] = False

    return hit, hit_cell, hit_normal, hit_t


def _render_ortho(
        voxel: Voxel,
        view: str,
        window: int | tuple[int, int] | None,
        size: int | None,
        depth_fade: float,
        path: str | Path | None,
        ) -> Image.Image:
    right, up, depth = {
        "front": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        "top": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
        "right": ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    }[view]
    right, up, depth = (np.array(vector, dtype=np.float64) for vector in (right, up, depth))
    horizontal = int(np.argmax(np.abs(right)))
    vertical = int(np.argmax(np.abs(up)))
    deep = int(np.argmax(np.abs(depth)))

    layers = voxel.data.reshape(*voxel.shape, 4)
    occupied = (layers[..., 0] & 0x80) != 0
    index = layers[..., 1].astype(np.int64)
    colors = np.asarray(voxel.color_space.get_color(np.arange(len(voxel.color_space))), dtype=np.float64)[:, :3]
    palette = np.zeros((256, 3), dtype=np.float64)
    palette[: len(colors)] = colors
    palette[len(colors):] = colors[-1]

    mask = None
    if window is not None:
        bounds = (int(window), int(window)) if isinstance(window, (int, np.integer)) else tuple(int(v) for v in window)
        if len(bounds) != 2:
            raise ValueError("slice must be an int or a (start, stop) pair")
        mask = np.zeros(voxel.shape, dtype=bool)
        low, high = max(0, bounds[0]), min(voxel.shape[deep] - 1, bounds[1])
        if low <= high:
            band = [slice(None)] * 3
            band[deep] = slice(low, high + 1)
            mask[tuple(band)] = True
    drawn = occupied if mask is None else occupied & mask

    if drawn.any():
        centers = np.argwhere(drawn).astype(np.float64) + 0.5
        across, along = centers @ right, centers @ up
        u0, u1 = float(across.min()) - 0.5, float(across.max()) + 0.5
        v0, v1 = float(along.min()) - 0.5, float(along.max()) + 0.5
    else:
        u0, u1 = sorted((right[horizontal] * 0.5, right[horizontal] * (voxel.shape[horizontal] - 0.5)))
        v0, v1 = sorted((up[vertical] * 0.5, up[vertical] * (voxel.shape[vertical] - 0.5)))
    u_center, v_center = (u0 + u1) / 2.0, (v0 + v1) / 2.0
    resolution = int(size) if size is not None else (1024 if max(u1 - u0, v1 - v0) > 128 else 512)
    scale = max(1e-6, (resolution - 2.0 * max(18.0, resolution * 0.05)) / max(u1 - u0, v1 - v0))

    far = float(max(voxel.shape)) + 1.0
    canvas = np.zeros((resolution, resolution, 3), dtype=np.float64)
    canvas[:, :] = (18, 18, 22)
    for y0 in range(0, resolution, 128):
        y1 = min(y0 + 128, resolution)
        rows, columns = np.mgrid[y0:y1, 0:resolution]
        across = u_center + ((columns + 0.5) - resolution / 2.0) / scale
        along = v_center - ((rows + 0.5) - resolution / 2.0) / scale
        origins = across[..., None] * right + along[..., None] * up + far * depth
        rays = np.empty_like(origins)
        rays[:] = -depth
        hit, cell, _, distance = _trace(origins.reshape(-1, 3), rays.reshape(-1, 3), occupied, mask)
        if not hit.any():
            continue
        depth_low, depth_high = float(distance[hit].min()), float(distance[hit].max())
        level = np.ones(int(hit.sum()), dtype=np.float64)
        if depth_high - depth_low > 1e-9:
            level = 1.0 - depth_fade * (distance[hit] - depth_low) / (depth_high - depth_low)
        canvas[y0:y1][hit.reshape(y1 - y0, resolution)] = (
            palette[index[cell[hit][:, 0], cell[hit][:, 1], cell[hit][:, 2]]] * level[:, None]
        )

    image = Image.fromarray(canvas.clip(0, 255).astype(np.uint8))
    fine = 1 if scale >= 10 else 2 if scale >= 5 else 5 if scale >= 3 else 10
    grid = np.asarray(image, dtype=np.float64)
    for step, alpha, color in ((fine, 0.30, (92, 92, 104)), (10, 0.55, (150, 150, 162))):
        color = np.array(color, dtype=np.float64)
        for boundary in range(0, voxel.shape[horizontal] + 1, step):
            if step != 10 and boundary % 10 == 0:
                continue
            column = int(round(resolution / 2.0 + (right[horizontal] * boundary - u_center) * scale))
            if 0 <= column < resolution:
                grid[:, column, :] = grid[:, column, :] * (1.0 - alpha) + color * alpha
        for boundary in range(0, voxel.shape[vertical] + 1, step):
            if step != 10 and boundary % 10 == 0:
                continue
            row = int(round(resolution / 2.0 - (up[vertical] * boundary - v_center) * scale))
            if 0 <= row < resolution:
                grid[row, :, :] = grid[row, :, :] * (1.0 - alpha) + color * alpha

    image = Image.fromarray(grid.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(10, resolution // 64))
    for boundary in range(0, voxel.shape[horizontal] + 1, 10):
        column = int(round(resolution / 2.0 + (right[horizontal] * boundary - u_center) * scale))
        if 2 <= column < resolution - 4:
            draw.text((column + 2, 3), str(boundary), fill=(216, 216, 224), font=font)
    for boundary in range(0, voxel.shape[vertical] + 1, 10):
        row = int(round(resolution / 2.0 - (up[vertical] * boundary - v_center) * scale))
        if 2 <= row < resolution - 4:
            draw.text((3, row + 2), str(boundary), fill=(216, 216, 224), font=font)

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
    return image


# ================================


def render_perspective(
        voxel: Voxel,
        position: tuple[float, float, float],
        *,
        fov: float = 60.0,
        size: int | None = None,
        ambient: float = 0.40,
        sun: float = 0.70,
        path: str | Path | None = None,
        ) -> Image.Image:
    layers = voxel.data.reshape(*voxel.shape, 4)
    occupied = (layers[..., 0] & 0x80) != 0
    index = layers[..., 1].astype(np.int64)
    colors = np.asarray(voxel.color_space.get_color(np.arange(len(voxel.color_space))), dtype=np.float64)[:, :3]
    palette = np.zeros((256, 3), dtype=np.float64)
    palette[: len(colors)] = colors
    palette[len(colors):] = colors[-1]

    span = 0
    if occupied.any():
        marked = np.argwhere(occupied)
        span = int((marked.max(axis=0) - marked.min(axis=0) + 1).max())
    resolution = int(size) if size is not None else (1024 if span > 128 else 512)

    eye = tuple(position)
    if len(eye) != 3:
        raise ValueError("position must be three numbers")
    eye = np.array(eye, dtype=np.float64)
    box = voxel.aabb()
    if box is None:
        target = (np.array(voxel.shape, dtype=np.float64) - 1.0) / 2.0
    else:
        target = (np.array(box[0].value, dtype=np.float64) + np.array(box[1].value, dtype=np.float64) + 1.0) / 2.0

    forward = target - eye
    distance = float(np.linalg.norm(forward))
    forward = np.array([0.0, 0.0, -1.0]) if distance < 1e-9 else forward / distance
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.cross(forward, np.array([0.0, 0.0, -1.0]))
    right = right / float(np.linalg.norm(right))
    up = np.cross(right, forward)
    light = np.array([0.5, 0.7, 0.5])
    light = light / float(np.linalg.norm(light))
    light = light[0] * right + light[1] * up - light[2] * forward

    tangent = math.tan(math.radians(fov) / 2.0)
    canvas = np.zeros((resolution, resolution, 3), dtype=np.float64)
    canvas[:, :] = (18, 18, 22)
    for y0 in range(0, resolution, 128):
        y1 = min(y0 + 128, resolution)
        rows, columns = np.mgrid[y0:y1, 0:resolution]
        ndc_x = (columns + 0.5) / resolution * 2.0 - 1.0
        ndc_y = 1.0 - (rows + 0.5) / resolution * 2.0
        rays = forward + ndc_x[..., None] * tangent * right + ndc_y[..., None] * tangent * up
        rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        origins = np.empty_like(rays)
        origins[:] = eye
        hit, cell, normal, _ = _trace(origins.reshape(-1, 3), rays.reshape(-1, 3), occupied)
        if not hit.any():
            continue
        level = np.clip(ambient + sun * np.clip(normal[hit].astype(np.float64) @ light, 0.0, None), 0.0, 1.0)
        canvas[y0:y1][hit.reshape(y1 - y0, resolution)] = (
            palette[index[cell[hit][:, 0], cell[hit][:, 1], cell[hit][:, 2]]] * level[:, None]
        )

    image = Image.fromarray(canvas.clip(0, 255).astype(np.uint8))
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
    return image


def render_front(
        voxel: Voxel,
        *,
        slice: int | tuple[int, int] | None = None,
        size: int | None = None,
        depth_fade: float = 0.28,
        path: str | Path | None = None,
        ) -> Image.Image:
    return _render_ortho(voxel, "front", slice, size, depth_fade, path)


def render_top(
        voxel: Voxel,
        *,
        slice: int | tuple[int, int] | None = None,
        size: int | None = None,
        depth_fade: float = 0.28,
        path: str | Path | None = None,
        ) -> Image.Image:
    return _render_ortho(voxel, "top", slice, size, depth_fade, path)


def render_right(
        voxel: Voxel,
        *,
        slice: int | tuple[int, int] | None = None,
        size: int | None = None,
        depth_fade: float = 0.28,
        path: str | Path | None = None,
        ) -> Image.Image:
    return _render_ortho(voxel, "right", slice, size, depth_fade, path)
