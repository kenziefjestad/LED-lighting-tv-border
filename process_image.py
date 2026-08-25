# Process Image for LED Lighting TV Border

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

DEFAULT_LAYOUT = {"top": 20, "left": 12, "right": 12, "bottom": 20}

# Track the previous frame's output so automatic temporal smoothing can work
# across sequential calls when the same script is used to process a video stream.
_LAST_FRAME_COLORS: Dict[str, List[Tuple[float, float, float]]] | None = None


def _rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    rgb = arr.astype(np.float32) / 255.0
    hsv = np.zeros_like(rgb, dtype=np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    max_channel = np.maximum(np.maximum(r, g), b)
    min_channel = np.minimum(np.minimum(r, g), b)
    delta = max_channel - min_channel

    hue = np.zeros_like(max_channel)
    mask = delta > 1e-6
    r_mask = mask & (max_channel == r)
    g_mask = mask & (max_channel == g)
    b_mask = mask & (max_channel == b)

    hue[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0
    hue[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0
    hue[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0
    hue *= 60.0

    saturation = np.zeros_like(max_channel)
    non_zero = max_channel > 1e-6
    saturation[non_zero] = delta[non_zero] / max_channel[non_zero]

    value = max_channel
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = saturation
    hsv[:, :, 2] = value
    return hsv


def _detect_content_bounds(image: np.ndarray) -> Tuple[int, int, int, int]:
    """Ignore black bars (letterboxing/pillarboxing) when sampling the edge zones."""
    grayscale = np.mean(image, axis=2)
    mask = grayscale > 16
    if not np.any(mask):
        height, width = image.shape[:2]
        return 0, 0, width, height

    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def _sample_region_mean(rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> Tuple[float, float, float]:
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return (0.0, 0.0, 0.0)

    hsv = _rgb_to_hsv(crop)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    valid_mask = (value >= 0.08) & (saturation >= 0.07)

    if valid_mask.any():
        sampled = crop[valid_mask]
    else:
        sampled = crop.reshape(-1, 3)

    average = sampled.mean(axis=0)
    return tuple(float(channel) for channel in average)


def _apply_gamma(rgb_triplet: Sequence[float], gamma: float = 2.2) -> Tuple[float, float, float]:
    corrected = np.clip(np.power(np.clip(np.asarray(rgb_triplet, dtype=np.float32), 0.0, 1.0), 1.0 / gamma), 0.0, 1.0)
    return tuple(float(channel) for channel in corrected)


def _smooth_with_previous(current: Sequence[float], previous: Sequence[float] | None, alpha: float = 0.35) -> Tuple[float, float, float]:
    if previous is None:
        return tuple(float(channel) for channel in current)
    prev = np.asarray(previous, dtype=np.float32)
    curr = np.asarray(current, dtype=np.float32)
    blended = alpha * curr + (1.0 - alpha) * prev
    return tuple(float(channel) for channel in blended)


def _build_zone_layout(width: int, height: int, layout: Dict[str, int]) -> Dict[str, List[Tuple[int, int, int, int]]]:
    x0, y0, x1, y1 = 0, 0, width, height
    inset = 0.08
    top_band = max(1, int(height * inset))
    bottom_band = max(1, int(height * inset))
    left_band = max(1, int(width * inset))
    right_band = max(1, int(width * inset))

    zones: Dict[str, List[Tuple[int, int, int, int]]] = {"top": [], "left": [], "right": [], "bottom": []}

    top_count = max(1, layout.get("top", DEFAULT_LAYOUT["top"]))
    left_count = max(1, layout.get("left", DEFAULT_LAYOUT["left"]))
    right_count = max(1, layout.get("right", DEFAULT_LAYOUT["right"]))
    bottom_count = max(1, layout.get("bottom", DEFAULT_LAYOUT["bottom"]))

    for idx in range(top_count):
        zone_x0 = x0 + idx * width // top_count
        zone_x1 = x0 + (idx + 1) * width // top_count
        zones["top"].append((zone_x0, y0, zone_x1, y0 + top_band))

    for idx in range(left_count):
        zone_y0 = y0 + idx * (height - top_band - bottom_band) // left_count
        zone_y1 = y0 + (idx + 1) * (height - top_band - bottom_band) // left_count
        zones["left"].append((x0, zone_y0, x0 + left_band, zone_y1))

    for idx in range(right_count):
        zone_y0 = y0 + idx * (height - top_band - bottom_band) // right_count
        zone_y1 = y0 + (idx + 1) * (height - top_band - bottom_band) // right_count
        zones["right"].append((x1 - right_band, zone_y0, x1, zone_y1))

    for idx in range(bottom_count):
        zone_x0 = x0 + idx * width // bottom_count
        zone_x1 = x0 + (idx + 1) * width // bottom_count
        zones["bottom"].append((zone_x0, y1 - bottom_band, zone_x1, y1))

    return zones


def _compute_zone_colors(image: np.ndarray, layout: Dict[str, int]) -> Dict[str, List[Tuple[float, float, float]]]:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = _detect_content_bounds(image)
    content_width = max(1, x1 - x0)
    content_height = max(1, y1 - y0)

    if x0 > 0 or y0 > 0 or x1 < width or y1 < height:
        crop = image[y0:y1, x0:x1]
    else:
        crop = image

    full_w, full_h = crop.shape[1], crop.shape[0]
    zones = _build_zone_layout(full_w, full_h, layout)

    result: Dict[str, List[Tuple[float, float, float]]] = {"top": [], "left": [], "right": [], "bottom": []}
    for key in ("top", "left", "right", "bottom"):
        for zone in zones[key]:
            zone_x0, zone_y0, zone_x1, zone_y1 = zone
            zone_color = _sample_region_mean(crop, zone_x0, zone_y0, zone_x1, zone_y1)
            zone_color = _apply_gamma(zone_color)
            result[key].append(zone_color)

    return result


def _smooth_zone_colors(zone_colors: Dict[str, List[Tuple[float, float, float]]]) -> Dict[str, List[Tuple[float, float, float]]]:
    global _LAST_FRAME_COLORS
    smoothed: Dict[str, List[Tuple[float, float, float]]] = {"top": [], "left": [], "right": [], "bottom": []}

    for key in ("top", "left", "right", "bottom"):
        prev_key = _LAST_FRAME_COLORS.get(key, []) if _LAST_FRAME_COLORS else []
        for idx, rgb in enumerate(zone_colors[key]):
            previous = prev_key[idx] if idx < len(prev_key) else None
            smoothed[key].append(_smooth_with_previous(rgb, previous, alpha=0.35))

    _LAST_FRAME_COLORS = smoothed
    return smoothed


def _save_zone_preview(output_path: Path, image: np.ndarray, zone_colors: Dict[str, List[Tuple[float, float, float]]], layout: Dict[str, int]) -> None:
    preview = np.array(image, copy=True)
    h, w = preview.shape[:2]
    x0, y0, x1, y1 = _detect_content_bounds(preview)
    content = preview[y0:y1, x0:x1]
    full_w, full_h = content.shape[1], content.shape[0]
    zones = _build_zone_layout(full_w, full_h, layout)

    for key in ("top", "left", "right", "bottom"):
        for idx, zone in enumerate(zones[key]):
            zone_x0, zone_y0, zone_x1, zone_y1 = zone
            zone_x0 += x0
            zone_y0 += y0
            zone_x1 += x0
            zone_y1 += y0
            rgb = np.asarray(zone_colors[key][idx], dtype=np.float32) * 255.0
            preview[zone_y0:zone_y1, zone_x0:zone_x1] = rgb

    out_img = Image.fromarray(preview.astype(np.uint8), mode="RGB")
    out_img.save(output_path)


def process_image(image_path, output_path):
    """Process an image frame into LED border colors and save a preview image.

    The function performs:
      1. black-bar detection and content-bound extraction,
      2. top/left/right/bottom zone mapping,
      3. average-dominant color extraction with HSV filtering,
      4. temporal smoothing for flicker reduction,
      5. gamma correction for LED brightness response.
    """
    source = Path(image_path)
    destination = Path(output_path)

    image = Image.open(source).convert("RGB")
    rgb_array = np.asarray(image)
    zone_colors = _compute_zone_colors(rgb_array, DEFAULT_LAYOUT)
    zone_colors = _smooth_zone_colors(zone_colors)

    output_dir = destination.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_zone_preview(destination, rgb_array, zone_colors, DEFAULT_LAYOUT)

    return {
        "layout": DEFAULT_LAYOUT,
        "zones": zone_colors,
        "output_path": str(destination),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image for LED border color sampling.")
    parser.add_argument("image_path", help="Input image to process")
    parser.add_argument("output_path", help="Output preview image path")
    args = parser.parse_args()

    result = process_image(args.image_path, args.output_path)
    print(f"Processed image: {result['output_path']}")
    print(f"Top zones: {len(result['zones']['top'])}, Left zones: {len(result['zones']['left'])}, Right zones: {len(result['zones']['right'])}, Bottom zones: {len(result['zones']['bottom'])}")