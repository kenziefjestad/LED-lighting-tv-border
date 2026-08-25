# Process Image for LED Lighting TV Border

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

ZONE_SIZE = 100

def output_image(image: np.ndarray, output_path: str) -> None:
    """Save the processed image to the specified output path."""
    output_image = Image.fromarray(image)
    output_image.save(output_path)
    print(f"Saved processed image to {output_path}")

def draw_square_on_image(image: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    """Fill the square on the image for visualization."""
    x0, y0 = top_left
    x1, y1 = bottom_right

    height, width = image.shape[:2]
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))

    image[y0:y1, x0:x1] = color

    # print(f"Drawing filled square from {top_left} to {bottom_right} with color {color}")

def draw_square_outline_on_image(image: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    """Draw only the border of a square on the image for visualization."""
    x0, y0 = top_left
    x1, y1 = bottom_right

    height, width = image.shape[:2]
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))

    image[y0:y1, x0] = color
    image[y0:y1, x1 - 1] = color
    image[y0, x0:x1] = color
    image[y1 - 1, x0:x1] = color

    # print(f"Drawing square border from {top_left} to {bottom_right} with color {color}")

def extract_from_array(image: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> np.ndarray:
    """Extract a sub-array from the image based on the specified coordinates."""
    x0, y0 = top_left
    x1, y1 = bottom_right

    height, width = image.shape[:2]
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))

    extracted_array = image[y0:y1, x0:x1]
    # print(f"Extracted array from {top_left} to {bottom_right}, shape: {extracted_array.shape}")
    return extracted_array

def get_zone_arrays(image: np.ndarray, num_x_zones: int, num_y_zones: int) -> Dict[str, List[np.ndarray]]:
    """Get the zone arrays for the LED strip layout."""
    height, width = image.shape[:2]
    zone_arrays = {
        "top_left_corner": [],
        "top_right_corner": [],
        "bottom_left_corner": [],
        "bottom_right_corner": [],
        "top_row": [],
        "bottom_row": [],
        "left_column": [],
        "right_column": []
    }

    # height, width = image.shape[:2]
    # image = np.array(image, copy=True)

    horizontal_zone_height = ZONE_SIZE
    horizontal_zone_width = ZONE_SIZE
    vertical_zone_height = ZONE_SIZE
    vertical_zone_width = ZONE_SIZE

    corner_height = ZONE_SIZE + (height % ZONE_SIZE // 2)
    corner_width = ZONE_SIZE + (width % ZONE_SIZE // 2)

    # print(f"Image dimensions: {width}x{height}")
    # print(f"Horizontal zone height: {horizontal_zone_height}, Horizontal zone width: {horizontal_zone_width}")
    # print(f"Vertical zone height: {vertical_zone_height}, Vertical zone width: {vertical_zone_width}")
    # print(f"Corner height: {corner_height}, Corner width: {corner_width}")

    top_row = max(1, num_x_zones)
    bottom_row = max(1, num_x_zones)
    left_column = max(1, num_y_zones)
    right_column = max(1, num_y_zones)

    # Draw corner zones
    zone_arrays["top_left_corner"].append(extract_from_array(image, (0, 0), (corner_width, corner_height)))  # Top-left corner
    zone_arrays["top_right_corner"].append(extract_from_array(image, (width - corner_width, 0), (width, corner_height)))  # Top-right corner
    zone_arrays["bottom_left_corner"].append(extract_from_array(image, (0, height - corner_height), (corner_width, height)))  # Bottom-left corner
    zone_arrays["bottom_right_corner"].append(extract_from_array(image, (width - corner_width, height - corner_height), (width, height)))  # Bottom-right corner

    for i in range(top_row):
        x0 = corner_width + (i * horizontal_zone_width)
        x1 = min(corner_width + ((i + 1) * horizontal_zone_width), width)
        # print(f"Drawing top zone {i}: from ({x0}, 0) to ({x1}, {horizontal_zone_height})")
        zone_arrays["top_row"].append(extract_from_array(image, (x0, 0), (x1, horizontal_zone_height)))

    for i in range(bottom_row):
        x0 = corner_width + (i * horizontal_zone_width)
        x1 = min(corner_width + ((i + 1) * horizontal_zone_width), width)
        zone_arrays["bottom_row"].append(extract_from_array(image, (x0, height - horizontal_zone_height), (x1, height)))

    for i in range(left_column):
        y0 = corner_height + (i * vertical_zone_height)
        y1 = min(corner_height + ((i + 1) * vertical_zone_height), height)
        zone_arrays["left_column"].append(extract_from_array(image, (0, y0), (horizontal_zone_width, y1)))

    for i in range(right_column):
        y0 = corner_height + (i * vertical_zone_height)
        y1 = min(corner_height + ((i + 1) * vertical_zone_height), height)
        zone_arrays["right_column"].append(extract_from_array(image, (width - vertical_zone_width, y0), (width, y1)))

    return zone_arrays

def draw_zones_on_image(image: np.ndarray, num_x_zones: int, num_y_zones: int) -> np.ndarray:
    """Draw only the border zones for the LED strip layout."""
    height, width = image.shape[:2]
    image = np.array(image, copy=True)

    horizontal_zone_height = ZONE_SIZE
    horizontal_zone_width = ZONE_SIZE
    vertical_zone_height = ZONE_SIZE
    vertical_zone_width = ZONE_SIZE

    corner_height = ZONE_SIZE + (height % ZONE_SIZE // 2)
    corner_width = ZONE_SIZE + (width % ZONE_SIZE // 2)

    top_row = max(1, num_x_zones)
    bottom_row = max(1, num_x_zones)
    left_column = max(1, num_y_zones)
    right_column = max(1, num_y_zones)

    # Draw corner zones
    draw_square_on_image(image, (0, 0), (corner_width, corner_height), (255, 255, 255))  # Top-left corner
    draw_square_on_image(image, (width - corner_width, 0), (width, corner_height), (255, 255, 255))  # Top-right corner
    draw_square_on_image(image, (0, height - corner_height), (corner_width, height), (255, 255, 255))  # Bottom-left corner
    draw_square_on_image(image, (width - corner_width, height - corner_height), (width, height), (255, 255, 255))  # Bottom-right corner

    for i in range(top_row):
        x0 = corner_width + (i * horizontal_zone_width)
        x1 = min(corner_width + ((i + 1) * horizontal_zone_width), width)
        # print(f"Drawing top zone {i}: from ({x0}, 0) to ({x1}, {horizontal_zone_height})")
        draw_square_on_image(image, (x0, 0), (x1, horizontal_zone_height), (255, 255, 255))

    for i in range(bottom_row):
        x0 = corner_width + (i * horizontal_zone_width)
        x1 = min(corner_width + ((i + 1) * horizontal_zone_width), width)
        draw_square_on_image(image, (x0, height - horizontal_zone_height), (x1, height), (255, 255, 255))

    for i in range(left_column):
        y0 = corner_height + (i * vertical_zone_height)
        y1 = min(corner_height + ((i + 1) * vertical_zone_height), height)
        draw_square_on_image(image, (0, y0), (horizontal_zone_width, y1), (255, 255, 255))

    for i in range(right_column):
        y0 = corner_height + (i * vertical_zone_height)
        y1 = min(corner_height + ((i + 1) * vertical_zone_height), height)
        draw_square_on_image(image, (width - vertical_zone_width, y0), (width, y1), (255, 255, 255))

    return image

def draw_outline_zones_on_image(image: np.ndarray, num_x_zones: int, num_y_zones: int) -> np.ndarray:
    """Draw only the border zones for the LED strip layout."""
    height, width = image.shape[:2]
    image = np.array(image, copy=True)

    horizontal_zone_height = ZONE_SIZE
    horizontal_zone_width = ZONE_SIZE
    vertical_zone_height = ZONE_SIZE
    vertical_zone_width = ZONE_SIZE

    corner_height = ZONE_SIZE + (height % ZONE_SIZE // 2)
    corner_width = ZONE_SIZE + (width % ZONE_SIZE // 2)

    top_row = max(1, num_x_zones)
    bottom_row = max(1, num_x_zones)
    left_column = max(1, num_y_zones)
    right_column = max(1, num_y_zones)

    # Draw corner zones
    draw_square_outline_on_image(image, (0, 0), (corner_width, corner_height), (255, 255, 255))  # Top-left corner
    draw_square_outline_on_image(image, (width - corner_width, 0), (width, corner_height), (255, 255, 255))  # Top-right corner
    draw_square_outline_on_image(image, (0, height - corner_height), (corner_width, height), (255, 255, 255))  # Bottom-left corner
    draw_square_outline_on_image(image, (width - corner_width, height - corner_height), (width, height), (255, 255, 255))  # Bottom-right corner

    for i in range(top_row):
        x0 = corner_width + (i * horizontal_zone_width)
        x1 = min(corner_width + ((i + 1) * horizontal_zone_width), width)
        # print(f"Drawing top zone {i}: from ({x0}, 0) to ({x1}, {horizontal_zone_height})")
        draw_square_outline_on_image(image, (x0, 0), (x1, horizontal_zone_height), (255, 255, 255))

    for i in range(bottom_row):
        x0 = corner_width + (i * horizontal_zone_width)
        x1 = min(corner_width + ((i + 1) * horizontal_zone_width), width)
        draw_square_outline_on_image(image, (x0, height - horizontal_zone_height), (x1, height), (255, 255, 255))

    for i in range(left_column):
        y0 = corner_height + (i * vertical_zone_height)
        y1 = min(corner_height + ((i + 1) * vertical_zone_height), height)
        draw_square_outline_on_image(image, (0, y0), (horizontal_zone_width, y1), (255, 255, 255))

    for i in range(right_column):
        y0 = corner_height + (i * vertical_zone_height)
        y1 = min(corner_height + ((i + 1) * vertical_zone_height), height)
        draw_square_outline_on_image(image, (width - vertical_zone_width, y0), (width, y1), (255, 255, 255))

    return image

def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get the dimensions (width, height) of an image."""
    with Image.open(image_path) as img:
        return img.size  # Returns (width, height)

def get_zones(rgb_array: np.ndarray):
    """Get the zone colors for an image without saving a preview."""
    # image = Image.open(image_path).convert("RGB")
    # rgb_array = np.asarray(image)
    # zone_colors = _compute_zone_colors(rgb_array, layout)
    # zone_colors = _smooth_zone_colors(zone_colors)
    # return zone_colors

    num_x_zones = (rgb_array.shape[1] - 2 * ZONE_SIZE) // ZONE_SIZE  # Example: 20 pixels per zone
    num_y_zones = (rgb_array.shape[0] - 2 * ZONE_SIZE) // ZONE_SIZE  # Example: 20 pixels per zone
    # zone_image = draw_zones_on_image(rgb_array, num_x_zones, num_y_zones)
    # output_image(zone_image, "images/output/zone_preview.png")

def output_zones(zones: Dict[str, List[Tuple[int, int, int]]], output_dir: str) -> None:
    """Output the zone colors to separate image files for visualization."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for zone_name, zone_colors in zones.items():
        for idx, zone_color in enumerate(zone_colors):
            # Create a simple image with the zone color
            zone_image = Image.new("RGB", (ZONE_SIZE, ZONE_SIZE), zone_color)
            zone_image.save(output_path / f"{zone_name}_{idx}.png")

def compute_average_zone_color(zone_array: np.ndarray) -> Tuple[int, int, int]:
    """Compute the average color of a zone array."""
    avg_color = np.mean(zone_array, axis=(0, 1)).astype(int)
    return tuple(avg_color)

def attach_zone_colors_to_image(image: np.ndarray, zones: Dict[str, List[Tuple[int, int, int]]]) -> np.ndarray:
    """Attach the average zone colors to the image for visualization."""
    height, width = image.shape[:2]
    output_image = np.array(image, copy=True)

    corner_height = ZONE_SIZE + (height % ZONE_SIZE // 2)
    corner_width = ZONE_SIZE + (width % ZONE_SIZE // 2)

    # Draw the average color of each zone on the image
    for zone_name, zone_colors in zones.items():
        for idx, zone_color in enumerate(zone_colors):
            if zone_name == "top_left_corner":
                draw_square_on_image(output_image, (0, 0), (corner_width, corner_height), zone_color)
            elif zone_name == "top_right_corner":
                draw_square_on_image(output_image, (width - corner_width, 0), (width, corner_height), zone_color)
            elif zone_name == "bottom_left_corner":
                draw_square_on_image(output_image, (0, height - corner_height), (corner_width, height), zone_color)
            elif zone_name == "bottom_right_corner":
                draw_square_on_image(output_image, (width - corner_width, height - corner_height), (width, height), zone_color)
            elif zone_name == "top_row":
                x0 = corner_width + (idx * ZONE_SIZE)
                x1 = min(corner_width + ((idx + 1) * ZONE_SIZE), width - corner_width)
                draw_square_on_image(output_image, (x0, 0), (x1, ZONE_SIZE), zone_color)
            elif zone_name == "bottom_row":
                x0 = corner_width + (idx * ZONE_SIZE)
                x1 = min(corner_width + ((idx + 1) * ZONE_SIZE), width - corner_width)
                draw_square_on_image(output_image, (x0, height - ZONE_SIZE), (x1, height), zone_color)
            elif zone_name == "left_column":
                y0 = corner_height + (idx * ZONE_SIZE)
                y1 = min(corner_height + ((idx + 1) * ZONE_SIZE), height - corner_height)
                draw_square_on_image(output_image, (0, y0), (ZONE_SIZE, y1), zone_color)
            elif zone_name == "right_column":
                y0 = corner_height + (idx * ZONE_SIZE)
                y1 = min(corner_height + ((idx + 1) * ZONE_SIZE), height - corner_height)
                draw_square_on_image(output_image, (width - ZONE_SIZE, y0), (width, y1), zone_color)

    return output_image

def process_zones(zones: Dict[str, List[np.ndarray]]) -> Dict[str, List[Tuple[int, int, int]]]:
    """Process the zone arrays to compute average colors for each zone."""
    processed_zones = {}
    for zone_name, zone_arrays in zones.items():
        processed_zones[zone_name] = [compute_average_zone_color(zone_array) for zone_array in zone_arrays]
    print(f"Processed zones: {processed_zones}")
    return processed_zones

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
    rgb_array = np.array(np.asarray(image), copy=True)

    num_x_zones = (rgb_array.shape[1] - 2 * ZONE_SIZE) // ZONE_SIZE  # Example: 20 pixels per zone
    num_y_zones = (rgb_array.shape[0] - 2 * ZONE_SIZE) // ZONE_SIZE  # Example: 20 pixels per zone

    zone_arrays = get_zone_arrays(rgb_array, num_x_zones=num_x_zones, num_y_zones=num_y_zones)
    zone_colors = process_zones(zone_arrays)

    attached_image = attach_zone_colors_to_image(rgb_array, zone_colors)
    output_image(attached_image, destination)

    # image = draw_outline_zones_on_image(rgb_array, num_x_zones=num_x_zones, num_y_zones=num_y_zones)
    # output_image(image, destination)
    
    # output_zones(zone_colors, "images/output/zones")
    # print(f"Zone arrays: {zone_arrays}")
    # print(f"Zone colors: {zone_colors}")

    # get_zones(rgb_array)

    # extracted_array = extract_from_array(rgb_array, (2, 2), (100, 100))
    # print(f"Extracted array shape: {extracted_array.shape}")
    # print(f"Extracted array:\n{extracted_array}")

    # draw_zones_on_image(rgb_array, num_x_zones=40, num_y_zones=45)

    # print(rgb_array.shape)
    # image_width, image_height = get_image_dimensions(image_path)
    # print(f"Image dimensions: {image_width}x{image_height}")
    # zone_colors = _compute_zone_colors(rgb_array, DEFAULT_LAYOUT)
    # zone_colors = _smooth_zone_colors(zone_colors)

    # output_dir = destination.parent
    # output_dir.mkdir(parents=True, exist_ok=True)

    # _save_zone_preview(destination, rgb_array, zone_colors, DEFAULT_LAYOUT)

    # return {
    #     "layout": DEFAULT_LAYOUT,
    #     "zones": zone_colors,
    #     "output_path": str(destination),
    # }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image for LED border color sampling.")
    parser.add_argument("image_path", help="Input image to process")
    args = parser.parse_args()

    result = process_image(f"images/input/{args.image_path}", f"images/output/output_{args.image_path}")
    # print(f"Processed image: {result['output_path']}")
    # print(f"Top zones: {len(result['zones']['top'])}, Left zones: {len(result['zones']['left'])}, Right zones: {len(result['zones']['right'])}, Bottom zones: {len(result['zones']['bottom'])}")