import argparse
import importlib.util
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _load_color_transfer():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lam", "stylization", "color_transfer.py")
    spec = importlib.util.spec_from_file_location("lam_color_transfer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.stylize_frames_with_reference


stylize_frames_with_reference = _load_color_transfer()


def _load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _save_rgb(array, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def _detect_face_bbox(rgb):
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        height, width = rgb.shape[:2]
        size = int(min(width, height) * 0.58)
        return (
            max(0, (width - size) // 2),
            max(0, int(height * 0.18)),
            size,
            size,
        )
    return tuple(max(faces, key=lambda rect: rect[2] * rect[3]))


def _soft_face_mask(shape, bbox, expand=0.28):
    height, width = shape[:2]
    x, y, w, h = bbox
    cx = x + w * 0.5
    cy = y + h * 0.54
    rx = w * (0.58 + expand)
    ry = h * (0.68 + expand)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dist = ((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2
    mask = np.clip((1.18 - dist) / 0.24, 0.0, 1.0)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=max(3.0, w * 0.025))


def _geometry_warp(rgb, bbox, strength=0.65):
    """Apply a smooth 2D face-shape warp toward the rounder style_image2 geometry."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return rgb

    height, width = rgb.shape[:2]
    x, y, w, h = bbox
    cx = x + w * 0.5
    cy = y + h * 0.52

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = (xx - cx) / max(w * 0.5, 1.0)
    yn = (yy - cy) / max(h * 0.5, 1.0)

    face_weight = np.exp(-((xn / 1.05) ** 2 + (yn / 1.18) ** 2) * 1.7)
    lower = np.clip((yn + 0.08) / 1.08, 0.0, 1.0)
    mid = np.exp(-(yn / 0.62) ** 2)

    # Inverse-map from output to input. Values below 1.0 make output features wider/rounder.
    width_scale = 1.0 - strength * face_weight * (0.10 + 0.18 * lower + 0.10 * mid)
    y_scale = 1.0 + strength * face_weight * (0.035 * lower - 0.045 * (1.0 - lower))

    map_x = cx + (xx - cx) * width_scale
    map_y = cy + (yy - cy) * y_scale

    # Local nose transfer: style_image2 has a softer, broader nose bridge/tip.
    nose_cx = cx
    nose_cy = y + h * 0.55
    nose_rx = w * 0.16
    nose_ry = h * 0.22
    nose_weight = np.exp(-(((xx - nose_cx) / max(nose_rx, 1.0)) ** 2 + ((yy - nose_cy) / max(nose_ry, 1.0)) ** 2) * 1.25)
    map_x = nose_cx + (map_x - nose_cx) * (1.0 - strength * 0.15 * nose_weight)

    warped = cv2.remap(
        rgb,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask = _soft_face_mask(rgb.shape, bbox)[..., None]
    return np.clip(warped.astype(np.float32) * mask + rgb.astype(np.float32) * (1.0 - mask), 0, 255).astype(np.uint8)


def _resize_to_height(image, height):
    width = max(1, round(image.width * height / image.height))
    if hasattr(Image, "Resampling"):
        return image.resize((width, height), Image.Resampling.LANCZOS)
    return image.resize((width, height), Image.LANCZOS)


def _labeled_panel(image_path, label, height):
    image = _resize_to_height(Image.open(image_path).convert("RGB"), height)
    label_h = 34
    panel = Image.new("RGB", (image.width, image.height + label_h), (246, 246, 246))
    panel.paste(image, (0, label_h))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 10), label, fill=(24, 24, 24))
    return panel


def _save_comparison(reference_path, style_path, output_path, grid_path, height=320, geometry_path=None):
    panels = [_labeled_panel(reference_path, "reference", height)]
    if geometry_path is not None:
        panels.append(_labeled_panel(geometry_path, "geometry", height))
    panels.extend([
        _labeled_panel(style_path, "style", height),
        _labeled_panel(output_path, "stylized", height),
    ])
    gap = 8
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (width, max(panel.height for panel in panels)), (230, 230, 230))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    os.makedirs(os.path.dirname(grid_path), exist_ok=True)
    canvas.save(grid_path)


def main():
    parser = argparse.ArgumentParser(description="Generate a stylized preview from a reference and style image.")
    parser.add_argument("--reference", default="assets/sample_input/status.png")
    parser.add_argument("--style", default="exps/train_lam/samples/style_average.png")
    parser.add_argument("--output", default="exps/train_lam/stylized/stylized_status.png")
    parser.add_argument("--grid", default="exps/train_lam/stylized/stylized_status_grid.png")
    parser.add_argument("--geometry-output", default=None)
    parser.add_argument("--strength", type=float, default=0.9)
    parser.add_argument("--geometry-strength", type=float, default=0.0)
    args = parser.parse_args()

    reference = _load_rgb(args.reference)
    geometry = reference
    geometry_path = None
    if args.geometry_strength > 0:
        geometry = _geometry_warp(reference, _detect_face_bbox(reference), strength=args.geometry_strength)
        geometry_path = args.geometry_output
        if geometry_path is None:
            root, ext = os.path.splitext(args.output)
            geometry_path = f"{root}_geometry{ext or '.png'}"
        _save_rgb(geometry, geometry_path)

    stylized = stylize_frames_with_reference(
        geometry,
        style_image_path=args.style,
        strength=args.strength,
    )[0]
    _save_rgb(stylized, args.output)
    _save_comparison(args.reference, args.style, args.output, args.grid, geometry_path=geometry_path)

    print(f"stylized_output={args.output}")
    if geometry_path is not None:
        print(f"geometry_output={geometry_path}")
    print(f"comparison_grid={args.grid}")


if __name__ == "__main__":
    main()
