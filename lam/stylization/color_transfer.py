import os

import cv2
import numpy as np
from PIL import Image


def _load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _prepare_mask(mask, height, width):
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask.mean(axis=-1)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    if mask.max() > 1.0:
        mask = mask / 255.0
    return np.clip(mask, 0.0, 1.0)


def _style_pixel_mask(rgb):
    """Drop near-white / near-black pixels before measuring a style's LAB stats.

    This only ever catches a background that is actually white or actually
    black. It is the right filter for the head crop the tracker exports
    (matted onto a flat white background), and the WRONG filter for a raw
    style photo, whose background is usually mid-tone: on
    figure/004_445_841_4k_guzz-soares-fefi-l.jpg -- a portrait on a muted
    green-gray backdrop -- it keeps 98% of the pixels, so the "style" the
    transfer aims at is mostly backdrop. Measured against that image's own
    tracked head, the resulting target is 21.8 L units darker and 9.1 a
    units greener, which lands on the render as dull greenish skin.

    So callers with access to the tracked crop should pass `style_rgb`
    (see stylize_frames_with_reference) rather than relying on this. This
    stays as the fallback for when tracking was unavailable.
    """
    pixels = rgb.reshape(-1, 3)
    non_white = pixels.max(axis=1) < 245
    non_black = pixels.min(axis=1) > 8
    keep = non_white & non_black
    if keep.sum() < max(100, pixels.shape[0] // 100):
        keep = np.ones(pixels.shape[0], dtype=bool)
    return keep


def _lab_stats(rgb, mask=None, ignore_style_background=False):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pixels = lab.reshape(-1, 3)

    if mask is not None:
        keep = mask.reshape(-1) > 0.05
        if keep.sum() >= 100:
            pixels = pixels[keep]
    elif ignore_style_background:
        keep = _style_pixel_mask(rgb)
        pixels = pixels[keep]

    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    return mean, np.maximum(std, 1.0)


def _transfer_frame(frame, style_mean, style_std, strength, mask=None):
    height, width = frame.shape[:2]
    mask = _prepare_mask(mask, height, width)
    content_mean, content_std = _lab_stats(frame, mask=mask)

    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
    transferred = (lab - content_mean.reshape(1, 1, 3)) / content_std.reshape(1, 1, 3)
    transferred = transferred * style_std.reshape(1, 1, 3) + style_mean.reshape(1, 1, 3)
    transferred = np.clip(transferred, 0, 255).astype(np.uint8)
    transferred = cv2.cvtColor(transferred, cv2.COLOR_LAB2RGB)

    styled = frame.astype(np.float32) * (1.0 - strength) + transferred.astype(np.float32) * strength
    if mask is not None:
        mask = mask[..., None]
        styled = styled * mask + frame.astype(np.float32) * (1.0 - mask)
    return np.clip(styled, 0, 255).astype(np.uint8)


def stylize_frames_with_reference(
    frames,
    style_image_path="./style_image.png",
    masks=None,
    strength=0.75,
    style_rgb=None,
):
    """Apply fast LAB color-statistics transfer from a style image to RGB frames.

    `style_rgb`, when given, is the style's already-matted head crop (uint8
    HWC, background flattened to white) and is measured instead of the file
    at `style_image_path`. Prefer it: the frames being recolored are a head
    on a background, so the statistics they should be matched to are the
    style's HEAD, not its head averaged with whatever backdrop the photo
    happened to have. See _style_pixel_mask for what goes wrong otherwise.
    The caller already has this crop -- prepare_style_reference returns it,
    built from the same FLAME tracking pass that produces the style betas,
    so using it costs nothing extra.
    """
    if style_rgb is None and (not style_image_path or not os.path.exists(style_image_path)):
        return frames

    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim == 3:
        frames = frames[None]

    strength = float(np.clip(strength, 0.0, 1.0))
    style = np.asarray(style_rgb, dtype=np.uint8) if style_rgb is not None else _load_rgb(style_image_path)
    style_mean, style_std = _lab_stats(style, ignore_style_background=True)

    styled_frames = []
    for idx, frame in enumerate(frames):
        mask = None
        if masks is not None:
            mask = masks[idx]
        styled_frames.append(_transfer_frame(frame, style_mean, style_std, strength, mask=mask))
    return np.stack(styled_frames, axis=0)
