"""Content/style image pairs for end-to-end style-adapter training.

Each content item is a FLAME-tracked CelebA-HQ image (see
tools/track_content_refs.py) with its own canonical_flame_param.npz and
per-image render camera (from the FLAME tracker's transforms.json), loaded
with the exact same preprocessing / camera conventions the working inference
path (lam/runners/infer/head_utils.py, lam/runners/infer/lam.py:infer_single)
already uses for arbitrary photos. Each style item is a plain AAHQ image with
no tracking required (appearance-only training does not need style geometry).
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from lam.runners.infer.head_utils import preprocess_image, _load_pose, scale_intrs

_CONTENT_FLAME_KEYS = ('expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation')


def _load_content_flame_params(shape_path):
    data = np.load(shape_path, allow_pickle=True)
    params = {}
    for key in _CONTENT_FLAME_KEYS:
        # canonical_flame_param.npz stores each field as [1, D] (single tracked frame);
        # strip that frame dim like head_utils.load_flame_params does, then re-add it
        # as the N_views=1 axis so batch-collate yields [B, 1, D].
        arr = np.asarray(data[key], dtype=np.float32)[0]
        params[key] = torch.from_numpy(arr).unsqueeze(0)  # [1, D]
    betas = np.asarray(data['shape'], dtype=np.float32)
    params['betas'] = torch.from_numpy(betas)  # [D] -> collate gives [B, D]
    return params


def _load_render_camera(export_dir, bg_color=1.0):
    transforms_path = os.path.join(export_dir, 'transforms.json')
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    frame_info = sorted(data['frames'], key=lambda x: x['flame_param_path'])[0]
    c2w, intrinsic = _load_pose(frame_info)
    intrinsic = scale_intrs(intrinsic, 0.5, 0.5)
    bg = torch.tensor([bg_color, bg_color, bg_color], dtype=torch.float32)
    return c2w.unsqueeze(0), intrinsic.unsqueeze(0), bg.unsqueeze(0)  # each [1, ...] -> collate gives [B, 1, ...]


class StylePairDataset(Dataset):
    """Yields (content image + flame params + render camera) paired with a random style image."""

    def __init__(self, content_records, style_image_paths, source_size=512, bg_color=1.0):
        self.content_records = list(content_records)
        self.style_image_paths = list(style_image_paths)
        self.source_size = int(source_size)
        self.bg_color = float(bg_color)
        if not self.content_records:
            raise ValueError('No content records provided.')
        if not self.style_image_paths:
            raise ValueError('No style images provided.')

    def __len__(self):
        return len(self.content_records)

    def _load_style_image(self, style_path):
        rgb, _, _, _ = preprocess_image(
            style_path,
            mask_path=None,
            intr=None,
            pad_ratio=0,
            bg_color=self.bg_color,
            max_tgt_size=None,
            aspect_standard=1.0,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=self.source_size,
            multiply=14,
            need_mask=False,
            get_shape_param=False,
        )
        return rgb[0]  # [3, H, W]

    def __getitem__(self, idx):
        record = self.content_records[idx]
        export_dir = os.path.dirname(record['shape_path'])
        image_path = os.path.join(export_dir, 'images', '00000_00.png')
        mask_path = os.path.join(export_dir, 'fg_masks', '00000_00.png')
        if not os.path.exists(mask_path):
            mask_path = None

        content_rgb, _, _, _ = preprocess_image(
            image_path,
            mask_path=mask_path,
            intr=None,
            pad_ratio=0,
            bg_color=self.bg_color,
            max_tgt_size=None,
            aspect_standard=1.0,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=self.source_size,
            multiply=14,
            need_mask=mask_path is not None,
            get_shape_param=False,
        )

        flame_params = _load_content_flame_params(record['shape_path'])
        render_c2w, render_intr, render_bg = _load_render_camera(export_dir, bg_color=self.bg_color)

        style_path = random.choice(self.style_image_paths)
        style_rgb = self._load_style_image(style_path)

        return {
            'content_image': content_rgb[0],  # [3, H, W]
            'style_image': style_rgb,  # [3, H, W]
            'render_c2w': render_c2w,  # [1, 4, 4]
            'render_intr': render_intr,  # [1, 4, 4]
            'render_bg': render_bg,  # [1, 3]
            'content_id': record['content_id'],
            'style_path': style_path,
            **{f'flame_{k}': v for k, v in flame_params.items()},
        }
