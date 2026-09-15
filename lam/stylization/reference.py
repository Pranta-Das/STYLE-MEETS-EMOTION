import os

from lam.runners.infer.head_utils import preprocess_image


def prepare_style_reference(
    style_image_path,
    flametracking,
    source_size,
    aspect_standard,
    dump_tmp_dir,
    bg_color=1.0,
    track_geometry=True,
):
    if not style_image_path or not os.path.exists(style_image_path):
        return None, None

    image_path = style_image_path
    mask_path = None
    can_read_shape = False

    if track_geometry and flametracking is not None:
        # FLAME-tracking a style image is expensive (preprocess + landmark
        # optimize + export, several seconds) and its result depends only on
        # the style image itself -- identical for every (content, emotion)
        # request that reuses this same style. flametracking has no
        # skip-if-tracked logic of its own (unlike the batch tools, e.g.
        # tools/track_ckplus_disgust.py's skip_existing_tracking), so a batch
        # that reuses the same M style images across many content images
        # would otherwise redo this work up to N times per style. Reuse the
        # export directory by content-id (same naming flametracking.export()
        # itself uses: os.path.splitext(os.path.basename(path))[0]) if it's
        # already there.
        content_id = os.path.splitext(os.path.basename(style_image_path))[0]
        cached_output_dir = os.path.join(flametracking.output_export, content_id)
        cached_image_path = os.path.join(cached_output_dir, "images/00000_00.png")
        cached_flame_path = os.path.join(cached_output_dir, "canonical_flame_param.npz")
        if os.path.exists(cached_image_path) and os.path.exists(cached_flame_path):
            print(f"Reusing cached style FLAME tracking: {cached_output_dir}")
            image_path = cached_image_path
            mask_path = os.path.join(cached_output_dir, "fg_masks/00000_00.png")
            can_read_shape = True
        else:
            try:
                print(f"Preparing style reference with FLAME tracking: {style_image_path}")
                return_code = flametracking.preprocess(style_image_path)
                assert return_code == 0, "style flametracking preprocess failed"
                return_code = flametracking.optimize()
                assert return_code == 0, "style flametracking optimize failed"
                return_code, output_dir = flametracking.export()
                assert return_code == 0, "style flametracking export failed"

                image_path = os.path.join(output_dir, "images/00000_00.png")
                mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")
                can_read_shape = True
            except Exception as exc:
                print(f"Warning: style FLAME tracking failed, using appearance only. Error: {exc}")

    if mask_path is not None and not os.path.exists(mask_path):
        mask_path = None

    if not can_read_shape:
        can_read_shape = os.path.exists(
            os.path.join(os.path.dirname(os.path.dirname(image_path)), "canonical_flame_param.npz")
        )

    style_image, _, _, style_shape_param = preprocess_image(
        image_path,
        mask_path=mask_path,
        intr=None,
        pad_ratio=0,
        bg_color=bg_color,
        max_tgt_size=None,
        aspect_standard=aspect_standard,
        enlarge_ratio=[1.0, 1.0],
        render_tgt_size=source_size,
        multiply=14,
        need_mask=True,
        get_shape_param=can_read_shape,
    )

    if dump_tmp_dir is not None:
        os.makedirs(dump_tmp_dir, exist_ok=True)

    return style_image, style_shape_param
