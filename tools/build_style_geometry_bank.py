import argparse
import glob
import json
import os
import shutil

import numpy as np
from omegaconf import OmegaConf


def _load_shape(path, shape_dims):
    data = np.load(path)
    if "shape" not in data:
        raise KeyError(f"{path} does not contain a 'shape' array")
    shape = np.asarray(data["shape"], dtype=np.float32).reshape(-1)
    if shape_dims is not None and int(shape_dims) > 0:
        shape = shape[: int(shape_dims)]
    return shape


def _style_id_from_shape_path(path):
    return os.path.basename(os.path.dirname(path))


def _find_style_image(style_image_dir, style_id):
    if not style_image_dir:
        return None
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = os.path.join(style_image_dir, style_id + ext)
        if os.path.exists(candidate):
            return candidate
    matches = glob.glob(os.path.join(style_image_dir, style_id + ".*"))
    return matches[0] if matches else None


def _export_image_for_shape(shape_path):
    export_dir = os.path.dirname(shape_path)
    candidate = os.path.join(export_dir, "images", "00000_00.png")
    return candidate if os.path.exists(candidate) else None


def build_geometry_bank(cfg):
    output_dir = cfg.get("output_dir", "exps/train_lam/stage3_geometry_aahq")
    os.makedirs(output_dir, exist_ok=True)

    shape_dims = cfg.get("shape_dims", 10)
    source_shape_path = cfg.source_shape_path
    source_shape = _load_shape(source_shape_path, shape_dims)

    rows = []
    for shape_path in sorted(glob.glob(cfg.style_shape_glob)):
        if os.path.normpath(shape_path) == os.path.normpath(source_shape_path):
            continue
        try:
            style_shape = _load_shape(shape_path, shape_dims)
        except Exception as exc:
            rows.append({
                "shape_path": shape_path,
                "error": str(exc),
            })
            continue

        dims = min(source_shape.shape[0], style_shape.shape[0])
        delta = style_shape[:dims] - source_shape[:dims]
        style_id = _style_id_from_shape_path(shape_path)
        style_image_path = _find_style_image(cfg.get("style_image_dir", None), style_id)
        if cfg.get("require_style_image_match", False) and not style_image_path:
            continue
        rows.append({
            "style_id": style_id,
            "shape_path": shape_path,
            "style_image_path": style_image_path,
            "export_image_path": _export_image_for_shape(shape_path),
            "shape_l2": float(np.linalg.norm(delta)),
            "shape_mean_abs": float(np.mean(np.abs(delta))),
            "shape_dims": int(dims),
        })

    valid = [row for row in rows if "shape_l2" in row]
    min_shape_l2 = float(cfg.get("min_shape_l2", 0.0))
    max_shape_l2 = float(cfg.get("max_shape_l2", 1.0e9))
    selected = [
        row for row in valid
        if min_shape_l2 <= row["shape_l2"] <= max_shape_l2
    ]
    selected.sort(key=lambda row: row["shape_l2"], reverse=True)
    selected = selected[: int(cfg.get("top_k", 32))]

    selected_dir = os.path.join(output_dir, "selected")
    if cfg.get("copy_selected_images", True):
        os.makedirs(selected_dir, exist_ok=True)
        if cfg.get("clean_selected_dir", False):
            for existing in glob.glob(os.path.join(selected_dir, "*")):
                if os.path.isfile(existing) or os.path.islink(existing):
                    os.remove(existing)
        for rank, row in enumerate(selected, start=1):
            image_path = row.get("style_image_path") or row.get("export_image_path")
            if not image_path or not os.path.exists(image_path):
                continue
            _, ext = os.path.splitext(image_path)
            dst = os.path.join(selected_dir, f"{rank:03d}_{row['style_id']}{ext or '.png'}")
            shutil.copy2(image_path, dst)
            row["selected_copy_path"] = dst

    summary = {
        "stage": "style_geometry_bank",
        "source_shape_path": source_shape_path,
        "style_shape_glob": cfg.style_shape_glob,
        "style_image_dir": cfg.get("style_image_dir", None),
        "require_style_image_match": bool(cfg.get("require_style_image_match", False)),
        "shape_dims": int(shape_dims),
        "num_candidates": len(valid),
        "num_selected": len(selected),
        "min_shape_l2": min_shape_l2,
        "max_shape_l2": max_shape_l2,
        "selected": selected,
        "all_candidates": sorted(valid, key=lambda row: row["shape_l2"], reverse=True),
        "errors": [row for row in rows if "error" in row],
    }

    summary_path = os.path.join(output_dir, "geometry_bank_summary.json")
    selected_path = os.path.join(output_dir, "selected_geometry_refs.json")
    selected_images_path = os.path.join(output_dir, "selected_style_images.txt")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(selected_path, "w") as f:
        json.dump(selected, f, indent=2)
    with open(selected_images_path, "w") as f:
        for row in selected:
            image_path = row.get("style_image_path") or row.get("export_image_path")
            if image_path:
                f.write(image_path + "\n")

    print(f"geometry_bank_summary={summary_path}")
    print(f"selected_geometry_refs={selected_path}")
    print(f"selected_style_images={selected_images_path}")
    print(f"num_candidates={len(valid)}")
    print(f"num_selected={len(selected)}")
    if selected:
        best = selected[0]
        print(f"top_style={best['style_id']} shape_l2={best['shape_l2']:.4f}")
        print(f"top_style_image={best.get('style_image_path') or best.get('export_image_path')}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build a ranked FLAME geometry reference bank.")
    parser.add_argument("--config", default="configs/train/lam_style_stage3_geometry_aahq.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    build_geometry_bank(cfg)


if __name__ == "__main__":
    main()
