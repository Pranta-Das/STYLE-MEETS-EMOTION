"""FLAME-track the padded CK+ disgust set (data/ckplus_disgust/*.jpg) to
augment the weak AffectNet-HQ disgust class -- see STYLE_GEOMETRY_PIPELINE.md
for why: AffectNet-HQ's disgust references have no strong distinctive
direction in FLAME expr space at all (best component gap 0.30 vs. every
other class comfortably above 0.5), independent of any training-side fix,
while CK+'s POSED, peak-apex-expression disgust photos read as clearly and
strongly disgusted even after being upscaled from their native 48x48.

Mirrors tools/track_emotion_refs.py's tracker construction and manifest
shape (see that file for why flame_param/00000.npz is the right file to
read, not canonical_flame_param.npz).
"""
import argparse
import json
import os
import time

from omegaconf import OmegaConf


def _content_id(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]


def _flame_param_path(tracking_output_dir, image_path):
    return os.path.join(tracking_output_dir, "export", _content_id(image_path), "flame_param", "00000.npz")


def track_ckplus_disgust(cfg, dry_run=False):
    output_dir = cfg.get("output_dir", "exps/train_lam/emotion_refs")
    os.makedirs(output_dir, exist_ok=True)
    image_dir = cfg.get("image_dir", "data/ckplus_disgust")
    tracking_output_dir = cfg.get("tracking_output_dir", "tracking_output")
    skip_existing = bool(cfg.get("skip_existing_tracking", True))
    max_images = int(cfg.get("max_images", 10**9))

    candidates = sorted(
        os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")
    )[:max_images]

    summary = {"stage": "track_ckplus_disgust", "num_candidates": len(candidates), "dry_run": bool(dry_run), "items": []}

    if dry_run:
        for index, image_path in enumerate(candidates, start=1):
            flame_path = _flame_param_path(tracking_output_dir, image_path)
            summary["items"].append({
                "index": index, "image_path": image_path, "flame_param_path": flame_path,
                "status": "would_skip_existing" if os.path.exists(flame_path) else "would_track",
            })
        summary["num_would_track"] = sum(1 for i in summary["items"] if i["status"] == "would_track")
        print(f"num_candidates={len(candidates)} num_would_track={summary['num_would_track']}")
        return summary

    from tools.flame_tracking_single_image import FlameTrackingSingleImage
    from types import SimpleNamespace

    tracker_args = SimpleNamespace(output_dir=tracking_output_dir, config_name="alignment", blender_path=None)
    tracker = FlameTrackingSingleImage(output_dir=tracking_output_dir, detect_iris_landmarks=False, args=tracker_args)

    for index, image_path in enumerate(candidates, start=1):
        started = time.time()
        flame_path = _flame_param_path(tracking_output_dir, image_path)
        row = {"index": index, "image_path": image_path, "flame_param_path": flame_path}

        if skip_existing and os.path.exists(flame_path):
            row["status"] = "skipped_existing"
            summary["items"].append(row)
            continue

        print(f"[track-ckplus-disgust] {index}/{len(candidates)} {image_path}", flush=True)
        try:
            if tracker.preprocess(image_path) != 0:
                row["status"] = "failed_preprocess"
                summary["items"].append(row)
                continue
            if tracker.optimize() != 0:
                row["status"] = "failed_optimize"
                summary["items"].append(row)
                continue
            return_code, export_dir = tracker.export()
            row["export_dir"] = export_dir
            row["status"] = "tracked" if (return_code == 0 and os.path.exists(flame_path)) else "failed_export"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
        row["seconds"] = round(time.time() - started, 3)
        summary["items"].append(row)

    summary["num_tracked"] = sum(1 for i in summary["items"] if i["status"] in {"tracked", "skipped_existing"})
    summary["num_failed"] = len(summary["items"]) - summary["num_tracked"]

    summary_path = os.path.join(output_dir, "ckplus_disgust_tracking_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    manifest = [
        {"emotion_label": "disgust", "image_path": i["image_path"], "flame_param_path": i["flame_param_path"]}
        for i in summary["items"]
        if i["status"] in {"tracked", "skipped_existing"} and os.path.exists(i["flame_param_path"])
    ]
    manifest_path = os.path.join(output_dir, "tracked_ckplus_disgust.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"summary={summary_path}")
    print(f"manifest={manifest_path}")
    print(f"num_tracked={summary['num_tracked']} num_failed={summary['num_failed']} num_usable={len(manifest)}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="FLAME-track the padded CK+ disgust set.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config) if args.config else OmegaConf.create({})
    track_ckplus_disgust(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
