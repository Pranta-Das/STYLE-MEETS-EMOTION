"""Fetch emotion-labeled reference photos and FLAME-track them.

Modeled on tools/track_content_refs.py -- same tracker construction, same
manifest/summary JSON shape -- but the candidate images don't already exist
as local files: they're streamed from a Hugging Face image-classification
dataset (default: Piro17/affectnethq, real photos, 7 Ekman classes: anger,
disgust, fear, happy, neutral, sad, surprise -- verified during Phase 0 to
track cleanly at these resolutions, unlike FER2013's 48x48 grayscale). So
this tool does two things per class: materialize `images_per_class` images
to local JPGs (skipping any already downloaded), then FLAME-track them the
same way track_content_refs.py tracks its CelebA-HQ glob.

The per-image target this pipeline actually needs is `expr`/`jaw_pose` from
each tracked frame's flame_param/00000.npz -- NOT canonical_flame_param.npz,
which stores the canonical (neutral, shape-only) FLAME params with expr
always zero and jaw_pose always the same fixed value regardless of the
photo's actual expression (confirmed empirically during Phase 0: every
canonical_flame_param.npz across 14 test images had identical zero expr).
"""
import argparse
import json
import os
import time

from omegaconf import OmegaConf

EMOTION_CLASSES = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _content_id(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]


def _flame_param_path(tracking_output_dir, image_path):
    # The tracker's export dir is keyed purely by the input image's
    # basename stem (see tools/track_content_refs.py's _export_shape_path,
    # which does the same thing) -- it does NOT know or care which class
    # subfolder the image came from. Filenames MUST therefore be globally
    # unique across all classes, not just within one class's folder:
    # reusing "00000.jpg" in every class directory (an earlier version of
    # this tool's download step did exactly that) makes multiple classes'
    # images collide into the SAME tracking_output/export/00000 slot,
    # which either silently overwrites a previous class's tracked result or
    # throws "Found multiple sequences by '00000'" -- confirmed the hard
    # way: a full 700-image run with non-unique filenames came back with
    # 0 usable results. The download step below bakes the class name into
    # the filename itself specifically to make this impossible.
    return os.path.join(tracking_output_dir, "export", _content_id(image_path), "flame_param", "00000.npz")


def download_class_images(cfg):
    """Stream the HF dataset, saving up to `images_per_class` real JPGs per
    class into `local_image_dir/<class>/<class>_NNNNN.jpg` -- the class
    prefix keeps filenames globally unique across classes (see
    _flame_param_path for why that's required, not just tidy). Skips
    classes that already have enough images cached locally so re-runs are
    cheap.
    """
    local_image_dir = cfg.get("local_image_dir", "data/emotion_refs/images")
    images_per_class = int(cfg.get("images_per_class", 300))

    counts = {}
    for cls in EMOTION_CLASSES:
        class_dir = os.path.join(local_image_dir, cls)
        os.makedirs(class_dir, exist_ok=True)
        counts[cls] = len([f for f in os.listdir(class_dir) if f.endswith(".jpg")])

    if all(counts[cls] >= images_per_class for cls in EMOTION_CLASSES):
        print(f"[download] all classes already have >= {images_per_class} images, skipping fetch")
        return counts

    from datasets import load_dataset

    hf_dataset = cfg.get("hf_dataset", "Piro17/affectnethq")
    print(f"[download] streaming {hf_dataset} to fill classes up to {images_per_class} images each")
    ds = load_dataset(hf_dataset, split="train", streaming=True)
    names = ds.features["label"].names
    assert set(EMOTION_CLASSES) <= set(names), f"dataset classes {names} missing some of {EMOTION_CLASSES}"

    for example in ds:
        if all(counts[cls] >= images_per_class for cls in EMOTION_CLASSES):
            break
        label = names[example["label"]]
        if label not in EMOTION_CLASSES or counts[label] >= images_per_class:
            continue
        idx = counts[label]
        out_path = os.path.join(local_image_dir, label, f"{label}_{idx:05d}.jpg")
        example["image"].convert("RGB").save(out_path, quality=95)
        counts[label] += 1

    print(f"[download] done: {counts}")
    return counts


def track_emotion_refs(cfg, dry_run=False):
    output_dir = cfg.get("output_dir", "exps/train_lam/emotion_refs")
    os.makedirs(output_dir, exist_ok=True)
    local_image_dir = cfg.get("local_image_dir", "data/emotion_refs/images")
    tracking_output_dir = cfg.get("tracking_output_dir", "tracking_output")
    skip_existing = bool(cfg.get("skip_existing_tracking", True))

    # Download is the cheap part (image save, no model inference) -- do it
    # even on --dry-run so the preview reflects real candidate counts,
    # unlike tracking itself (the expensive part) which --dry-run skips.
    download_class_images(cfg)

    candidates = []
    for cls in EMOTION_CLASSES:
        class_dir = os.path.join(local_image_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        for fn in sorted(os.listdir(class_dir)):
            if fn.endswith(".jpg"):
                candidates.append((cls, os.path.join(class_dir, fn)))

    images_per_class = int(cfg.get("images_per_class", 300))
    candidates = candidates[: images_per_class * len(EMOTION_CLASSES)]

    summary = {
        "stage": "track_emotion_refs",
        "hf_dataset": cfg.get("hf_dataset", "Piro17/affectnethq"),
        "tracking_output_dir": tracking_output_dir,
        "num_candidates": len(candidates),
        "skip_existing_tracking": skip_existing,
        "dry_run": bool(dry_run),
        "items": [],
    }

    if not candidates:
        summary["error"] = "No candidate images found (download step may have failed)."
        summary_path = os.path.join(output_dir, "tracking_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        raise FileNotFoundError("No emotion reference images found to track.")

    if dry_run:
        for index, (cls, image_path) in enumerate(candidates, start=1):
            flame_path = _flame_param_path(tracking_output_dir, image_path)
            summary["items"].append({
                "index": index,
                "emotion_label": cls,
                "image_path": image_path,
                "flame_param_path": flame_path,
                "status": "would_skip_existing" if os.path.exists(flame_path) else "would_track",
            })
        summary["num_would_track"] = sum(1 for item in summary["items"] if item["status"] == "would_track")
        summary_path = os.path.join(output_dir, "tracking_dry_run_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"tracking_dry_run_summary={summary_path}")
        print(f"num_candidates={len(candidates)}")
        print(f"num_would_track={summary['num_would_track']}")
        return summary

    from tools.flame_tracking_single_image import FlameTrackingSingleImage
    from types import SimpleNamespace

    tracker_args = SimpleNamespace(output_dir=tracking_output_dir, config_name="alignment", blender_path=None)
    tracker = FlameTrackingSingleImage(
        output_dir=tracking_output_dir,
        detect_iris_landmarks=False,
        args=tracker_args,
    )

    for index, (cls, image_path) in enumerate(candidates, start=1):
        started = time.time()
        flame_path = _flame_param_path(tracking_output_dir, image_path)
        row = {
            "index": index,
            "emotion_label": cls,
            "image_path": image_path,
            "flame_param_path": flame_path,
        }

        if skip_existing and os.path.exists(flame_path):
            row["status"] = "skipped_existing"
            row["seconds"] = 0.0
            summary["items"].append(row)
            continue

        print(f"[track-emotion] {index}/{len(candidates)} [{cls}] {image_path}", flush=True)
        try:
            return_code = tracker.preprocess(image_path)
            if return_code != 0:
                row["status"] = "failed_preprocess"
                row["return_code"] = int(return_code)
                summary["items"].append(row)
                continue

            return_code = tracker.optimize()
            if return_code != 0:
                row["status"] = "failed_optimize"
                row["return_code"] = int(return_code)
                summary["items"].append(row)
                continue

            return_code, returned_export_dir = tracker.export()
            row["export_dir"] = returned_export_dir
            if return_code != 0:
                row["status"] = "failed_export"
                row["return_code"] = int(return_code)
            elif os.path.exists(flame_path):
                row["status"] = "tracked"
            else:
                row["status"] = "missing_flame_param_after_export"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)

        row["seconds"] = round(time.time() - started, 3)
        summary["items"].append(row)

    summary["num_tracked"] = sum(1 for item in summary["items"] if item["status"] == "tracked")
    summary["num_skipped_existing"] = sum(1 for item in summary["items"] if item["status"] == "skipped_existing")
    summary["num_failed"] = sum(
        1 for item in summary["items"] if item["status"] not in {"tracked", "skipped_existing"}
    )
    for cls in EMOTION_CLASSES:
        summary[f"num_usable_{cls}"] = sum(
            1 for item in summary["items"]
            if item["emotion_label"] == cls
            and item["status"] in {"tracked", "skipped_existing"}
            and os.path.exists(item["flame_param_path"])
        )

    summary_path = os.path.join(output_dir, "tracking_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    manifest = [
        {
            "emotion_label": item["emotion_label"],
            "image_path": item["image_path"],
            "flame_param_path": item["flame_param_path"],
        }
        for item in summary["items"]
        if item["status"] in {"tracked", "skipped_existing"} and os.path.exists(item["flame_param_path"])
    ]
    manifest_path = os.path.join(output_dir, "tracked_emotion_refs.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"tracking_summary={summary_path}")
    print(f"tracked_emotion_refs={manifest_path}")
    print(f"num_tracked={summary['num_tracked']}")
    print(f"num_skipped_existing={summary['num_skipped_existing']}")
    print(f"num_failed={summary['num_failed']}")
    print(f"num_usable={len(manifest)}")
    for cls in EMOTION_CLASSES:
        print(f"  {cls}: {summary[f'num_usable_{cls}']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Fetch and FLAME-track emotion reference photos.")
    parser.add_argument("--config", default="configs/train/track_emotion_refs.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Only report candidate counts, skip tracking.")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    track_emotion_refs(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
