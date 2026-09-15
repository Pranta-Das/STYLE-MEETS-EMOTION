import argparse
import glob
import json
import os
import random
import time
from types import SimpleNamespace

from omegaconf import OmegaConf


def _style_id(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]


def _export_shape_path(output_dir, image_path):
    return os.path.join(
        output_dir,
        "export",
        _style_id(image_path),
        "canonical_flame_param.npz",
    )


def _select_candidates(paths, cfg):
    paths = sorted(paths)
    max_track_images = int(cfg.get("max_track_images", 24))
    if max_track_images <= 0 or len(paths) <= max_track_images:
        return paths

    strategy = str(cfg.get("sample_strategy", "random")).lower()
    if strategy == "first":
        return paths[:max_track_images]
    if strategy == "spread":
        if max_track_images == 1:
            return [paths[0]]
        step = (len(paths) - 1) / float(max_track_images - 1)
        return [paths[round(idx * step)] for idx in range(max_track_images)]

    rng = random.Random(int(cfg.get("seed", 1234)))
    selected = paths[:]
    rng.shuffle(selected)
    return sorted(selected[:max_track_images])


def track_geometry_refs(cfg, dry_run=False):
    output_dir = cfg.get("output_dir", "exps/train_lam/stage3_geometry_aahq")
    os.makedirs(output_dir, exist_ok=True)

    tracking_output_dir = cfg.get("tracking_output_dir", "tracking_output")
    candidate_paths = sorted(glob.glob(cfg.candidate_style_glob))
    selected_paths = _select_candidates(candidate_paths, cfg)
    skip_existing = bool(cfg.get("skip_existing_tracking", True))

    summary = {
        "stage": "track_stage3_geometry_refs",
        "candidate_style_glob": cfg.candidate_style_glob,
        "tracking_output_dir": tracking_output_dir,
        "num_candidates": len(candidate_paths),
        "num_selected": len(selected_paths),
        "skip_existing_tracking": skip_existing,
        "dry_run": bool(dry_run),
        "items": [],
    }

    if not selected_paths:
        summary["error"] = "No candidate style images found."
        summary_path = os.path.join(output_dir, "tracking_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        raise FileNotFoundError(f"No images matched {cfg.candidate_style_glob}")

    if dry_run:
        for index, image_path in enumerate(selected_paths, start=1):
            shape_path = _export_shape_path(tracking_output_dir, image_path)
            summary["items"].append({
                "index": index,
                "image_path": image_path,
                "style_id": _style_id(image_path),
                "shape_path": shape_path,
                "status": "would_skip_existing" if os.path.exists(shape_path) else "would_track",
            })
        summary["num_would_track"] = sum(1 for item in summary["items"] if item["status"] == "would_track")
        summary["num_would_skip_existing"] = sum(1 for item in summary["items"] if item["status"] == "would_skip_existing")
        summary_path = os.path.join(output_dir, "tracking_dry_run_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"tracking_dry_run_summary={summary_path}")
        print(f"num_candidates={len(candidate_paths)}")
        print(f"num_selected={len(selected_paths)}")
        print(f"num_would_track={summary['num_would_track']}")
        print(f"num_would_skip_existing={summary['num_would_skip_existing']}")
        return summary

    from tools.flame_tracking_single_image import FlameTrackingSingleImage

    tracker_args = SimpleNamespace(
        output_dir=tracking_output_dir,
        config_name="alignment",
        blender_path=None,
    )
    tracker = FlameTrackingSingleImage(
        output_dir=tracking_output_dir,
        detect_iris_landmarks=False,
        args=tracker_args,
    )

    for index, image_path in enumerate(selected_paths, start=1):
        started = time.time()
        shape_path = _export_shape_path(tracking_output_dir, image_path)
        row = {
            "index": index,
            "image_path": image_path,
            "style_id": _style_id(image_path),
            "shape_path": shape_path,
        }

        if skip_existing and os.path.exists(shape_path):
            row["status"] = "skipped_existing"
            row["seconds"] = 0.0
            summary["items"].append(row)
            print(f"[stage3-track] skip existing {image_path}")
            continue

        print(f"[stage3-track] {index}/{len(selected_paths)} {image_path}", flush=True)
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

            return_code, export_dir = tracker.export()
            row["export_dir"] = export_dir
            if return_code != 0:
                row["status"] = "failed_export"
                row["return_code"] = int(return_code)
            elif os.path.exists(shape_path):
                row["status"] = "tracked"
            else:
                row["status"] = "missing_shape_after_export"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)

        row["seconds"] = round(time.time() - started, 3)
        summary["items"].append(row)

    summary["num_tracked"] = sum(1 for item in summary["items"] if item["status"] == "tracked")
    summary["num_skipped_existing"] = sum(1 for item in summary["items"] if item["status"] == "skipped_existing")
    summary["num_failed"] = sum(
        1 for item in summary["items"]
        if item["status"] not in {"tracked", "skipped_existing"}
    )

    summary_path = os.path.join(output_dir, "tracking_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"tracking_summary={summary_path}")
    print(f"num_tracked={summary['num_tracked']}")
    print(f"num_skipped_existing={summary['num_skipped_existing']}")
    print(f"num_failed={summary['num_failed']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Track AAHQ style refs for Stage 3 geometry.")
    parser.add_argument("--config", default="configs/train/lam_style_stage3_geometry_aahq.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Only write the selected candidate list.")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    track_geometry_refs(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
