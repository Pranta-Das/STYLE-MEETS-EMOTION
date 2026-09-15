import argparse
import json
import os
import subprocess
import sys
import time

from omegaconf import OmegaConf


def _read_style_paths(path, limit):
    with open(path) as f:
        paths = [line.strip() for line in f if line.strip()]
    if limit is not None and int(limit) > 0:
        paths = paths[: int(limit)]
    return paths


def _style_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def _bool_value(value):
    return "true" if bool(value) else "false"


def _build_infer_command(cfg, style_image_path, case_name, geometry_strength):
    validation = cfg.validation
    output_root = validation.get("output_dir", "exps/train_lam/stage3_geometry_aahq/validation")
    style_slug = _style_id(style_image_path)
    case_root = os.path.join(output_root, style_slug, case_name)

    image_dump = os.path.join(case_root, "images")
    video_dump = os.path.join(case_root, "videos")
    mesh_dump = os.path.join(case_root, "meshes")
    tmp_dump = os.path.join(case_root, "tmp")

    return [
        sys.executable,
        "-m",
        "lam.launch",
        "infer.lam",
        "--config",
        validation.train_config,
        f"model_name={validation.model_name}",
        f"image_input={validation.image_input}",
        f"motion_seqs_dir={validation.motion_seqs_dir}",
        "motion_img_dir=null",
        f"image_dump={image_dump}",
        f"video_dump={video_dump}",
        f"mesh_dump={mesh_dump}",
        f"save_tmp_dump={tmp_dump}",
        f"export_video={_bool_value(validation.get('export_video', True))}",
        f"export_mesh={_bool_value(validation.get('export_mesh', False))}",
        f"save_img={_bool_value(validation.get('save_img', True))}",
        f"save_ply={_bool_value(validation.get('save_ply', False))}",
        f"render_fps={int(validation.get('render_fps', 30))}",
        f"motion_video_read_fps={int(validation.get('motion_video_read_fps', 30))}",
        f"style_image_path={style_image_path}",
        f"style_strength={float(validation.get('style_strength', 0.0))}",
        f"style_internal_appearance_strength={float(validation.get('style_internal_appearance_strength', 0.0))}",
        f"style_geometry_strength={float(geometry_strength)}",
        f"style_opacity_strength={float(validation.get('style_opacity_strength', 0.0))}",
        f"style_track_geometry={_bool_value(validation.get('style_track_geometry', True))}",
    ]


def _command_to_string(command):
    return " ".join(command)


def run_validation(cfg, limit=None, dry_run=False, geometry_only=True):
    validation = cfg.validation
    output_root = validation.get("output_dir", "exps/train_lam/stage3_geometry_aahq/validation")
    os.makedirs(output_root, exist_ok=True)

    style_list_path = validation.get(
        "selected_style_images",
        os.path.join(cfg.output_dir, "selected_style_images.txt"),
    )
    selected_paths = _read_style_paths(style_list_path, limit or validation.get("top_k", 3))
    cases = []

    for style_path in selected_paths:
        cases.append({
            "style_image_path": style_path,
            "style_id": _style_id(style_path),
            "case": "geom_off",
            "geometry_strength": 0.0,
            "command": _build_infer_command(cfg, style_path, "geom_off", 0.0),
        })
        cases.append({
            "style_image_path": style_path,
            "style_id": _style_id(style_path),
            "case": "geom_on",
            "geometry_strength": float(validation.get("style_geometry_strength", 1.0)),
            "command": _build_infer_command(
                cfg,
                style_path,
                "geom_on",
                float(validation.get("style_geometry_strength", 1.0)),
            ),
        })

    summary = {
        "stage": "stage3_geometry_validation",
        "dry_run": bool(dry_run),
        "geometry_only": bool(geometry_only),
        "selected_style_images": style_list_path,
        "num_style_refs": len(selected_paths),
        "num_cases": len(cases),
        "output_dir": output_root,
        "cases": [],
    }

    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", os.path.abspath(".cache/matplotlib"))
    os.makedirs(env["MPLCONFIGDIR"], exist_ok=True)

    for case in cases:
        command = case["command"]
        log_dir = os.path.join(output_root, case["style_id"], case["case"])
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "infer.log")
        row = {
            "style_id": case["style_id"],
            "style_image_path": case["style_image_path"],
            "case": case["case"],
            "geometry_strength": case["geometry_strength"],
            "log_path": log_path,
            "command": command,
            "command_string": _command_to_string(command),
        }

        if dry_run:
            row["status"] = "dry_run"
            summary["cases"].append(row)
            print(row["command_string"])
            continue

        print(f"[stage3-validate] {case['style_id']} {case['case']}", flush=True)
        started = time.time()
        with open(log_path, "w") as log_file:
            proc = subprocess.run(command, cwd=os.getcwd(), env=env, stdout=log_file, stderr=subprocess.STDOUT)
        row["seconds"] = round(time.time() - started, 3)
        row["return_code"] = int(proc.returncode)
        row["status"] = "ok" if proc.returncode == 0 else "failed"
        summary["cases"].append(row)

        if proc.returncode != 0:
            print(f"[stage3-validate] failed; see {log_path}", flush=True)
            break

    summary_path = os.path.join(output_root, "validation_summary.json" if not dry_run else "validation_dry_run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"validation_summary={summary_path}")
    print(f"num_style_refs={summary['num_style_refs']}")
    print(f"num_cases={summary['num_cases']}")
    if not dry_run:
        print(f"num_failed={sum(1 for case in summary['cases'] if case.get('status') == 'failed')}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3 geometry-off/on validation inference.")
    parser.add_argument("--config", default="configs/train/lam_style_stage3_geometry_aahq.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Number of selected style refs to validate.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    run_validation(cfg, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
