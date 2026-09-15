import argparse
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np


ROOT = os.path.dirname(os.path.dirname(__file__))
EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
MOUTH_PAIRS = [(48, 54), (49, 53), (50, 52), (59, 55), (58, 56), (60, 64), (61, 63), (67, 65)]


def _content_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def _build_command(args, emotion, out_dir, seed):
    return [
        args.python,
        "-m",
        "lam.launch",
        "infer.lam",
        "--config",
        args.config,
        f"model_name={args.model_name}",
        f"image_input={args.image_input}",
        f"motion_seqs_dir={args.motion_seqs_dir}",
        "export_video=false",
        "export_mesh=true",
        "test_sample=true",
        f"style_image_path={args.style_image}",
        "style_optimize_color=true",
        "style_optimize_cache=false",
        f"style_optimize_steps={args.style_optimize_steps}",
        f"style_optimize_num_views={args.style_optimize_num_views}",
        f"style_geometry_strength={args.style_geometry_strength}",
        "style_track_geometry=true",
        f"emotion_class={emotion}",
        f"emotion_optimize={str(args.emotion_optimize).lower()}",
        "emotion_safety=true",
        f"emotion_final_symmetry={args.emotion_final_symmetry}",
        f"gs_mouth_symmetry={args.gs_mouth_symmetry}",
        f"seed={seed}",
        "save_img=true",
        f"image_dump={out_dir}",
        f"video_dump={out_dir}_video",
    ]


def _score_mouth_shape(image_path, tracker, tag):
    scratch_path = os.path.join("/tmp", "lam_emotion_best_of_n", f"{tag}.png")
    os.makedirs(os.path.dirname(scratch_path), exist_ok=True)
    shutil.copy(image_path, scratch_path)

    return_code = tracker.preprocess(scratch_path)
    if return_code != 0:
        return None

    landmarks_path = os.path.join(tracker.output_dir, "preprocess", tag, "landmark2d", "landmarks.npz")
    if not os.path.exists(landmarks_path):
        return None

    landmarks = np.load(landmarks_path, allow_pickle=True)["face_landmark_2d"][0][:, :2] * 1024
    top = landmarks[27]
    chin = landmarks[8]
    axis = chin - top
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    normal = np.array([axis[1], -axis[0]])
    width = np.linalg.norm(landmarks[54] - landmarks[48]) + 1e-9

    residuals = []
    for left, right in MOUTH_PAIRS:
        residuals.append(abs(np.dot(landmarks[left] - top, normal) + np.dot(landmarks[right] - top, normal)) / width)
        residuals.append(abs(np.dot(landmarks[left] - top, axis) - np.dot(landmarks[right] - top, axis)) / width)
    return float(np.mean(residuals) * 100.0)


def _make_tracker(output_dir):
    from tools.flame_tracking_single_image import FlameTrackingSingleImage

    return FlameTrackingSingleImage(
        output_dir=output_dir,
        alignment_model_path="./model_zoo/flame_tracking_models/68_keypoints_model.pkl",
        vgghead_model_path="./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd",
        human_matting_path="./model_zoo/flame_tracking_models/matting/stylematte_synth.pt",
        facebox_model_path="./model_zoo/flame_tracking_models/FaceBoxesV2.pth",
        detect_iris_landmarks=False,
        args=SimpleNamespace(output_dir=output_dir, config_name="alignment", blender_path=None),
    )


def main():
    parser = argparse.ArgumentParser(description="Render each emotion multiple times and keep the best mouth-shape result.")
    parser.add_argument("image_input")
    parser.add_argument("style_image")
    parser.add_argument("--output_dir", default="emotion_best_of_n")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emotions", nargs="*", default=EMOTIONS)
    parser.add_argument("--python", default="./lam_env/bin/python")
    parser.add_argument("--config", default="configs/inference/lam-20k-8gpu.yaml")
    parser.add_argument("--model_name", default="exps/train_lam/style_appearance_e2e_v8/model_final.pt")
    parser.add_argument("--motion_seqs_dir", default="assets/sample_motion/export/Look_In_My_Eyes/")
    parser.add_argument("--style_optimize_steps", type=int, default=100)
    parser.add_argument("--style_optimize_num_views", type=int, default=4)
    parser.add_argument("--style_geometry_strength", type=float, default=1.0)
    parser.add_argument("--emotion_optimize", action="store_true")
    parser.add_argument("--emotion_final_symmetry", type=float, default=0.5)
    parser.add_argument("--gs_mouth_symmetry", type=float, default=0.0)
    args = parser.parse_args()

    os.chdir(ROOT)
    os.makedirs(args.output_dir, exist_ok=True)
    tracker = _make_tracker(os.path.join(args.output_dir, "_mouth_score_tracking"))
    cid = _content_id(args.image_input)
    summary = {}

    for emotion in args.emotions:
        best = None
        summary[emotion] = []
        for attempt in range(args.attempts):
            seed = args.seed + attempt
            attempt_dir = os.path.join(args.output_dir, "attempts", f"{emotion}_seed{seed}")
            cmd = _build_command(args, emotion, attempt_dir, seed)
            print(f"[{emotion}] attempt {attempt + 1}/{args.attempts} seed={seed}", flush=True)
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
            subprocess.run(cmd, check=True, env=env)

            result_path = os.path.join(attempt_dir, cid, "stylized_preview.png")
            score = _score_mouth_shape(result_path, tracker, f"{emotion}_seed{seed}") if os.path.exists(result_path) else None
            row = {"seed": seed, "result": result_path, "mouth_shape_score": score}
            summary[emotion].append(row)
            if score is not None and (best is None or score < best["mouth_shape_score"]):
                best = row

        if best is None:
            print(f"[{emotion}] no scoreable result", flush=True)
            continue

        final_dir = os.path.join(args.output_dir, "best", emotion, cid)
        os.makedirs(final_dir, exist_ok=True)
        final_path = os.path.join(final_dir, "stylized_preview.png")
        shutil.copy(best["result"], final_path)
        summary[emotion + "_best"] = {**best, "selected_result": final_path}
        print(f"[{emotion}] selected seed={best['seed']} score={best['mouth_shape_score']:.2f} -> {final_path}", flush=True)

    summary_path = os.path.join(args.output_dir, "best_of_n_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary={summary_path}")


if __name__ == "__main__":
    sys.exit(main())
