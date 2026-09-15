#!/usr/bin/env python3
"""Best-of-N emotion renders: generate each emotion several times, keep the
cleanest one.

Why this exists rather than another "fix": the pipeline is NONDETERMINISTIC.
Three runs with identical config and identical seed=42 produce three
different images (verified by md5), because the Gaussian rasterizer's
blending order is not reproducible. Measured spread on one case: lip shape
asymmetry 5.08 / 6.54 / 5.25, sd 0.80. So good renders already occur; the
problem is that you cannot ask for one directly.

Each candidate is scored on two things:
  * whether an independent expression classifier still reads the intended
    emotion (a beautiful render of the wrong expression is useless), and
  * lip SHAPE asymmetry -- reflect the mouth contour about the face's own
    nose-bridge->chin midline and compare it to itself, normalised by mouth
    width. Source photographs score ~1.8 on this; renders average ~6.2.
    This is deliberately NOT the lip-centroid measure used earlier in this
    project, which is blind to a lip that is smeared or malformed while its
    average point sits centred.

Candidates that match the requested emotion always beat candidates that do
not; within each group the lowest asymmetry wins.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# 68-landmark mirror pairs across the mouth
MOUTH_PAIRS = [(48, 54), (49, 53), (50, 52), (59, 55), (58, 56),
               (60, 64), (61, 63), (67, 65)]
VIT_LABELS = {"angry": "anger", "disgust": "disgust", "fear": "fear", "happy": "happy",
              "neutral": "neutral", "sad": "sad", "surprise": "surprise"}


def render(image_input, style_image, emotion, out_dir, extra_args, attempt):
    """One candidate render. Returns the produced image path, or None."""
    content_id = os.path.splitext(os.path.basename(image_input))[0]
    dump = os.path.join(out_dir, "_candidates", f"{emotion}_{attempt}")
    cmd = [
        "lam_env/bin/python", "-m", "lam.launch", "infer.lam",
        "--config", "configs/inference/lam-20k-8gpu.yaml",
        "model_name=exps/train_lam/style_appearance_e2e_v8/model_final.pt",
        f"image_input={image_input}",
        "motion_seqs_dir=assets/sample_motion/export/Look_In_My_Eyes/",
        "export_video=false", "export_mesh=true", "test_sample=true",
        f"style_image_path={style_image}",
        "style_optimize_color=true", "style_optimize_steps=100",
        "style_optimize_num_views=4",
        "style_geometry_strength=1.0", "style_track_geometry=true",
        f"emotion_class={emotion}", "emotion_optimize=true",
        "save_img=true",
        f"image_dump={dump}", f"video_dump={dump}_v",
    ] + list(extra_args)
    env = dict(os.environ, PYTHONPATH=".", CUDA_VISIBLE_DEVICES="0")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    src = os.path.join(dump, content_id, "stylized_preview.png")
    if proc.returncode != 0 or not os.path.exists(src):
        return None
    keep = os.path.join(out_dir, "_candidates", f"{emotion}_{attempt}.png")
    os.makedirs(os.path.dirname(keep), exist_ok=True)
    shutil.move(src, keep)
    shutil.rmtree(dump, ignore_errors=True)
    shutil.rmtree(dump + "_v", ignore_errors=True)
    return keep


class Scorer:
    """Loads the landmark tracker and the judge classifier once."""

    def __init__(self):
        sys.path.insert(0, ROOT)
        from tools.flame_tracking_single_image import FlameTrackingSingleImage
        from types import SimpleNamespace
        from transformers import pipeline
        self._track_dir = "tracking_output_bestofn"
        self.tracker = FlameTrackingSingleImage(
            output_dir=self._track_dir, detect_iris_landmarks=False,
            args=SimpleNamespace(output_dir=self._track_dir,
                                 config_name="alignment", blender_path=None),
        )
        self.clf = pipeline("image-classification",
                            model="trpakov/vit-face-expression", device=-1)

    def lip_asymmetry(self, path, tag):
        staged = os.path.join(self._track_dir, "_staged", f"{tag}.png")
        os.makedirs(os.path.dirname(staged), exist_ok=True)
        shutil.copy(path, staged)
        self.tracker.preprocess(staged)
        lmk_path = os.path.join(self._track_dir, "preprocess", tag,
                                "landmark2d", "landmarks.npz")
        if not os.path.exists(lmk_path):
            return None
        lmk = np.load(lmk_path, allow_pickle=True)["face_landmark_2d"][0][:, :2] * 1024
        top, chin = lmk[27], lmk[8]
        axis = chin - top
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        normal = np.array([axis[1], -axis[0]])
        width = np.linalg.norm(lmk[54] - lmk[48]) + 1e-9
        res = []
        for a, b in MOUTH_PAIRS:
            res.append(abs(np.dot(lmk[a] - top, normal) + np.dot(lmk[b] - top, normal)) / width)
            res.append(abs(np.dot(lmk[a] - top, axis) - np.dot(lmk[b] - top, axis)) / width)
        return float(np.mean(res) * 100)

    def emotion_match(self, path, emotion):
        preds = self.clf(path, top_k=7)
        top1 = VIT_LABELS[max(preds, key=lambda p: p["score"])["label"]]
        score = next(p["score"] for p in preds if VIT_LABELS[p["label"]] == emotion)
        return top1 == emotion, float(score)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image_input")
    ap.add_argument("style_image")
    ap.add_argument("out_dir")
    ap.add_argument("-n", "--attempts", type=int, default=4,
                    help="candidate renders per emotion (default 4)")
    ap.add_argument("--emotions", default=",".join(EMOTIONS))
    ap.add_argument("--keep-candidates", action="store_true",
                    help="keep every candidate instead of only the winners")
    # Any leftover key=value arguments are forwarded to the runner untouched,
    # so callers can override any inference config option without this script
    # needing to know about it.
    args, extra = ap.parse_known_args()
    args.extra = [a for a in extra if "=" in a]
    unknown = [a for a in extra if "=" not in a]
    if unknown:
        ap.error(f"unrecognized arguments: {' '.join(unknown)}")

    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    scorer = Scorer()

    print(f"best-of-{args.attempts} over {len(emotions)} emotions "
          f"({args.attempts * len(emotions)} renders)\n", flush=True)
    summary = []
    for emotion in emotions:
        cands = []
        for attempt in range(args.attempts):
            t0 = time.time()
            path = render(args.image_input, args.style_image, emotion,
                          args.out_dir, args.extra, attempt)
            if path is None:
                print(f"  {emotion} #{attempt}: render FAILED", flush=True)
                continue
            asym = scorer.lip_asymmetry(path, f"{emotion}_{attempt}")
            matched, conf = scorer.emotion_match(path, emotion)
            cands.append({"path": path, "asym": asym, "match": matched, "conf": conf})
            print(f"  {emotion} #{attempt}: asym "
                  f"{'n/a' if asym is None else f'{asym:5.2f}'}  "
                  f"{'match' if matched else 'MISS '} {conf*100:5.1f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if not cands:
            print(f"{emotion}: no usable candidate\n", flush=True)
            continue
        # Matched candidates first, ranked by cleanest mouth. Among UNMATCHED
        # candidates, rank by how strongly the target emotion reads instead --
        # ranking those by asymmetry too would reward the least expressive
        # render, since a barely-moved face is trivially symmetric. Observed
        # directly: 'anger' with no matching candidate was picked at
        # asymmetry 1.76 (better than the source photo) precisely because the
        # winner was an almost neutral face, which is useless as an anger
        # figure.
        cands.sort(key=lambda c: (
            (0, c["asym"] if c["asym"] is not None else 1e9) if c["match"]
            else (1, -c["conf"])
        ))
        best = cands[0]
        final = os.path.join(args.out_dir, f"{emotion}.png")
        shutil.copy(best["path"], final)
        summary.append((emotion, best, len(cands)))
        print(f"{emotion}: picked asym {best['asym']:.2f} "
              f"({'match' if best['match'] else 'no match'}) -> {final}\n", flush=True)

        if not args.keep_candidates:
            for c in cands:
                os.remove(c["path"])

    if not args.keep_candidates:
        shutil.rmtree(os.path.join(args.out_dir, "_candidates"), ignore_errors=True)

    print("=" * 58)
    print(f"{'emotion':10s} {'picked asym':>12s} {'match':>7s} {'conf':>7s}")
    for emotion, best, _ in summary:
        print(f"{emotion:10s} {best['asym']:12.2f} "
              f"{'yes' if best['match'] else 'no':>7s} {best['conf']*100:6.1f}%")
    if summary:
        vals = [b["asym"] for _, b, _ in summary]
        print(f"\nmean picked asymmetry {np.mean(vals):.2f} "
              f"(source photographs score ~1.8)")
        print(f"emotion match {sum(1 for _, b, _ in summary if b['match'])}/{len(summary)}")


if __name__ == "__main__":
    main()
