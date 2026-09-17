# STYLE-MEETS-EMOTION

Reference-guided stylization and discrete emotion control for single-image
3D Gaussian head avatars.

Given one portrait, one style reference image, and one of seven emotion labels,
the system produces an animatable 3D Gaussian avatar that adopts the style
reference's appearance and geometry while playing the requested emotion. The
two controls are selected independently and neither module is retrained.

| control | acts on |
|---|---|
| geometric style | FLAME identity shape + canonical Gaussian query points |
| appearance style | decoded Gaussian RGB, then image-space LAB statistics |
| emotion | FLAME expression and jaw parameters |

Because style and emotion write to disjoint parameter subspaces, either can be
disabled without changing the other's interface. Emotion classes are
`anger`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

## Install

CUDA 12.1 or 11.8, Python 3.10:

```bash
./scripts/install/install_cu121.sh      # or install_cu118.sh
```

Windows: see [scripts/install/WINDOWS_INSTALL.md](scripts/install/WINDOWS_INSTALL.md).

## Weights

we added the drive link.

| file | size | where |
|---|---|---|
| `style_appearance_e2e_v8/model_final.pt` | 2.25 GB | *(https://drive.google.com/file/d/1RITKCi4BTDq4BDNcYbPSGgWdU_45nkZO/view?usp=sharing)* |
| `emotion_adapter_symmetric/emotion_adapter.pt` | 0.2 MB | *(https://drive.google.com/file/d/1fM6lCL2NkUBe6-jn9YssasYLQfFpv1Y_/view?usp=sharing)* |

Place them under `exps/train_lam/` matching those paths. The base LAM backbone
and FLAME assets go in `model_zoo/`; see the upstream LAM instructions.

## Usage

Stills and videos for all seven emotions on one content/style pair:

```bash
./run_emotion_style_check.sh <input_image> <style_image> [out_dir]
```

Useful switches:

```bash
EMOTIONS="happy sad"  ./run_emotion_style_check.sh ...   # subset of classes
FULL=true             ./run_emotion_style_check.sh ...   # real-time video, all frames
STEPS=30              ./run_emotion_style_check.sh ...   # faster, weaker refinement
```

By default the video renders 50 frames sampled from the driving clip's 519 and
writes them at 30 fps, so playback is ~10x fast-forward. `FULL=true` renders
every frame. The still is identical either way.

Emotion only, no style:

```bash
./run_7emotions.sh <input_image> [out_dir]
```


## Relationship to LAM

This is a modified derivative of [LAM](https://github.com/aigc3d/LAM)
(Apache License 2.0). The LAM backbone, FLAME rendering stack, VHAP tracking,
and landmark/matting utilities are upstream code, retained under that license.

Files modified or added by this work include `lam/stylization/` (new),
`lam/models/emotion_adapter.py` (new), `lam/models/motion_symmetry.py` (new),
`lam/runners/train/` (new), `lam/datasets/style_pairs.py` (new),
`configs/train/` (new), several `tools/` scripts, and changes to
`lam/runners/infer/lam.py`, `lam/models/modeling_lam.py`,
`lam/models/rendering/gs_renderer.py`, and `lam/runners/infer/head_utils.py`.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Result

we added the drive link.

| input | style image | video/image |
|---|---|---|
| (https://drive.google.com/file/d/1Kpnl4nVNiUGVFOTcCdxNaY2lrPpOHC4n/view?usp=sharing) | (https://drive.google.com/file/d/1Yx_K71nXY8VRF642N-8IC0bgygOc0v7T/view?usp=sharing) | *(https://drive.google.com/drive/folders/1EwnajYMEsP9XPiZwAFAsbqTD-XaSihA-?usp=sharing)* |
| (https://drive.google.com/file/d/1Kpnl4nVNiUGVFOTcCdxNaY2lrPpOHC4n/view?usp=sharing) | (https://drive.google.com/file/d/1cRr1b0BmxauUs59D9zKdrbpKMhxuV-B2/view?usp=sharing) | *(https://drive.google.com/drive/folders/1mEN7tfkaR6JzxK7sHA8uW3NqjOTimlmc?usp=sharing)* |




## Citation

____
