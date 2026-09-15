# LAM Three-Stage Style + Geometry Pipeline - Status & Results

## Pipeline Overview
This document tracks the progress of the LAM style/geometry pipeline:
- **Stage 1**: Train style autoencoder on CelebA-HQ natural faces.
- **Stage 2**: Fine-tune style autoencoder on aligned AAHQ artistic faces.
- **Stage 3**: Build a FLAME geometry reference bank from tracked AAHQ styles for full LAM inference.
- **Stage 4**: Fine-tune the full LAM style geometry adapter for stronger learned query-point deformation.

Full pipeline details are in `STYLE_GEOMETRY_PIPELINE.md`.

## Datasets
- **CelebA-HQ**: `./celeba_hq_256/` - 30,000 256x256 face images
- **AAHQ**: `./ffhq-dataset/aahq-dataset/raw/` - 23,763 raw artistic face images (being aligned)

## Configurations

### Stage 1: Natural Face Training
- **Config**: `configs/train/lam_style_stage1_celeba.yaml`
- **Dataset**: CelebA-HQ 30,000 images
- **Epochs**: 100
- **Batch Size**: 4
- **Image Size**: 256x256
- **Learning Rates**: 
  - Style Adapter: 5.0e-4
  - Other Modules: 1.0e-6
- **Frozen Modules**: Encoder, Transformer, Renderer
- **Output**: `exps/train_lam/stage1_celeba/`

### Stage 2: Artistic Style Fine-tuning
- **Config**: `configs/train/lam_style_stage2_aahq.yaml`
- **Dataset**: AAHQ aligned images (in progress)
- **Epochs**: 30
- **Batch Size**: 2
- **Image Size**: 256x256
- **Learning Rates**:
  - Style Adapter: 2.0e-4
  - Other Modules: 1.0e-5
- **Frozen Modules**: Encoder only
- **Input Model**: Stage 1 checkpoint
- **Output**: `exps/train_lam/stage2_aahq/`

### Stage 3: Geometry Reference Bank
- **Config**: `configs/train/lam_style_stage3_geometry_aahq.yaml`
- **Input**: tracked FLAME params from `tracking_output/export/*/canonical_flame_param.npz`
- **Dataset**: selected tracked AAHQ style references
- **Output**: `exps/train_lam/stage3_geometry_aahq/`
- **Purpose**: choose style images whose FLAME shape is meaningfully different from the source identity, then validate geometry-only full LAM inference.
- **Tracking Tool**: `tools/track_stage3_geometry_refs.py`
- **Bank Tool**: `tools/build_style_geometry_bank.py`

### Stage 4: Full-LAM Geometry Adapter Fine-Tuning
- **Config**: `configs/train/lam_style_stage4_geometry_adapter_aahq.yaml`
- **Input Model**: `model_zoo/lam_models/releases/lam/lam-20k/step_045500/`
- **Input References**: `exps/train_lam/stage3_geometry_aahq/selected_geometry_refs.json`
- **Trainable Module**: `ModelLAM.style_adapter.geometry_mlp`
- **Frozen Modules**: encoder, transformer, renderer, appearance adapter
- **Loss**: Smooth L1 query-point target loss plus small offset regularization
- **Shape Target Dim**: full 300-D tracked FLAME shape for query-point targets
- **Output**: `exps/train_lam/stage4_geometry_adapter_aahq/`

## Running Processes

### Pipeline Script
```bash
bash run_two_stage_training.sh
```
Location: `./run_two_stage_training.sh`

### Background Tasks
1. **Stage 1 Training**: Running with CelebA-HQ dataset
2. **AAHQ Alignment**: Running with 8 workers (in parallel)
   - Aligns ~23,763 artistic face images
   - Output: `ffhq-dataset/aahq-dataset/aligned/`

## Expected Output Structure

```
exps/train_lam/
├── stage1_celeba/
│   ├── style_autoencoder_final.pt     # Final trained model
│   ├── style_train_summary.json        # Training metadata
│   └── samples/
│       ├── style_inputs_grid.png
│       ├── style_reconstructions_grid.png
│       ├── style_training_grid.png
│       ├── style_average.png
│       └── style_latent_blends.png
│
└── stage2_aahq/
    ├── style_autoencoder_final.pt     # Fine-tuned model
    ├── style_train_summary.json
    └── samples/
        ├── style_inputs_grid.png
        ├── style_reconstructions_grid.png
        ├── style_training_grid.png
        ├── style_average.png
        └── style_latent_blends.png
```

## Key Outputs

### Style Autoencoder Checkpoints
- **Stage 1**: `exps/train_lam/stage1_celeba/style_autoencoder_final.pt`
- **Stage 2**: `exps/train_lam/stage2_aahq/style_autoencoder_final.pt`

### Training Summaries
- **Stage 1**: `exps/train_lam/stage1_celeba/style_train_summary.json`
- **Stage 2**: `exps/train_lam/stage2_aahq/style_train_summary.json`

### Sample Visualizations
Grid images showing:
- Original style images used for training
- Reconstructions from the autoencoder
- Combined training grid
- Style interpolation/blending effects

## Current Status Before Stage 2

- **Stage 1 checkpoint exists**: `exps/train_lam/stage1_celeba/style_autoencoder_final.pt`
- **Stage 1 summary exists**: `exps/train_lam/stage1_celeba/style_train_summary.json`
- **Stage 1 actual training images**: 5,004 images from `celeba_hq_256/`
- **Stage 1 epochs completed**: 15
- **Stage 1 final loss**: 0.011877
- **AAHQ raw images currently present**: 4,320 files in `ffhq-dataset/aahq-dataset/raw/`
- **AAHQ aligned images currently present**: 1,982 PNG files in `ffhq-dataset/aahq-dataset/aligned/`
- **Stage 2 checkpoint exists**: `exps/train_lam/stage2_aahq/style_autoencoder_final.pt`
- **Stage 2 summary exists**: `exps/train_lam/stage2_aahq/style_train_summary.json`
- **Stage 2 actual training images**: 1,982 aligned AAHQ images
- **Stage 2 epochs completed**: 15
- **Stage 2 final loss**: 0.012351
- **Stage 3 geometry bank exists**: `exps/train_lam/stage3_geometry_aahq/geometry_bank_summary.json`
- **Stage 3 selected references**: `exps/train_lam/stage3_geometry_aahq/selected_geometry_refs.json`
- **Stage 3 selected style images list**: `exps/train_lam/stage3_geometry_aahq/selected_style_images.txt`
- **Stage 3 dry-run candidate list exists**: `exps/train_lam/stage3_geometry_aahq/tracking_dry_run_summary.json`
- **Stage 3 dry-run selected candidates**: 24 AAHQ images from 1,982 aligned images
- **Stage 3 validation tool**: `tools/run_stage3_geometry_validation.py`
- **Stage 3 validation launcher**: `run_stage3_geometry_validation.sh`
- **Stage 3 validation dry run**: `exps/train_lam/stage3_geometry_aahq/validation/validation_dry_run_summary.json`
- **Stage 4 config exists**: `configs/train/lam_style_stage4_geometry_adapter_aahq.yaml`
- **Stage 4 launcher exists**: `run_stage4_geometry_adapter.sh`
- **Stage 4 checkpoint exists**: `exps/train_lam/stage4_geometry_adapter_aahq/model_final.pt`
- **Stage 4 adapter checkpoint exists**: `exps/train_lam/stage4_geometry_adapter_aahq/style_geometry_adapter.pt`
- **Stage 4 summary exists**: `exps/train_lam/stage4_geometry_adapter_aahq/geometry_adapter_train_summary.json`
- **Stage 4 epochs completed**: 20
- **Stage 4 final loss**: 0.000064

Important: Stage 2 should train from `ffhq-dataset/aahq-dataset/aligned/*.png`, not directly from `raw/`. The raw images are useful if you want to continue alignment first.

## Stage 2 Terminal Runbook

### 1. Go to the project root
```bash
cd /media/uvll/8d1366b3-c06e-4683-8aee-cf9759a4008c/LAM13
```

### 2. Check that Stage 1 and AAHQ aligned data are ready
```bash
ls -lh exps/train_lam/stage1_celeba/style_autoencoder_final.pt
find ffhq-dataset/aahq-dataset/aligned -maxdepth 1 -type f -iname '*.png' | wc -l
```

You already have more than 1,000 aligned images, so Stage 2 can start now.

### 3. Optional: continue AAHQ alignment first
Run this only if you want to align more of the raw dataset before Stage 2.

```bash
cd ffhq-dataset/aahq-dataset
../../lam_env/bin/python face_alignment.py \
  --json_dir AAHQ-dataset.json \
  --raw_dir raw \
  --save_dir aligned \
  --n_worker 8
cd ../..
```

Then re-check:
```bash
find ffhq-dataset/aahq-dataset/aligned -maxdepth 1 -type f -iname '*.png' | wc -l
```

### 4. Start Stage 2 training
```bash
mkdir -p logs .cache/matplotlib
export PYTHONPATH=.
export MPLCONFIGDIR="$PWD/.cache/matplotlib"

nohup lam_env/bin/python lam/launch.py train.lam \
  --config configs/train/lam_style_stage2_aahq.yaml \
  > logs/stage2_aahq.log 2>&1 &

echo $! > run_stage2.pid
```

### 5. Watch progress
```bash
tail -f logs/stage2_aahq.log
```

Expected first useful lines:
```text
[LAM] style training start: ... images, batch_size=2, num_workers=4, epochs=15
[LAM] loaded style checkpoint from ./exps/train_lam/stage1_celeba/style_autoencoder_final.pt
[LAM] START EPOCH 1/15
```

### 6. Check final Stage 2 outputs
```bash
ls -lh exps/train_lam/stage2_aahq/
ls -lh exps/train_lam/stage2_aahq/samples/
cat exps/train_lam/stage2_aahq/style_train_summary.json
```

Expected final files:
- `exps/train_lam/stage2_aahq/style_autoencoder_final.pt`
- `exps/train_lam/stage2_aahq/style_train_summary.json`
- `exps/train_lam/stage2_aahq/samples/style_inputs_grid.png`
- `exps/train_lam/stage2_aahq/samples/style_reconstructions_grid.png`
- `exps/train_lam/stage2_aahq/samples/style_training_grid.png`
- `exps/train_lam/stage2_aahq/samples/style_average.png`
- `exps/train_lam/stage2_aahq/samples/style_latent_blends.png`

## Monitoring

Check alignment progress:
```bash
find ffhq-dataset/aahq-dataset/aligned -maxdepth 1 -type f -iname '*.png' | wc -l
```

Check training logs:
```bash
ls -lh exps/train_lam/stage1_celeba/
ls -lh exps/train_lam/stage2_aahq/
```

## Notes

- **CPU Training**: Currently running on CPU (no CUDA detected)
- **Failed Downloads**: ~20 AAHQ images failed to download (noted in alignment log)
- **Actual Dataset Count**: ~23,740+ images available after download failures
- **Training Time**: Stage 1 (100 epochs on 30K images) may take several hours on CPU
- **Pipeline Logic**: Stage 2 waits for alignment to reach 1000+ images before starting

## Next Steps After Completion

1. Validate learned style representations using the visual samples
2. Validate Stage 3 geometry references with color disabled and `style_geometry_strength=1.0`
3. Use selected Stage 3 style references in downstream LAM inference
4. Compare `style_geometry_strength=0.0` vs `1.0` outputs to confirm true shape changes
5. If FLAME-reference geometry is still too weak, run Stage 4 full-LAM geometry-adapter fine-tuning:

```bash
bash run_stage4_geometry_adapter.sh
tail -f logs/stage4_geometry_adapter_aahq.log
```
