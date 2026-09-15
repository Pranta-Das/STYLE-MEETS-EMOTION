# Lightweight style-image training runner

Run the local style-image training path with:

```bash
./lam_env/bin/python -m lam.launch train.lam --config configs/train/lam_minimal.yaml
```

The minimal config trains a tiny denoising autoencoder on:

- `style_image.png`
- `style_image1.png`
- `style_image2.png`
- `style_image3.png`

Outputs are written to `exps/train_lam`:

- `style_autoencoder_final.pt`
- `style_train_summary.json`
- `samples/style_inputs_grid.png`
- `samples/style_reconstructions_grid.png`
- `samples/style_training_grid.png`
- `samples/style_average.png`
- `samples/style_latent_blends.png`

The original full LAM fine-tuning path remains available by setting `training_mode: full_lam` and providing the normal video-head training dataset plus model config.
