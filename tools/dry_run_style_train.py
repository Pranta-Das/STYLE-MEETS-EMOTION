import argparse
import os
from pathlib import Path
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class StyleImageDataset(Dataset):
    def __init__(self, image_glob_list, image_size=256):
        # image_glob_list: list of glob patterns or paths
        paths = []
        for p in image_glob_list:
            paths.extend(sorted(Path('.').glob(p)))
        self.image_paths = [str(p) for p in paths]
        self.image_size = image_size

    def __len__(self):
        return max(0, len(self.image_paths))

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            im = ImageOps.fit(im, (self.image_size, self.image_size))
            arr = np.asarray(im, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2,0,1)
        return {'image': tensor, 'path': path, 'name': os.path.basename(path)}

class TinyStyleAutoEncoder(nn.Module):
    def __init__(self, base_channels=24, latent_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels*2, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, image):
        return self.encoder(image)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, image):
        return self.decode(self.encode(image))


def run_train(image_globs, epochs=2, batch_size=2, out_dir='exps/dry_run'):
    dataset = StyleImageDataset(image_globs, image_size=256)
    if len(dataset) == 0:
        print('No images found for', image_globs)
        return 1
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyStyleAutoEncoder().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    os.makedirs(out_dir, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            imgs = batch['image'].to(device)
            recon = model(imgs)
            loss = l1(recon, imgs) + 0.1 * mse(recon, imgs)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        print(f'Epoch {epoch+1}/{epochs} mean_loss={sum(losses)/len(losses):.6f}')
    # save checkpoint
    torch.save({'model': model.state_dict()}, os.path.join(out_dir, 'dry_style_autoencoder.pt'))
    print('Saved checkpoint to', out_dir)
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--globs', nargs='+', default=['data/natural_faces/*.png'])
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--out_dir', type=str, default='exps/dry_run')
    args = parser.parse_args()
    run_train(args.globs, epochs=args.epochs, batch_size=args.batch_size, out_dir=args.out_dir)
