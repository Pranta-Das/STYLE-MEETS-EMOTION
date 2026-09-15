# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors. All rights reserved.

import argparse
import glob
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lam.runners import REGISTRY_RUNNERS
from lam.runners.abstract import Runner

logger = get_logger(__name__)


def _resample_bicubic():
    if hasattr(Image, 'Resampling'):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def _style_path_sort_key(path):
    name = os.path.basename(path)
    match = re.fullmatch(r'style_image(\d*)\.png', name)
    if match:
        suffix = match.group(1)
        return 0, int(suffix) if suffix else 0, name
    return 1, name


def _tensor_to_pil(image):
    image = image.detach().cpu().clamp(0, 1)
    array = image.permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode='RGB')


def _save_grid(images, path, cols=None, gap=6, bg=(242, 242, 242)):
    if not images:
        return
    pil_images = [_tensor_to_pil(image) if torch.is_tensor(image) else image for image in images]
    width, height = pil_images[0].size
    cols = cols or len(pil_images)
    rows = (len(pil_images) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * width + (cols - 1) * gap, rows * height + (rows - 1) * gap), bg)
    for idx, image in enumerate(pil_images):
        x = (idx % cols) * (width + gap)
        y = (idx // cols) * (height + gap)
        canvas.paste(image, (x, y))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


class StyleImageDataset(Dataset):
    def __init__(self, image_paths, image_size):
        self.image_paths = list(image_paths)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            image = ImageOps.fit(
                image,
                (self.image_size, self.image_size),
                method=_resample_bicubic(),
                centering=(0.5, 0.5),
            )
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return {
            'image': tensor,
            'path': image_path,
            'name': os.path.basename(image_path),
        }


class TinyStyleAutoEncoder(nn.Module):
    def __init__(self, base_channels=24, latent_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels * 2, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
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


def gram_matrix(feat):
    b, c, h, w = feat.shape
    feat = feat.view(b, c, h * w)
    G = torch.bmm(feat, feat.transpose(1, 2))
    return G / (c * h * w)


def channelwise_mean_std(feat):
    mean = feat.mean(dim=(2, 3), keepdim=True)
    var = feat.var(dim=(2, 3), unbiased=False, keepdim=True)
    std = torch.sqrt(var + 1e-5)
    return mean, std


class VGGFeatureExtractor(nn.Module):
    """Frozen VGG19 feature extractor for content and style losses."""
    def __init__(self, layers=('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3')):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.layers = layers
        self.layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 17,
            'relu4_3': 26,
        }
        max_idx = max(self.layer_map[l] for l in layers)
        self.vgg = vgg[:max_idx + 1].eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        feats = {}
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            for name, target_idx in self.layer_map.items():
                if idx == target_idx and name in self.layers:
                    feats[name] = x
        return feats


class StyleTransferLoss(nn.Module):
    def __init__(self,
                 content_layers=('relu3_3',),
                 style_layers=('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'),
                 lambda_content=1.0,
                 lambda_style=1e5,
                 lambda_pixel=0.01,
                 lambda_tv=1e-4,
                 lambda_adain=1.0):
        super().__init__()
        all_layers = tuple(set(content_layers) | set(style_layers))
        self.vgg = VGGFeatureExtractor(layers=all_layers)
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.lambda_content = lambda_content
        self.lambda_style = lambda_style
        self.lambda_pixel = lambda_pixel
        self.lambda_tv = lambda_tv
        self.lambda_adain = lambda_adain
        self.l1 = nn.L1Loss()

    def _tv_loss(self, x):
        b, c, h, w = x.shape
        horiz = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean()
        vert = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()
        return horiz + vert

    def _adain_loss(self, output, style_ref):
        out_feats = self.vgg(output)
        style_feats = self.vgg(style_ref)
        loss = 0.0
        for layer in self.style_layers:
            out_feat = out_feats[layer]
            style_feat = style_feats[layer]
            out_mean, out_std = channelwise_mean_std(out_feat)
            style_mean, style_std = channelwise_mean_std(style_feat)
            loss = loss + F.mse_loss(out_mean, style_mean) + F.mse_loss(out_std, style_std)
        return loss / max(1, len(self.style_layers))

    def forward(self, output, original, style_ref):
        out_feats = self.vgg(output)
        content_feats = self.vgg(original)
        style_feats = self.vgg(style_ref)

        content_loss = sum(
            F.mse_loss(out_feats[l], content_feats[l]) for l in self.content_layers
        )
        style_loss = sum(
            F.mse_loss(gram_matrix(out_feats[l]), gram_matrix(style_feats[l]))
            for l in self.style_layers
        )
        pixel_loss = self.l1(output, original)
        adain_loss = self._adain_loss(output, style_ref)
        tv_loss = self._tv_loss(output)

        total = (
            self.lambda_content * content_loss +
            self.lambda_style * style_loss +
            self.lambda_pixel * pixel_loss +
            self.lambda_adain * adain_loss +
            self.lambda_tv * tv_loss
        )

        return total, {
            'content': content_loss.item(),
            'style': style_loss.item(),
            'adain': adain_loss.item(),
            'pixel': pixel_loss.item(),
            'tv': tv_loss.item(),
            'total': total.item(),
        }


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--root_dirs', type=str, default=None)
    parser.add_argument('--meta_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--style_image_path', type=str, default=None)
    parser.add_argument('--style_track_geometry', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    args, unknown = parser.parse_known_args()

    cli_cfg = OmegaConf.from_cli([item for item in unknown if '=' in item])
    cfg = OmegaConf.create()
    if args.config is not None:
        cfg = OmegaConf.load(args.config)
    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.root_dirs is not None:
        cfg.root_dirs = args.root_dirs
    if args.meta_path is not None:
        cfg.meta_path = args.meta_path
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.style_image_path is not None:
        cfg.style_image_path = args.style_image_path
    if args.style_track_geometry:
        cfg.style_track_geometry = True
    if args.seed is not None:
        cfg.seed = args.seed

    cfg.merge_with(cli_cfg)
    cfg.setdefault('logger', 'INFO')
    cfg.setdefault('training_mode', 'style_images')
    cfg.setdefault('epochs', 1)
    cfg.setdefault('batch_size', 1)
    cfg.setdefault('lr', 1e-4)
    cfg.setdefault('root_dirs', './train_data/vfhq_vhap/export')
    cfg.setdefault('meta_path', './train_data/vfhq_vhap/label/valid_id_train_list.json')
    cfg.setdefault('output_dir', 'exps/train_lam')
    cfg.setdefault('seed', 1234)
    cfg.setdefault('num_workers', 0)
    cfg.setdefault('sample_side_views', 7)
    cfg.setdefault('source_image_res', 256)
    cfg.setdefault('render_image_res', 256)
    cfg.setdefault('render_region_size', 256)
    cfg.setdefault('style_image_path', None)
    cfg.setdefault('style_image_paths', None)
    cfg.setdefault('style_image_glob', './style_image*.png')
    cfg.setdefault('style_track_geometry', False)
    cfg.setdefault('style_appearance_strength', 0.35)
    cfg.setdefault('style_geometry_strength', 0.25)
    cfg.setdefault('style_opacity_strength', 0.0)
    cfg.setdefault('model', {})
    cfg.setdefault('dataset', {})
    cfg.setdefault('save_interval', 1)
    cfg.setdefault('image_size', 128)
    cfg.setdefault('style_base_channels', 24)
    cfg.setdefault('style_latent_channels', 64)
    cfg.setdefault('noise_std', 0.03)
    cfg.setdefault('latent_blend_steps', 5)
    cfg.setdefault('style_content_loss_weight', 1.0)
    cfg.setdefault('style_style_loss_weight', 1e5)
    cfg.setdefault('style_pixel_loss_weight', 0.01)
    cfg.setdefault('style_adain_loss_weight', 1.0)
    cfg.setdefault('style_tv_loss_weight', 1e-4)
    cfg.setdefault('style_content_layers', ['relu3_3'])
    cfg.setdefault('style_loss_layers', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
    cfg.setdefault('source_image_path', 'assets/sample_input/status.png')
    cfg.setdefault('source_shape_path', 'tracking_output/export/status/canonical_flame_param.npz')
    cfg.setdefault('selected_geometry_refs', 'exps/train_lam/stage3_geometry_aahq/selected_geometry_refs.json')
    cfg.setdefault('shape_dims', 10)
    cfg.setdefault('adapter_offset_reg_weight', 0.01)
    cfg.setdefault('target_geometry_blend', 1.0)
    cfg.setdefault('style_adapter_max_query_offset', None)
    cfg.setdefault('content_refs_manifest', 'exps/train_lam/content_refs/tracked_content_refs.json')
    cfg.setdefault('bg_color', 1.0)
    cfg.setdefault('grad_accum_steps', 4)
    cfg.setdefault('sample_interval', 50)
    cfg.setdefault('identity_loss_weight', 0.1)
    cfg.setdefault('train_style_appearance_strength', 1.0)
    cfg.setdefault('train_style_opacity_strength', 0.0)
    return cfg


def render_view_with_grad(model, image, render_c2ws, render_intrs, render_bg_colors, flame_params,
                           style_image=None, style_shape_params=None,
                           style_appearance_strength=None, style_geometry_strength=None,
                           style_opacity_strength=None):
    """Differentiable equivalent of ModelLAM.infer_single_view.

    infer_single_view is decorated with @torch.no_grad() (inference-only), so
    it cannot be used to train style_adapter. This mirrors its body exactly,
    minus that decorator, so gradients flow from rendered pixels back into
    style_adapter.appearance_mlp/geometry_mlp.
    """
    import math
    from collections import defaultdict

    from einops import rearrange

    assert image.shape[0] == 1
    render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(render_intrs[0, 0, 0, 2] * 2)
    num_views = render_c2ws.shape[1]
    query_points = None
    if style_geometry_strength is None:
        style_geometry_strength = model.default_style_geometry_strength
    flame_params = model._blend_style_shape_params(flame_params, style_shape_params, style_geometry_strength)

    if model.latent_query_points_type.startswith("e2e_flame"):
        query_points, flame_params = model.renderer.get_query_points(flame_params, device=image.device)
    latent_points, image_feats, query_points, color_style = model.forward_latent_points(
        image[:, 0],
        camera=None,
        query_points=query_points,
        style_image=style_image,
        style_appearance_strength=style_appearance_strength,
        style_geometry_strength=style_geometry_strength,
        style_opacity_strength=style_opacity_strength,
    )
    image_feats_bchw = rearrange(image_feats, "b (h w) c -> b c h w", h=int(math.sqrt(image_feats.shape[1])))

    gs_model_list, query_points, flame_params, _ = model.renderer.forward_gs(
        gs_hidden_features=latent_points,
        query_points=query_points,
        flame_data=flame_params,
        additional_features={"image_feats": image_feats, "image": image[:, 0], "image_feats_bchw": image_feats_bchw},
        color_style=color_style,
    )

    render_res_list = []
    for view_idx in range(num_views):
        render_res = model.renderer.forward_animate_gs(
            gs_model_list,
            query_points,
            model.renderer.get_single_view_smpl_data(flame_params, view_idx),
            render_c2ws[:, view_idx:view_idx + 1],
            render_intrs[:, view_idx:view_idx + 1],
            render_h,
            render_w,
            render_bg_colors[:, view_idx:view_idx + 1],
        )
        render_res_list.append(render_res)

    out = defaultdict(list)
    for res in render_res_list:
        for k, v in res.items():
            out[k].append(v)
    for k, v in out.items():
        if isinstance(v[0], torch.Tensor):
            out[k] = torch.concat(v, dim=1)
            if k in ["comp_rgb", "comp_mask", "comp_depth"]:
                out[k] = out[k][0].permute(0, 2, 3, 1)
        else:
            out[k] = v
    out['cano_gs_lst'] = gs_model_list
    return out


@REGISTRY_RUNNERS.register('train.lam')
class LAMTrainer(Runner):
    EXP_TYPE: str = 'lam'

    def __init__(self):
        super().__init__()
        self.cfg = parse_configs()
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.seed_everything(self.cfg.seed)
        os.makedirs(self.cfg.output_dir, exist_ok=True)

    def seed_everything(self, seed: int):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self, cfg):
        from lam.models import ModelLAM

        model = ModelLAM(**cfg.model)
        if cfg.get('model_name') is not None and os.path.isdir(cfg.model_name) and os.path.exists(os.path.join(cfg.model_name, 'model.safetensors')):
            ckpt_path = os.path.join(cfg.model_name, 'model.safetensors')
        elif cfg.get('model_name') is not None and os.path.exists(cfg.model_name):
            ckpt_path = cfg.model_name
        else:
            ckpt_path = None

        if ckpt_path is not None:
            if ckpt_path.endswith('safetensors'):
                from safetensors.torch import load_file
                ckpt = load_file(ckpt_path, device='cpu')
            else:
                ckpt = torch.load(ckpt_path, map_location='cpu')
            state_dict = model.state_dict()
            for k, v in ckpt.items():
                if k in state_dict and state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
            logger.info('Loaded weights from %s', ckpt_path)
        return model

    def _build_dataloader(self, cfg):
        from lam.datasets.video_head import VideoHeadDataset

        dataset = VideoHeadDataset(
            root_dirs=[cfg.root_dirs] if isinstance(cfg.root_dirs, str) else cfg.root_dirs,
            meta_path=cfg.meta_path,
            sample_side_views=int(cfg.sample_side_views),
            render_image_res_low=int(cfg.render_image_res),
            render_image_res_high=int(cfg.render_image_res),
            render_region_size=int(cfg.render_region_size),
            source_image_res=int(cfg.source_image_res),
            repeat_num=int(cfg.get('repeat_num', 1)),
            enlarge_ratio=cfg.get('enlarge_ratio', [0.8, 1.2]),
            debug=cfg.get('debug', False),
            is_val=False,
            multiply=cfg.get('multiply', 16),
        )
        return DataLoader(
            dataset,
            batch_size=int(cfg.batch_size),
            shuffle=True,
            num_workers=int(cfg.num_workers),
            pin_memory=torch.cuda.is_available(),
        )

    def _prepare_style_reference(self, cfg, dump_dir):
        from lam.stylization.reference import prepare_style_reference

        style_image_path = cfg.get('style_image_path', None)
        if not style_image_path or not os.path.exists(style_image_path):
            return None, None
        return prepare_style_reference(
            style_image_path,
            self.flametracking if hasattr(self, 'flametracking') else None,
            source_size=int(cfg.source_image_res),
            aspect_standard=1.0,
            dump_tmp_dir=dump_dir,
            bg_color=1.0,
            track_geometry=cfg.get('style_track_geometry', False),
        )

    def _loss(self, pred, target):
        from lam.losses.pixelwise import PixelLoss

        return PixelLoss(option='l1')(pred, target)

    def _build_optimizer(self, model, cfg):
        # Allow freezing modules via config and use only trainable parameters
        freeze_modules = bool(cfg.get('freeze_modules', False))
        freeze_config = cfg.get('freeze_config', {}) or {}
        if freeze_modules:
            self._freeze_modules(model, freeze_config)

        # Optionally use parameter groups (e.g., higher LR for style adapter)
        if cfg.get('use_param_groups', False):
            param_groups = self._get_optimizer_param_groups(model, cfg)
            return torch.optim.Adam(param_groups)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable_params, lr=float(cfg.lr))

    def _freeze_modules(self, model, freeze_config: dict):
        """Freeze specified modules to preserve pretrained weights."""
        if freeze_config.get('freeze_encoder', False) and hasattr(model, 'encoder'):
            for param in model.encoder.parameters():
                param.requires_grad = False
            logger.info('Frozen encoder parameters')

        if freeze_config.get('freeze_transformer', False) and hasattr(model, 'transformer'):
            for param in model.transformer.parameters():
                param.requires_grad = False
            logger.info('Frozen transformer parameters')

        if freeze_config.get('freeze_renderer', False) and hasattr(model, 'renderer'):
            for param in model.renderer.parameters():
                param.requires_grad = False
            logger.info('Frozen renderer parameters')

    def _get_optimizer_param_groups(self, model, cfg):
        """Create parameter groups with different learning rates.

        By default, style_adapter parameters get `lr_style_adapter` and others get `lr_other_modules`.
        """
        adapter_params = []
        if hasattr(model, 'style_adapter'):
            adapter_params = list(model.style_adapter.parameters())

        adapter_param_ids = {id(p) for p in adapter_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in adapter_param_ids]

        groups = []
        if adapter_params:
            groups.append({
                'params': adapter_params,
                'lr': float(cfg.get('lr_style_adapter', cfg.lr)),
            })
        if other_params:
            groups.append({
                'params': other_params,
                'lr': float(cfg.get('lr_other_modules', float(cfg.lr) * 0.1)),
            })
        return groups

    def _build_scheduler(self, optimizer, cfg, total_steps):
        from lam.utils.scheduler import CosineWarmupScheduler

        return CosineWarmupScheduler(optimizer, warmup_iters=max(1, total_steps // 10), max_iters=max(1, total_steps), initial_lr=1e-6)

    def _resolve_style_image_paths(self, cfg):
        paths = []
        configured_paths = cfg.get('style_image_paths', None)
        if configured_paths is not None:
            if isinstance(configured_paths, str):
                configured_paths = [configured_paths]
            else:
                configured_paths = OmegaConf.to_container(configured_paths, resolve=True)
            for path in configured_paths:
                if glob.has_magic(path):
                    paths.extend(sorted(glob.glob(path)))
                elif os.path.isdir(path):
                    paths.extend(sorted(glob.glob(os.path.join(path, '*'))))
                elif os.path.exists(path):
                    paths.append(path)

        single_path = cfg.get('style_image_path', None)
        if single_path:
            if glob.has_magic(single_path):
                paths.extend(sorted(glob.glob(single_path)))
            elif os.path.isdir(single_path):
                paths.extend(sorted(glob.glob(os.path.join(single_path, '*'))))
            elif os.path.exists(single_path):
                paths.append(single_path)

        if not paths:
            paths.extend(glob.glob(cfg.get('style_image_glob', './style_image*.png')))

        deduped = []
        seen = set()
        for path in sorted(paths, key=_style_path_sort_key):
            norm_path = os.path.normpath(path)
            if norm_path in seen:
                continue
            if os.path.exists(norm_path):
                deduped.append(norm_path)
                seen.add(norm_path)

        if not deduped:
            raise FileNotFoundError(
                'No style images found. Set style_image_paths, style_image_path, '
                'or style_image_glob in the train config.'
            )
        return deduped

    def _write_style_image_outputs(self, model, dataset, losses, output_dir):
        sample_dir = os.path.join(output_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)

        eval_loader = DataLoader(dataset, batch_size=max(1, min(8, len(dataset))), shuffle=False)
        originals = []
        reconstructions = []
        latents = []
        names = []

        model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                images = batch['image'].to(self.device)
                latent = model.encode(images)
                reconstructed = model.decode(latent)
                originals.extend(image.cpu() for image in images)
                reconstructions.extend(image.cpu() for image in reconstructed)
                latents.append(latent.cpu())
                names.extend(batch['name'])

        print(f"[LAM] saving samples to {sample_dir}", flush=True)
        save_limit = max(1, min(16, len(reconstructions)))
        for idx, (name, reconstructed) in enumerate(zip(names[:save_limit], reconstructions[:save_limit])):
            stem, _ = os.path.splitext(name)
            _tensor_to_pil(reconstructed).save(os.path.join(sample_dir, f'reconstruction_{stem}.png'))

        cols = min(4, len(originals[:save_limit]))
        _save_grid(originals[:save_limit], os.path.join(sample_dir, 'style_inputs_grid.png'), cols=cols)
        _save_grid(reconstructions[:save_limit], os.path.join(sample_dir, 'style_reconstructions_grid.png'), cols=cols)
        _save_grid(originals[:save_limit] + reconstructions[:save_limit], os.path.join(sample_dir, 'style_training_grid.png'), cols=cols)

        latent_tensor = torch.cat(latents, dim=0) if latents else torch.empty(0, 3, image_size, image_size)
        average = model.decode(latent_tensor.mean(dim=0, keepdim=True).to(self.device))[0] if latent_tensor.numel() > 0 else None
        average_path = os.path.join(sample_dir, 'style_average.png')
        if average is not None:
            _tensor_to_pil(average).save(average_path)

        blend_path = None
        if latent_tensor.shape[0] >= 2:
            blend_steps = max(2, int(self.cfg.get('latent_blend_steps', 5)))
            blend_images = []
            with torch.no_grad():
                for pair_idx in range(min(latent_tensor.shape[0] - 1, 3)):
                    start = latent_tensor[pair_idx:pair_idx + 1].to(self.device)
                    end = latent_tensor[pair_idx + 1:pair_idx + 2].to(self.device)
                    for alpha in torch.linspace(0, 1, steps=blend_steps, device=self.device):
                        latent = start * (1.0 - alpha) + end * alpha
                        blend_images.append(model.decode(latent)[0].cpu())
            blend_path = os.path.join(sample_dir, 'style_latent_blends.png')
            _save_grid(blend_images, blend_path, cols=blend_steps)

        checkpoint_path = os.path.join(output_dir, 'style_autoencoder_final.pt')
        torch.save({
            'model': model.state_dict(),
            'config': OmegaConf.to_container(self.cfg, resolve=True),
            'losses': losses,
            'image_paths': dataset.image_paths,
        }, checkpoint_path)

        summary = {
            'mode': 'style_images',
            'num_images': len(dataset),
            'image_paths': dataset.image_paths,
            'epochs': int(self.cfg.epochs),
            'final_loss': float(losses[-1]) if losses else None,
            'checkpoint': checkpoint_path,
            'samples': {
                'inputs_grid': os.path.join(sample_dir, 'style_inputs_grid.png'),
                'reconstructions_grid': os.path.join(sample_dir, 'style_reconstructions_grid.png'),
                'training_grid': os.path.join(sample_dir, 'style_training_grid.png'),
                'average': average_path,
                'latent_blends': blend_path,
            },
        }
        summary_path = os.path.join(output_dir, 'style_train_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        return summary_path, summary

    def _load_image_tensor(self, image_path, image_size):
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            image = ImageOps.fit(
                image,
                (int(image_size), int(image_size)),
                method=_resample_bicubic(),
                centering=(0.5, 0.5),
            )
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _load_flame_betas(self, shape_path, shape_dims):
        if not shape_path or not os.path.exists(shape_path):
            raise FileNotFoundError(f"FLAME shape path does not exist: {shape_path}")
        data = np.load(shape_path, allow_pickle=True)
        if 'shape' in data:
            shape = data['shape']
        elif 'betas' in data:
            shape = data['betas']
        else:
            raise KeyError(f"Expected 'shape' or 'betas' in {shape_path}")
        shape = np.asarray(shape, dtype=np.float32).reshape(-1)[:int(shape_dims)]
        if shape.shape[0] < int(shape_dims):
            shape = np.pad(shape, (0, int(shape_dims) - shape.shape[0]))
        return torch.from_numpy(shape).unsqueeze(0)

    def _load_geometry_ref_records(self, cfg):
        records_path = cfg.get('selected_geometry_refs', None)
        if not records_path or not os.path.exists(records_path):
            raise FileNotFoundError(f"Selected geometry refs file does not exist: {records_path}")
        with open(records_path, 'r') as f:
            records = json.load(f)
        usable = []
        for record in records:
            image_path = record.get('style_image_path') or record.get('selected_copy_path')
            shape_path = record.get('shape_path')
            if image_path and shape_path and os.path.exists(image_path) and os.path.exists(shape_path):
                usable.append(record)
        if not usable:
            raise FileNotFoundError(f"No usable geometry refs found in {records_path}")
        limit = int(cfg.get('max_style_refs', 0) or 0)
        return usable[:limit] if limit > 0 else usable

    def train_geometry_adapter(self):
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.model = self._build_model(self.cfg).to(self.device)
        self.model.train()

        if self.cfg.get('style_adapter_max_query_offset', None) is not None:
            offset = float(self.cfg.get('style_adapter_max_query_offset'))
            self.model.style_adapter.max_query_offset.fill_(offset)

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.style_adapter.geometry_mlp.parameters():
            param.requires_grad = True
        self.model.style_adapter.max_query_offset.requires_grad = False

        source_image = self._load_image_tensor(
            self.cfg.get('source_image_path'),
            int(self.cfg.get('source_image_res', self.cfg.get('image_size', 512))),
        ).unsqueeze(0).to(self.device)
        source_betas = self._load_flame_betas(
            self.cfg.get('source_shape_path'),
            int(self.cfg.get('shape_dims', 10)),
        ).to(self.device)
        source_flame = {'betas': source_betas}
        with torch.no_grad():
            source_query_points, _ = self.model.renderer.get_query_points(source_flame, device=self.device)

        records = self._load_geometry_ref_records(self.cfg)
        optimizer = torch.optim.AdamW(
            self.model.style_adapter.geometry_mlp.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.get('weight_decay', 0.0)),
        )
        loss_fn = nn.SmoothL1Loss(beta=float(self.cfg.get('smooth_l1_beta', 0.01)))
        losses = []
        reg_weight = float(self.cfg.get('adapter_offset_reg_weight', 0.01))
        target_blend = float(self.cfg.get('target_geometry_blend', 1.0))
        geometry_strength = float(self.cfg.get('style_geometry_strength', 1.0))

        print(
            f"[LAM] stage4 geometry-adapter training start: {len(records)} refs, "
            f"epochs={int(self.cfg.epochs)}, lr={float(self.cfg.lr):.2e}, "
            f"max_query_offset={float(self.model.style_adapter.max_query_offset.item()):.4f}",
            flush=True,
        )

        with torch.no_grad():
            source_feats = self.model.forward_encode_image(source_image)
            source_tokens = self.model.forward_transformer(
                source_feats,
                camera_embeddings=None,
                query_points=source_query_points,
                query_feats=None,
            ).detach()

            train_items = []
            for record in records:
                style_image = self._load_image_tensor(
                    record.get('style_image_path') or record.get('selected_copy_path'),
                    int(self.cfg.get('source_image_res', self.cfg.get('image_size', 512))),
                ).unsqueeze(0).to(self.device)
                style_betas = self._load_flame_betas(
                    record['shape_path'],
                    int(self.cfg.get('shape_dims', 10)),
                ).to(self.device)
                target_query_points, _ = self.model.renderer.get_query_points({'betas': style_betas}, device=self.device)
                target_query_points = source_query_points + (target_query_points - source_query_points) * target_blend
                train_items.append({
                    'style_id': record.get('style_id', os.path.basename(record.get('shape_path', 'style'))),
                    'style_feats': self.model.forward_encode_image(style_image).detach(),
                    'target_query_points': target_query_points.detach(),
                })
        print(f"[LAM] cached Stage 4 features for {len(train_items)} refs", flush=True)

        for epoch in range(int(self.cfg.epochs)):
            random.shuffle(train_items)
            epoch_losses = []
            print(f"[LAM] START STAGE4 EPOCH {epoch + 1}/{int(self.cfg.epochs)}", flush=True)
            for item in train_items:
                optimizer.zero_grad(set_to_none=True)
                _, pred_query_points, _, _ = self.model.style_adapter(
                    source_tokens,
                    item['style_feats'],
                    query_points=source_query_points.detach(),
                    appearance_strength=0.0,
                    geometry_strength=geometry_strength,
                    opacity_strength=0.0,
                )
                pred_delta = pred_query_points - source_query_points.detach()
                loss = loss_fn(pred_query_points, item['target_query_points'])
                if reg_weight > 0:
                    loss = loss + reg_weight * pred_delta.pow(2).mean()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

            mean_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
            losses.append(mean_loss)
            print(f"[LAM] END STAGE4 EPOCH {epoch + 1}/{int(self.cfg.epochs)} loss={mean_loss:.6f}", flush=True)
            torch.save(self.model.state_dict(), os.path.join(self.cfg.output_dir, f'model_epoch_{epoch + 1:03d}.pt'))

        final_path = os.path.join(self.cfg.output_dir, 'model_final.pt')
        adapter_path = os.path.join(self.cfg.output_dir, 'style_geometry_adapter.pt')
        torch.save(self.model.state_dict(), final_path)
        torch.save({
            'style_adapter': self.model.style_adapter.state_dict(),
            'geometry_mlp': self.model.style_adapter.geometry_mlp.state_dict(),
            'config': OmegaConf.to_container(self.cfg, resolve=True),
            'losses': losses,
        }, adapter_path)

        summary = {
            'mode': 'full_lam_geometry_adapter',
            'num_style_refs': len(records),
            'epochs': int(self.cfg.epochs),
            'final_loss': losses[-1] if losses else None,
            'losses': losses,
            'source_image_path': self.cfg.get('source_image_path'),
            'source_shape_path': self.cfg.get('source_shape_path'),
            'selected_geometry_refs': self.cfg.get('selected_geometry_refs'),
            'model_checkpoint': final_path,
            'adapter_checkpoint': adapter_path,
            'style_adapter_max_query_offset': float(self.model.style_adapter.max_query_offset.item()),
            'target_geometry_blend': target_blend,
            'style_geometry_strength': geometry_strength,
        }
        summary_path = os.path.join(self.cfg.output_dir, 'geometry_adapter_train_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Stage 4 geometry-adapter training finished. Saved {final_path}", flush=True)
        print(f"Adapter-only checkpoint saved to {adapter_path}", flush=True)
        print(f"Summary written to {summary_path}", flush=True)

    def train_style_appearance_e2e(self):
        """Train ReferenceStyleAdapter.appearance_mlp end-to-end.

        Unlike Stage 1/2 (TinyStyleAutoEncoder, trained disconnected from the
        real style path) and Stage 4 (which only trains geometry_mlp), this
        renders the actual model output via infer_single_view (fully
        differentiable, no torch.no_grad inside it) and backpropagates a
        Gatys-style content+style loss straight into appearance_mlp, the
        module that was never trained anywhere before.
        """
        import json as json_module
        from lam.datasets.style_pairs import StylePairDataset

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.model = self._build_model(self.cfg).to(self.device)
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.style_adapter.appearance_mlp.parameters():
            param.requires_grad = True
        trainable_params = list(self.model.style_adapter.appearance_mlp.parameters())

        content_manifest_path = self.cfg.get('content_refs_manifest')
        with open(content_manifest_path, 'r') as f:
            content_records = json_module.load(f)
        style_paths = sorted(glob.glob(self.cfg.get('style_image_glob')))
        if not style_paths:
            raise FileNotFoundError(f"No style images found at {self.cfg.get('style_image_glob')}")

        source_size = int(self.cfg.get('source_image_res', 512))
        bg_color = float(self.cfg.get('bg_color', 1.0))
        dataset = StylePairDataset(content_records, style_paths, source_size=source_size, bg_color=bg_color)

        generator = torch.Generator()
        generator.manual_seed(int(self.cfg.seed))
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # infer_single_view asserts image.shape[0] == 1
            shuffle=True,
            num_workers=int(self.cfg.get('num_workers', 2)),
            generator=generator,
        )

        criterion = StyleTransferLoss(
            content_layers=tuple(self.cfg.get('style_content_layers', ['relu3_3'])),
            style_layers=tuple(self.cfg.get('style_loss_layers', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])),
            lambda_content=float(self.cfg.get('style_content_loss_weight', 1.0)),
            lambda_style=float(self.cfg.get('style_style_loss_weight', 1e5)),
            lambda_pixel=float(self.cfg.get('style_pixel_loss_weight', 0.01)),
            lambda_adain=float(self.cfg.get('style_adain_loss_weight', 1.0)),
            lambda_tv=float(self.cfg.get('style_tv_loss_weight', 1e-4)),
        ).to(self.device)

        identity_loss_weight = float(self.cfg.get('identity_loss_weight', 0.1))
        appearance_strength = float(self.cfg.get('train_style_appearance_strength', 1.0))
        opacity_strength = float(self.cfg.get('train_style_opacity_strength', 0.0))

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.get('weight_decay', 0.0)),
        )
        accum_steps = max(1, int(self.cfg.get('grad_accum_steps', 4)))
        # Was hardcoded to 1.0, a holdover from when appearance styling could
        # corrupt geometry and needed a tight leash. Measured pre-clip norms of
        # 20-70 on this (now color-only, architecturally safe) path -- the old
        # value was clipping essentially every step down to the same ceiling
        # regardless of loss weights, which is why reweighting content/style/
        # identity alone couldn't produce a stronger learned response.
        grad_clip_norm = float(self.cfg.get('grad_clip_norm', 1.0))
        sample_interval = max(1, int(self.cfg.get('sample_interval', 50)))
        sample_dir = os.path.join(self.cfg.output_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)

        dtype = torch.float32
        losses = []
        global_step = 0

        print(
            f"[LAM] appearance e2e training start: {len(dataset)} content refs, "
            f"{len(style_paths)} style images, accum_steps={accum_steps}, "
            f"epochs={int(self.cfg.epochs)}, lr={float(self.cfg.lr):.2e}",
            flush=True,
        )

        for epoch in range(int(self.cfg.epochs)):
            print(f"[LAM] START APPEARANCE EPOCH {epoch + 1}/{int(self.cfg.epochs)}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            epoch_losses = []

            for batch in dataloader:
                content_image = batch['content_image'].to(self.device, dtype)
                style_image = batch['style_image'].to(self.device, dtype)
                render_c2w = batch['render_c2w'].to(self.device, dtype)
                render_intr = batch['render_intr'].to(self.device, dtype)
                render_bg = batch['render_bg'].to(self.device, dtype)
                flame_params = {
                    k[len('flame_'):]: v.to(self.device, dtype)
                    for k, v in batch.items() if k.startswith('flame_')
                }

                res = render_view_with_grad(
                    self.model,
                    content_image.unsqueeze(1),
                    render_c2ws=render_c2w,
                    render_intrs=render_intr,
                    render_bg_colors=render_bg,
                    flame_params=flame_params,
                    style_image=style_image,
                    style_shape_params=None,
                    style_appearance_strength=appearance_strength,
                    style_geometry_strength=0.0,
                    style_opacity_strength=opacity_strength,
                )
                # comp_rgb comes back batch-squeezed as [Nv, H, W, 3] (Nv=1 here).
                rendered = res['comp_rgb'][0].permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)  # [1, 3, H, W]
                # The renderer's output resolution is derived from the tracked camera's
                # cx/cy and can differ by a few px from the dataset's preprocessed size;
                # align the content target so content/pixel losses compare matching shapes.
                if rendered.shape[-2:] != content_image.shape[-2:]:
                    content_image_for_loss = F.interpolate(
                        content_image, size=rendered.shape[-2:], mode='bilinear', align_corners=False
                    )
                else:
                    content_image_for_loss = content_image

                loss, loss_dict = criterion(rendered, content_image_for_loss, style_image)

                identity_loss = torch.zeros((), device=self.device)
                if identity_loss_weight > 0:
                    # DINOv2 patch embed requires input dims divisible by 14; content_image
                    # already is (preprocess_image enforces multiply=14), rendered's native
                    # size comes from the tracked camera intrinsics and may not be.
                    if rendered.shape[-2:] != content_image.shape[-2:]:
                        rendered_for_id = F.interpolate(
                            rendered, size=content_image.shape[-2:], mode='bilinear', align_corners=False
                        )
                    else:
                        rendered_for_id = rendered
                    with torch.no_grad():
                        content_feat = self.model.forward_encode_image(content_image).mean(dim=1)
                    rendered_feat = self.model.forward_encode_image(rendered_for_id).mean(dim=1)
                    identity_loss = (1.0 - F.cosine_similarity(content_feat, rendered_feat, dim=-1)).mean()
                    loss = loss + identity_loss_weight * identity_loss
                loss_dict['identity'] = float(identity_loss.item())
                loss_dict['step_total'] = float(loss.item())
                global_step += 1

                if not torch.isfinite(loss):
                    # Defense in depth: the appearance_mlp tanh-bound (modeling_lam.py)
                    # and the lowered style-loss weight (see config comment) fix the root
                    # cause of the divergence that crashed an earlier run, but a single
                    # degenerate (content, style) pair could still in principle produce a
                    # non-finite loss -- drop it rather than corrupt the optimizer state.
                    print(f"[LAM] WARNING step={global_step}: non-finite loss, skipping", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                (loss / accum_steps).backward()

                if global_step % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                epoch_losses.append(loss_dict['step_total'])
                losses.append(loss_dict)
                print(
                    f"[LAM] step={global_step} epoch={epoch + 1} "
                    f"content={loss_dict['content']:.4f} style={loss_dict['style']:.4f} "
                    f"pixel={loss_dict['pixel']:.4f} identity={loss_dict['identity']:.4f} "
                    f"total={loss_dict['step_total']:.4f}",
                    flush=True,
                )

                if global_step % sample_interval == 0:
                    style_image_for_grid = F.interpolate(
                        style_image, size=rendered.shape[-2:], mode='bilinear', align_corners=False
                    ) if style_image.shape[-2:] != rendered.shape[-2:] else style_image
                    _save_grid(
                        [content_image_for_loss[0].clamp(0, 1), style_image_for_grid[0].clamp(0, 1), rendered[0].clamp(0, 1)],
                        os.path.join(sample_dir, f'step_{global_step:06d}.png'),
                        cols=3,
                    )

            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            print(f"[LAM] END APPEARANCE EPOCH {epoch + 1}/{int(self.cfg.epochs)} mean_loss={mean_loss:.6f}", flush=True)
            torch.save(self.model.state_dict(), os.path.join(self.cfg.output_dir, f'model_epoch_{epoch + 1:03d}.pt'))

        final_path = os.path.join(self.cfg.output_dir, 'model_final.pt')
        torch.save(self.model.state_dict(), final_path)

        summary = {
            'mode': 'style_appearance_e2e',
            'num_content_refs': len(dataset),
            'num_style_images': len(style_paths),
            'epochs': int(self.cfg.epochs),
            'final_loss': losses[-1]['step_total'] if losses else None,
            'losses': losses,
            'content_refs_manifest': content_manifest_path,
            'style_image_glob': self.cfg.get('style_image_glob'),
            'model_checkpoint': final_path,
        }
        summary_path = os.path.join(self.cfg.output_dir, 'style_appearance_train_summary.json')
        with open(summary_path, 'w') as f:
            json_module.dump(summary, f, indent=2)
        print(f"Appearance e2e training finished. Saved {final_path}", flush=True)
        print(f"Summary written to {summary_path}", flush=True)

    def train_style_images(self):
        style_paths = self._resolve_style_image_paths(self.cfg)
        image_size = int(self.cfg.get('image_size', 128))
        if image_size % 4 != 0:
            raise ValueError('image_size must be divisible by 4 for the tiny style autoencoder.')

        dataset = StyleImageDataset(style_paths, image_size=image_size)
        batch_size = max(1, min(int(self.cfg.batch_size), len(dataset)))
        generator = torch.Generator()
        generator.manual_seed(int(self.cfg.seed))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(self.cfg.num_workers),
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )

        print(f"[LAM] style training start: {len(dataset)} images, batch_size={batch_size}, num_workers={int(self.cfg.num_workers)}, epochs={int(self.cfg.epochs)}", flush=True)

        model = TinyStyleAutoEncoder(
            base_channels=int(self.cfg.get('style_base_channels', 24)),
            latent_channels=int(self.cfg.get('style_latent_channels', 64)),
        ).to(self.device)
        model_name = self.cfg.get('model_name', None)
        if model_name:
            if not os.path.exists(model_name):
                raise FileNotFoundError(f"Configured model_name checkpoint does not exist: {model_name}")
            ckpt = torch.load(model_name, map_location='cpu')
            state_dict = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning('Missing keys while loading style checkpoint: %s', missing)
            if unexpected:
                logger.warning('Unexpected keys while loading style checkpoint: %s', unexpected)
            print(f"[LAM] loaded style checkpoint from {model_name}", flush=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.cfg.lr))
        criterion = StyleTransferLoss(
            content_layers=tuple(self.cfg.get('style_content_layers', ['relu3_3'])),
            style_layers=tuple(self.cfg.get('style_loss_layers', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])),
            lambda_content=float(self.cfg.get('style_content_loss_weight', 1.0)),
            lambda_style=float(self.cfg.get('style_style_loss_weight', 1e5)),
            lambda_pixel=float(self.cfg.get('style_pixel_loss_weight', 0.01)),
            lambda_adain=float(self.cfg.get('style_adain_loss_weight', 1.0)),
            lambda_tv=float(self.cfg.get('style_tv_loss_weight', 1e-4)),
        ).to(self.device)
        noise_std = float(self.cfg.get('noise_std', 0.03))
        losses = []

        for epoch in range(int(self.cfg.epochs)):
            print(f"[LAM] START EPOCH {epoch + 1}/{int(self.cfg.epochs)}", flush=True)
            model.train()
            epoch_losses = []
            for batch in dataloader:
                clean = batch['image'].to(self.device)
                noisy = clean
                if noise_std > 0:
                    noisy = torch.clamp(clean + torch.randn_like(clean) * noise_std, 0.0, 1.0)

                optimizer.zero_grad(set_to_none=True)
                reconstructed = model(noisy)
                loss, loss_dict = criterion(reconstructed, clean, clean)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            mean_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
            losses.append(mean_loss)
            print(f"[LAM] END EPOCH {epoch + 1}/{int(self.cfg.epochs)} loss={mean_loss:.6f}", flush=True)
            logger.info('style epoch=%d/%d loss=%.6f', epoch + 1, int(self.cfg.epochs), mean_loss)

        summary_path, summary = self._write_style_image_outputs(model, dataset, losses, self.cfg.output_dir)
        print(f"Style-image training finished. Saved {summary['checkpoint']}")
        print(f"Sample outputs are in {os.path.join(self.cfg.output_dir, 'samples')}")
        print(f"Summary written to {summary_path}")

    def train_one_batch(self, batch, style_image=None, style_shape_params=None):
        self.model.train()
        image = batch['source_rgbs'].to(self.device)
        source_c2ws = batch['source_c2ws'].to(self.device)
        source_intrs = batch['source_intrs'].to(self.device)
        render_c2ws = batch['c2ws'].to(self.device)
        render_intrs = batch['intrs'].to(self.device)
        render_bg_colors = batch['render_bg_colors'].to(self.device)
        flame_params = {k: v.to(self.device) for k, v in batch.items() if k in ['expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation', 'betas']}
        flame_params['betas'] = batch['betas'].to(self.device)

        out = self.model(
            image=image.unsqueeze(1),
            source_c2ws=source_c2ws.unsqueeze(1),
            source_intrs=source_intrs.unsqueeze(1),
            render_c2ws=render_c2ws.unsqueeze(1),
            render_intrs=render_intrs.unsqueeze(1),
            render_bg_colors=render_bg_colors.unsqueeze(1),
            flame_params=flame_params,
            style_image=style_image,
            style_shape_params=style_shape_params,
            style_appearance_strength=self.cfg.get('style_appearance_strength', 0.35),
            style_geometry_strength=self.cfg.get('style_geometry_strength', 0.25),
            style_opacity_strength=self.cfg.get('style_opacity_strength', 0.0),
        )
        target = batch['render_image'].to(self.device)
        pred = out['comp_rgb'].permute(0, 3, 1, 2)
        loss = self._loss(pred, target)
        return loss, out

    def infer(self):
        self.model = self._build_model(self.cfg).to(self.device)
        self.model.train()
        dataloader = self._build_dataloader(self.cfg)
        optimizer = self._build_optimizer(self.model, self.cfg)
        total_steps = max(1, len(dataloader) * int(self.cfg.epochs))
        scheduler = self._build_scheduler(optimizer, self.cfg, total_steps)

        style_image, style_shape_params = self._prepare_style_reference(self.cfg, self.cfg.output_dir)
        if style_image is not None:
            style_image = style_image.unsqueeze(0).to(self.device)
            if style_shape_params is not None:
                style_shape_params = style_shape_params.unsqueeze(0).to(self.device)

        for epoch in range(int(self.cfg.epochs)):
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad(set_to_none=True)
                loss, _ = self.train_one_batch(batch, style_image=style_image, style_shape_params=style_shape_params)
                loss.backward()
                optimizer.step()
                scheduler.step()
                logger.info('epoch=%d step=%d loss=%.6f lr=%.6e', epoch + 1, step + 1, loss.item(), optimizer.param_groups[0]['lr'])
                if (step + 1) % int(self.cfg.get('save_interval', 1)) == 0:
                    ckpt_path = os.path.join(self.cfg.output_dir, f'epoch_{epoch + 1}_step_{step + 1}.pt')
                    torch.save(self.model.state_dict(), ckpt_path)

        final_path = os.path.join(self.cfg.output_dir, 'model_final.pt')
        torch.save(self.model.state_dict(), final_path)
        logger.info('Training finished. Saved %s', final_path)

    def run(self):
        training_mode = str(self.cfg.get('training_mode', 'style_images')).lower()
        if training_mode in {'style_images', 'style-image', 'lightweight', 'lightweight_style'}:
            self.train_style_images()
            return
        if training_mode in {'full_lam_geometry_adapter', 'geometry_adapter', 'stage4_geometry_adapter'}:
            self.train_geometry_adapter()
            return
        if training_mode in {'style_appearance_e2e', 'appearance_adapter'}:
            self.train_style_appearance_e2e()
            return
        self.infer()
