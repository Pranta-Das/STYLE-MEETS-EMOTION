# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import math
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from accelerate.logging import get_logger
from einops import rearrange, repeat

from .transformer import TransformerDecoder
from lam.models.rendering.gs_renderer import GS3DRenderer, PointEmbed
from diffusers.utils import is_torch_version

logger = get_logger(__name__)


class ReferenceStyleAdapter(nn.Module):
    def __init__(self, feature_dim, max_query_offset=0.03, max_appearance_scale=0.5):
        super().__init__()
        self.register_buffer("max_query_offset", torch.tensor(float(max_query_offset)), persistent=True)
        # Appearance styling outputs a small RGB gain+shift, applied to decoded
        # Gaussian color only (see GS3DRenderer.forward_gs's color_style
        # argument) -- never to gs_hidden_features / latent_points.
        #
        # It used to modulate latent_points directly (a multiplicative gain +
        # additive shift on the whole feature vector), the same shared vector
        # that xyz-offset/scaling/rotation/opacity are ALSO decoded from
        # downstream. Once trained, that let a "color" adjustment inflate
        # Gaussian scale and blur every styled render (confirmed by an isolation
        # test: appearance-only render was blurry, geometry-only and the
        # unstyled baseline were both sharp). Keeping latent_points untouched
        # and applying style strictly post-decode, in RGB space, removes that
        # leak at the source rather than fighting it with loss weights.
        self.register_buffer("max_appearance_scale", torch.tensor(float(max_appearance_scale)), persistent=True)

        # A per-point cross-attention variant (each query point attending to the
        # style image's spatial DINOv2 feature grid individually) was tried here
        # and measured worse than this simple global tint (see
        # STYLE_GEOMETRY_PIPELINE.md) -- less spatial differentiation, not more,
        # because nothing in the training loss gives per-point attention a
        # region-specific reason to specialize. It also introduced a correctness
        # bug: cross_attn has no residual connection, so any checkpoint that
        # doesn't already contain trained style_cross_attn weights (i.e. every
        # checkpoint in this project) fed appearance_mlp a signal scrambled by an
        # untrained, randomly-initialized attention layer instead of a clean
        # style summary -- removed rather than left in place unused.
        self.appearance_mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 4),
            nn.SiLU(),
            nn.Linear(feature_dim // 4, 6),  # 3 gamma + 3 beta, RGB
        )
        self.geometry_mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, 3),
        )
        nn.init.zeros_(self.appearance_mlp[-1].weight)
        nn.init.zeros_(self.appearance_mlp[-1].bias)
        nn.init.zeros_(self.geometry_mlp[-1].weight)
        nn.init.zeros_(self.geometry_mlp[-1].bias)

    def forward(self, latent_points, style_feats, query_points=None,
                appearance_strength=0.0, geometry_strength=0.0,
                opacity_strength=0.0):
        style_feats = style_feats.to(latent_points.dtype)
        style_code = style_feats.mean(dim=1)  # still used by geometry_mlp below, unchanged

        # Same style_code broadcast to every query point, so all points share
        # one global RGB gain+shift -- a uniform tint, not spatial texture.
        style_context = style_code[:, None, :].expand(-1, latent_points.shape[1], -1)
        color_gamma, color_beta = self.appearance_mlp(style_context).chunk(2, dim=-1)  # [B, N_points, 3] each
        max_scale = self.max_appearance_scale.to(color_gamma.dtype)
        color_strength = appearance_strength + 0.45 * opacity_strength
        color_gamma = torch.tanh(color_gamma) * max_scale * color_strength
        color_beta = torch.tanh(color_beta) * max_scale * color_strength

        if query_points is not None and geometry_strength > 0:
            style_points = latent_points + style_code[:, None] * (0.02 + 0.08 * geometry_strength)
            delta = torch.tanh(self.geometry_mlp(style_points))
            query_points = query_points + delta * (self.max_query_offset.to(delta.dtype) * (1.0 + 1.5 * geometry_strength))

        return latent_points, query_points, color_gamma, color_beta


class ModelLAM(nn.Module):
    """
    Full model of the basic single-view large reconstruction model.
    """
    def __init__(self,
                 transformer_dim: int, transformer_layers: int, transformer_heads: int,
                 transformer_type="cond",
                 tf_grad_ckpt=False,
                 encoder_grad_ckpt=False,
                 encoder_freeze: bool = True, encoder_type: str = 'dino',
                 encoder_model_name: str = 'facebook/dino-vitb16', encoder_feat_dim: int = 768,
                 num_pcl: int=2048, pcl_dim: int=512,
                 human_model_path="./model_zoo/human_parametric_models",
                 flame_subdivide_num=2,
                 flame_type="flame",
                 gs_query_dim=None,
                 gs_use_rgb=False,
                 gs_sh=3,
                 gs_mlp_network_config=None,
                 gs_xyz_offset_max_step=1.8 / 32,
                 gs_clip_scaling=0.2,
                 shape_param_dim=100,
                 expr_param_dim=50,
                 fix_opacity=False,
                 fix_rotation=False,
                 flame_scale=1.0,
                 **kwargs,
                 ):
        super().__init__()
        self.gradient_checkpointing = tf_grad_ckpt
        self.encoder_gradient_checkpointing = encoder_grad_ckpt
        
        # attributes
        self.encoder_feat_dim = encoder_feat_dim
        self.conf_use_pred_img = False
        self.conf_cat_feat = False and self.conf_use_pred_img  # True # False
        self.default_style_appearance_strength = kwargs.get("style_appearance_strength", 0.0)
        self.default_style_geometry_strength = kwargs.get("style_geometry_strength", 0.0)
        self.default_style_opacity_strength = kwargs.get("style_opacity_strength", 0.0)
        self.style_feature_blend = kwargs.get("style_feature_blend", 0.0)

        # modules
        # image encoder
        self.encoder = self._encoder_fn(encoder_type)(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
            encoder_feat_dim=encoder_feat_dim,
        )

        # learnable points embedding
        skip_decoder = False
        self.latent_query_points_type = kwargs.get("latent_query_points_type", "e2e_flame")
        if self.latent_query_points_type == "embedding":
            self.num_pcl = num_pcl
            self.pcl_embeddings = nn.Embedding(num_pcl , pcl_dim)
        elif self.latent_query_points_type.startswith("flame"):
            latent_query_points_file = os.path.join(human_model_path, "flame_points", f"{self.latent_query_points_type}.npy")
            pcl_embeddings = torch.from_numpy(np.load(latent_query_points_file)).float()
            print(f"==========load flame points:{latent_query_points_file}, shape:{pcl_embeddings.shape}")
            self.register_buffer("pcl_embeddings", pcl_embeddings)
            self.pcl_embed = PointEmbed(dim=pcl_dim)
        elif self.latent_query_points_type.startswith("e2e_flame"):
            skip_decoder = True
            self.pcl_embed = PointEmbed(dim=pcl_dim)
        else:
            raise NotImplementedError
        print("==="*16*3, f"\nskip_decoder: {skip_decoder}", "\n"+"==="*16*3)
        # transformer
        self.transformer = TransformerDecoder(
            block_type=transformer_type,
            num_layers=transformer_layers, num_heads=transformer_heads,
            inner_dim=transformer_dim, cond_dim=encoder_feat_dim, mod_dim=None,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.style_adapter = ReferenceStyleAdapter(
            transformer_dim,
            max_query_offset=kwargs.get("style_adapter_max_query_offset", 0.03),
            max_appearance_scale=kwargs.get("style_adapter_max_appearance_scale", 0.5),
        )
        
        # renderer
        self.renderer = GS3DRenderer(human_model_path=human_model_path,
                                     subdivide_num=flame_subdivide_num,
                                     smpl_type=flame_type,
                                     feat_dim=transformer_dim,
                                     query_dim=gs_query_dim,
                                     use_rgb=gs_use_rgb,
                                     sh_degree=gs_sh,
                                     mlp_network_config=gs_mlp_network_config,
                                     xyz_offset_max_step=gs_xyz_offset_max_step,
                                     clip_scaling=gs_clip_scaling,
                                     scale_sphere=kwargs.get("scale_sphere", False),
                                     shape_param_dim=shape_param_dim,
                                     expr_param_dim=expr_param_dim,
                                     fix_opacity=fix_opacity,
                                     fix_rotation=fix_rotation,
                                     skip_decoder=skip_decoder,
                                     decode_with_extra_info=kwargs.get("decode_with_extra_info", None),
                                     gradient_checkpointing=self.gradient_checkpointing,
                                     add_teeth=kwargs.get("add_teeth", True),
                                     teeth_bs_flag=kwargs.get("teeth_bs_flag", False),
                                     oral_mesh_flag=kwargs.get("oral_mesh_flag", False),
                                     use_mesh_shading=kwargs.get('use_mesh_shading', False),
                                     render_rgb=kwargs.get("render_rgb", True),
                                     )

    def get_last_layer(self):
        return self.renderer.gs_net.out_layers["shs"].weight
    
    @staticmethod
    def _encoder_fn(encoder_type: str):
        from .encoders.dinov2_fusion_wrapper import Dinov2FusionWrapper
        return Dinov2FusionWrapper
        
    def forward_transformer(self, image_feats, camera_embeddings, query_points, query_feats=None):
        # assert image_feats.shape[0] == camera_embeddings.shape[0], \
        #     "Batch size mismatch for image_feats and camera_embeddings!"
        B = image_feats.shape[0]
        if self.latent_query_points_type == "embedding":
            range_ = torch.arange(self.num_pcl, device=image_feats.device)
            x =  self.pcl_embeddings(range_).unsqueeze(0).repeat((B, 1, 1)) # [B, L, D]
            
        elif self.latent_query_points_type.startswith("flame"):
            x = self.pcl_embed(self.pcl_embeddings.unsqueeze(0)).repeat((B, 1, 1)) # [B, L, D]

        elif self.latent_query_points_type.startswith("e2e_flame"):
            x = self.pcl_embed(query_points) # [B, L, D]

        x = x.to(image_feats.dtype)
        if query_feats is not None:
            x = x + query_feats.to(image_feats.dtype)
        x = self.transformer(
            x,
            cond=image_feats,
            mod=camera_embeddings,
        )  # [B, L, D]
        # x = x.to(image_feats.dtype)
        return x

    def forward_encode_image(self, image):
        # encode image
        if self.training and self.encoder_gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            image_feats = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.encoder),
                image,
                **ckpt_kwargs,
            )
        else:
            image_feats = self.encoder(image)
        return image_feats

    @torch.compile
    def _normalize_style_image(self, style_image):
        if style_image is None:
            return None
        if style_image.dim() == 5:
            style_image = style_image[:, 0]
        return style_image

    def _blend_style_shape_params(self, flame_params, style_shape_params, geometry_strength):
        if style_shape_params is None or geometry_strength <= 0:
            return flame_params

        style_shape_params = style_shape_params.to(
            device=flame_params["betas"].device,
            dtype=flame_params["betas"].dtype,
        )
        if style_shape_params.dim() == 1:
            style_shape_params = style_shape_params.unsqueeze(0)
        if style_shape_params.shape[0] == 1 and flame_params["betas"].shape[0] > 1:
            style_shape_params = style_shape_params.repeat(flame_params["betas"].shape[0], 1)

        # Capped at 1.0 here used to mean "style_geometry_strength=1.0" was
        # already the hardest push available -- passing anything higher on
        # the CLI silently did nothing. Raising the ceiling to 1.5 lets
        # strength > 1.0 extrapolate PAST the style's own shape.
        #
        # An earlier version of this blend gave the first ~24 (most visually
        # dominant) PCA shape dimensions a boosted weight, up to 1.6x, on the
        # theory that this would make strength feel stronger. Measured
        # directly (3D vertex distance from the blended shape to the style's
        # own tracked shape, same identity/style pair): that version got
        # FARTHER from the style shape as strength increased past ~0.5 --
        # strength=1.0 was already 2.6% of face-bbox-diagonal away from the
        # true style target, and strength=1.3 was 4.5% away, WORSE than not
        # blending at all (4.06% away, i.e. plain content). Root cause: the
        # boosted early dims overshot past the style target while the other
        # ~276 dims matched it exactly, and since PCA concentrates the most
        # visual influence in the earliest dims, that overshoot dominated the
        # net displacement -- "more strength" was moving away from the style
        # image, the opposite of what the parameter promises.
        #
        # Plain linear interpolation/extrapolation, uniform across all dims,
        # doesn't have this problem: strength=1.0 is mathematically GUARANTEED
        # to equal the style's own tracked shape exactly (verified: 0.0
        # residual displacement), since betas = content + 1.0*(style-content)
        # = style. strength > 1.0 extrapolates past that true target in a
        # straight, predictable line (e.g. strength=1.5 lands at
        # style + 0.5*(style-content)), not a direction skewed by a
        # weighting scheme that was never validated against the actual
        # target it was supposed to approach.
        strength = float(max(0.0, min(1.5, geometry_strength)))
        n_dims = min(flame_params["betas"].shape[-1], style_shape_params.shape[-1])
        betas = flame_params["betas"].clone()

        if n_dims > 0:
            content = flame_params["betas"][..., :n_dims]
            style = style_shape_params[..., :n_dims]
            betas[..., :n_dims] = content + strength * (style - content)

        flame_params = dict(flame_params)
        flame_params["betas"] = betas
        return flame_params

    @torch.compile
    def forward_latent_points(self, image, camera, query_points=None, additional_features=None,
                              style_image=None, style_appearance_strength=None,
                              style_geometry_strength=None,
                              style_opacity_strength=None):
        # image: [B, C_img, H_img, W_img]
        # camera: [B, D_cam_raw]
        B = image.shape[0]

        # encode image
        image_feats = self.forward_encode_image(image)
        style_feats = None
        style_image = self._normalize_style_image(style_image)
        if style_image is not None:
            style_feats = self.forward_encode_image(style_image)
            feature_blend = float(max(0.0, min(1.0, self.style_feature_blend)))
            if feature_blend > 0:
                image_feats = image_feats * (1.0 - feature_blend) + style_feats.to(image_feats.dtype) * feature_blend
        
        assert image_feats.shape[-1] == self.encoder_feat_dim, \
            f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"

        if additional_features is not None and len(additional_features.keys()) > 0:
            image_feats_bchw = rearrange(image_feats, "b (h w) c -> b c h w", h=int(math.sqrt(image_feats.shape[1])))
            additional_features["source_image_feats"] = image_feats_bchw
            proj_feats = self.renderer.get_batch_project_feats(None, query_points, additional_features=additional_features, feat_nms=['source_image_feats'], use_mesh=True)
            query_feats = proj_feats['source_image_feats']
        else:
            query_feats = None
        # # embed camera
        # camera_embeddings = self.camera_embedder(camera)
        # assert camera_embeddings.shape[-1] == self.camera_embed_dim, \
        #     f"Feature dimension mismatch: {camera_embeddings.shape[-1]} vs {self.camera_embed_dim}"

        # transformer generating latent points
        tokens = self.forward_transformer(image_feats, camera_embeddings=None, query_points=query_points, query_feats=query_feats)

        color_style = None
        if style_feats is not None:
            if style_appearance_strength is None:
                style_appearance_strength = self.default_style_appearance_strength
            if style_geometry_strength is None:
                style_geometry_strength = self.default_style_geometry_strength
            if style_opacity_strength is None:
                style_opacity_strength = self.default_style_opacity_strength
            tokens, query_points, color_gamma, color_beta = self.style_adapter(
                tokens,
                style_feats,
                query_points=query_points,
                appearance_strength=float(style_appearance_strength),
                geometry_strength=float(style_geometry_strength),
                opacity_strength=float(style_opacity_strength),
            )
            color_style = (color_gamma, color_beta)

        return tokens, image_feats, query_points, color_style

    def forward(self, image, source_c2ws, source_intrs, render_c2ws, render_intrs, render_bg_colors, flame_params, source_flame_params=None, render_images=None, data=None,
                style_image=None, style_shape_params=None, style_appearance_strength=None, style_geometry_strength=None,
                style_opacity_strength=None):
        # image: [B, N_ref, C_img, H_img, W_img]
        # source_c2ws: [B, N_ref, 4, 4]
        # source_intrs: [B, N_ref, 4, 4]
        # render_c2ws: [B, N_source, 4, 4]
        # render_intrs: [B, N_source, 4, 4]
        # render_bg_colors: [B, N_source, 3]
        # flame_params: Dict, e.g., pose_shape: [B, N_source, 21, 3], betas:[B, 100]
        assert image.shape[0] == render_c2ws.shape[0], "Batch size mismatch for image and render_c2ws"
        assert image.shape[0] == render_bg_colors.shape[0], "Batch size mismatch for image and render_bg_colors"
        assert image.shape[0] == flame_params["betas"].shape[0], "Batch size mismatch for image and flame_params"
        assert image.shape[0] == flame_params["expr"].shape[0], "Batch size mismatch for image and flame_params"
        assert len(flame_params["betas"].shape) == 2
        render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(render_intrs[0, 0, 0, 2] * 2)
        query_points = None
        if style_geometry_strength is None:
            style_geometry_strength = self.default_style_geometry_strength
        flame_params = self._blend_style_shape_params(flame_params, style_shape_params, style_geometry_strength)

        if self.latent_query_points_type.startswith("e2e_flame"):
            query_points, flame_params = self.renderer.get_query_points(flame_params,
                                                                        device=image.device)

        additional_features = {}
                                                          
        latent_points, image_feats, query_points, color_style = self.forward_latent_points(
            image[:, 0],
            camera=None,
            query_points=query_points,
            additional_features=additional_features,
            style_image=style_image,
            style_appearance_strength=style_appearance_strength,
            style_geometry_strength=style_geometry_strength,
            style_opacity_strength=style_opacity_strength,
        )  # [B, N, C]

        additional_features.update({
            "image_feats": image_feats, "image": image[:, 0],
        })
        image_feats_bchw = rearrange(image_feats, "b (h w) c -> b c h w", h=int(math.sqrt(image_feats.shape[1])))
        additional_features["image_feats_bchw"] = image_feats_bchw

        # render target views
        render_results = self.renderer(gs_hidden_features=latent_points,
                                       query_points=query_points,
                                       flame_data=flame_params,
                                       c2w=render_c2ws,
                                       intrinsic=render_intrs,
                                       height=render_h,
                                       width=render_w,
                                       background_color=render_bg_colors,
                                       additional_features=additional_features,
                                       color_style=color_style,
        )

        N, M = render_c2ws.shape[:2]
        assert render_results['comp_rgb'].shape[0] in [N, N], "Batch size mismatch for render_results"
        assert render_results['comp_rgb'].shape[1] in [M, M*2], "Number of rendered views should be consistent with render_cameras"

        if self.use_conf_map:
            b, v = render_images.shape[:2]
            if self.conf_use_pred_img:
                render_images = repeat(render_images, "b v c h w -> (b v r) c h w", r=2)
                pred_images = rearrange(render_results['comp_rgb'].detach().clone(), "b v c h w -> (b v) c h w")
            else:
                render_images = rearrange(render_images, "b v c h w -> (b v) c h w")
                pred_images = None
            conf_sigma_l1, conf_sigma_percl = self.conf_net(render_images, pred_images)  # Bx2xHxW
            conf_sigma_l1 = rearrange(conf_sigma_l1, "(b v) c h w -> b v c h w", b=b, v=v)
            conf_sigma_percl = rearrange(conf_sigma_percl, "(b v) c h w -> b v c h w", b=b, v=v)
            conf_dict = {
                "conf_sigma_l1": conf_sigma_l1,
                "conf_sigma_percl": conf_sigma_percl,
            }
        else:
            conf_dict = {}
            # self.conf_sigma_l1 = conf_sigma_l1[:,:1]
            # self.conf_sigma_l1_flip = conf_sigma_l1[:,1:]
            # self.conf_sigma_percl = conf_sigma_percl[:,:1]
            # self.conf_sigma_percl_flip = conf_sigma_percl[:,1:]

        return {
            'latent_points': latent_points,
            **render_results,
            **conf_dict,
        }
        
    @torch.no_grad()
    def infer_single_view(self, image, source_c2ws, source_intrs, render_c2ws,
                          render_intrs, render_bg_colors, flame_params,
                          style_image=None, style_shape_params=None,
                          style_appearance_strength=None, style_geometry_strength=None,
                          style_opacity_strength=None, color_style_override=None):
        # image: [B, N_ref, C_img, H_img, W_img]
        # source_c2ws: [B, N_ref, 4, 4]
        # source_intrs: [B, N_ref, 4, 4]
        # render_c2ws: [B, N_source, 4, 4]
        # render_intrs: [B, N_source, 4, 4]
        # render_bg_colors: [B, N_source, 3]
        # flame_params: Dict, e.g., pose_shape: [B, N_source, 21, 3], betas:[B, 100]
        assert image.shape[0] == render_c2ws.shape[0], "Batch size mismatch for image and render_c2ws"
        assert image.shape[0] == render_bg_colors.shape[0], "Batch size mismatch for image and render_bg_colors"
        assert image.shape[0] == flame_params["betas"].shape[0], "Batch size mismatch for image and flame_params"
        assert image.shape[0] == flame_params["expr"].shape[0], "Batch size mismatch for image and flame_params"
        assert len(flame_params["betas"].shape) == 2
        render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(render_intrs[0, 0, 0, 2] * 2)
        assert image.shape[0] == 1
        num_views = render_c2ws.shape[1]
        query_points = None
        if style_geometry_strength is None:
            style_geometry_strength = self.default_style_geometry_strength
        flame_params = self._blend_style_shape_params(flame_params, style_shape_params, style_geometry_strength)
        
        if self.latent_query_points_type.startswith("e2e_flame"):
            query_points, flame_params = self.renderer.get_query_points(flame_params,
                                                                        device=image.device)
        latent_points, image_feats, query_points, color_style = self.forward_latent_points(
            image[:, 0],
            camera=None,
            query_points=query_points,
            style_image=style_image,
            style_appearance_strength=style_appearance_strength,
            style_geometry_strength=style_geometry_strength,
            style_opacity_strength=style_opacity_strength,
        )  # [B, N, C]
        if color_style_override is not None:
            # Bypasses style_adapter.appearance_mlp's (generalized, dataset-trained)
            # color prediction with a per-point (gamma, beta) directly optimized for
            # this exact (content, style) pair -- see lam/stylization/color_optimize.py.
            # Geometry (query_points) above is untouched, still from style_adapter.
            color_style = color_style_override
        image_feats_bchw = rearrange(image_feats, "b (h w) c -> b c h w", h=int(math.sqrt(image_feats.shape[1])))

        gs_model_list, query_points, flame_params, _ = self.renderer.forward_gs(gs_hidden_features=latent_points,
                                                query_points=query_points,
                                                flame_data=flame_params,
                                                additional_features={"image_feats": image_feats, "image": image[:, 0], "image_feats_bchw": image_feats_bchw},
                                                color_style=color_style)

        render_res_list = []
        for view_idx in range(num_views):
            render_res = self.renderer.forward_animate_gs(gs_model_list, 
                                                          query_points,
                                                          self.renderer.get_single_view_smpl_data(flame_params, view_idx), 
                                                          render_c2ws[:, view_idx:view_idx+1], 
                                                          render_intrs[:, view_idx:view_idx+1], 
                                                          render_h, 
                                                          render_w, 
                                                          render_bg_colors[:, view_idx:view_idx+1])
            render_res_list.append(render_res)

        out = defaultdict(list)
        for res in render_res_list:
            for k, v in res.items():
                out[k].append(v)
        for k, v in out.items():
            # print(f"out key:{k}")
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.concat(v, dim=1)
                if k in ["comp_rgb", "comp_mask", "comp_depth"]:
                    out[k] = out[k][0].permute(0, 2, 3, 1)  # [1, Nv, 3, H, W] -> [Nv, 3, H, W] - > [Nv, H, W, 3] 
            else:
                out[k] = v
        out['cano_gs_lst'] = gs_model_list
        return out
