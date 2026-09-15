# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors. All rights reserved.
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
import traceback
import time
import torch
import os
import json
import argparse
import random
import mcubes
import trimesh
import numpy as np
from PIL import Image
from glob import glob
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from accelerate.logging import get_logger

from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image


from .base_inferrer import Inferrer
from lam.datasets.cam_utils import build_camera_principle, build_camera_standard, surrounding_views_linspace, create_intrinsics
from lam.utils.logging import configure_logger
from lam.runners import REGISTRY_RUNNERS
from lam.utils.video import images_to_video
from lam.utils.hf_hub import wrap_model_hub
from lam.models.modeling_lam import ModelLAM
from lam.runners.train.lam import TinyStyleAutoEncoder
from safetensors.torch import load_file
import moviepy.editor as mpy
from tools.flame_tracking_single_image import FlameTrackingSingleImage
from lam.stylization import prepare_style_reference, stylize_frames_with_reference
from lam.stylization.color_optimize import optimize_style_colors
from lam.stylization.emotion_optimize import optimize_emotion_expr
from lam.models.motion_symmetry import symmetrize_motion, expr_symmetry_matrix
from lam.models.emotion_adapter import (
    EMOTION_CLASSES, EmotionAdapter, default_emotion_strength, default_emotion_adapter_path,
    default_emotion_optimize_reg, default_emotion_optimize_reg_jaw,
    CLASSES_WITH_DISTINCTIVE_SHARPENING, distinctive_component_mask,
)


logger = get_logger(__name__)


def _seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _cfg_float_by_class(cfg, key, emotion_class, default):
    value = cfg.get(key, default)
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        value = value.get(emotion_class, value.get("default", default))
    return float(value)


def parse_configs():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--infer', type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    # parse from ENV
    if os.environ.get('APP_INFER') is not None:
        args.infer = os.environ.get('APP_INFER')
    if os.environ.get('APP_MODEL_NAME') is not None:
        cli_cfg.model_name = os.environ.get('APP_MODEL_NAME')

    if args.config is not None:
        cfg = OmegaConf.load(args.config)
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        cfg.render_size = cfg_train.dataset.render_image.high

        model_name = cli_cfg.get('model_name', None)
        if model_name is None:
            model_name = cfg.get('model_name', None)
        if model_name is None:
            model_name = ''

        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(model_name).split('_')[-1] if model_name else 'default',
        )

        cfg.save_tmp_dump = os.path.join("exps", 'save_tmp', _relative_path)
        cfg.image_dump = os.path.join("exps", 'images', _relative_path)
        cfg.video_dump = os.path.join("exps", 'videos', _relative_path)
        cfg.mesh_dump = os.path.join("exps", 'meshes', _relative_path)
        
    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault("save_tmp_dump", os.path.join("exps", cli_cfg.model_name, 'save_tmp'))
        cfg.setdefault("image_dump", os.path.join("exps", cli_cfg.model_name, 'images'))
        cfg.setdefault('video_dump', os.path.join("dumps", cli_cfg.model_name, 'videos'))
        cfg.setdefault('mesh_dump', os.path.join("dumps", cli_cfg.model_name, 'meshes'))
    
    cfg.merge_with(cli_cfg)
    if cfg.get('seed', None) is None and cfg.get('experiment', None) is not None:
        cfg.seed = cfg.experiment.get('seed', 42)
    cfg.setdefault('seed', 42)
    cfg.setdefault('model_name', None)
    cfg.setdefault('image_input', './assets/sample_input/status.png')
    cfg.setdefault('export_video', True)
    cfg.setdefault('export_mesh', False)
    cfg.setdefault('motion_seqs_dir', './assets/sample_motion/export/Look_In_My_Eyes/')
    cfg.setdefault('motion_img_dir', None)
    cfg.setdefault('motion_img_need_mask', True)
    cfg.setdefault('vis_motion', False)
    cfg.setdefault('render_fps', 30)
    cfg.setdefault('motion_video_read_fps', 30)
    cfg.setdefault('save_ply', False)
    cfg.setdefault('save_img', False)
    cfg.setdefault('cross_id', False)
    cfg.setdefault('test_sample', False)
    cfg.setdefault('gaga_track_type', '')
    # Captured before the fallback below fills it in -- './style_image.png'
    # (the fallback) happens to be a real leftover file in this project's
    # root from earlier testing, so checking existence AFTER defaulting can't
    # tell "no style requested" from "style requested"; see emotion_optimize's
    # default below, which needs that distinction.
    _user_requested_style = cfg.get('style_image_path', None) not in (None, '')
    cfg.setdefault('style_image_path', './style_image.png')
    # Raised 0.75 -> 1.0 after a 20-style before/after measurement
    # (tools/check_style_color.py's directional-magnitude metric): mean
    # style match 41.9% -> 53.2%, median 42.4% -> 59.0%, 18/20 styles
    # improved (+7 to +25 points each). The 2 regressions (s06, s07 in that
    # sweep) were already wrong-direction at 0.75 too -- raising strength
    # amplifies whatever direction the color-shift estimate landed on, so it
    # helps a good estimate and hurts a bad one, but doesn't change which
    # side of zero a style lands on. See STYLE_GEOMETRY_PIPELINE.md.
    cfg.setdefault('style_strength', 1.0)
    cfg.setdefault('style_internal_appearance_strength', 1.0)
    cfg.setdefault('style_geometry_strength', 0.85)
    cfg.setdefault('style_opacity_strength', 0.45)
    cfg.setdefault('style_track_geometry', True)
    cfg.setdefault('style_autoencoder_path', None)
    cfg.setdefault('style_autoencoder_base_channels', 48)
    cfg.setdefault('style_autoencoder_latent_channels', 128)
    cfg.setdefault('emotion_class', None)
    # History: these were raised 0.8/0.1 -> 1.3/0.5 because the emotion read
    # too weakly across a real driving clip. That was treating a symptom.
    # The real cause was EmotionAdapter's max_expr_delta cap sitting at 0.6,
    # below the 1.07-2.54 per-component range of the tracked targets, so the
    # expr delta came out clipped AND reshaped -- surprise lost ~35% of its
    # amplitude and half of what remained migrated out of the leading FLAME
    # expression components into the near-noise tail. That is also why
    # jaw_influence looked like "the more impactful of the two knobs": jaw
    # was never clipped (its targets peak at 0.084 rad against a 0.15 cap),
    # so it was the only one of the two still doing what it said. Raising
    # emotion_strength could not fix it -- it scales AFTER the cap, so it
    # just amplified a mis-shaped vector, and at strength 1.5 it bought no
    # more actual mesh motion than the unamplified correct delta (0.81 mm
    # vs 0.84 mm mean vertex displacement).
    #
    # With the cap corrected to 3.0 and the adapter retrained, the delta is
    # right-shaped (cosine to the tracked per-class target 0.91 -> 0.996 for
    # surprise) and ~1.4x larger, so 1.0 here now delivers slightly more
    # than the old 1.3 did. Keep jaw_influence at 0.5, and watch it on
    # speech-heavy content: more jaw influence means more of the driving
    # video's real mouth articulation gets overridden.
    # Per-class, not one flat number: emotions with a naturally subtle
    # anatomical signal (sad's frown vs. surprise's dropped jaw) need more
    # push at the same perceived strength -- see
    # emotion_adapter.PER_CLASS_DEFAULT_STRENGTH for the measured evidence
    # behind this (and why the other classes are still on the global
    # default rather than a guessed value). cfg.setdefault is a no-op if the
    # CLI already passed emotion_strength/emotion_jaw_influence explicitly,
    # so an explicit override always wins over this lookup.
    _default_strength, _default_jaw_influence = default_emotion_strength(cfg.get('emotion_class', None))
    cfg.setdefault('emotion_strength', _default_strength)
    cfg.setdefault('emotion_jaw_influence', _default_jaw_influence)
    # Per-class, same precedence rule as emotion_strength above (explicit CLI
    # value always wins). No single trained checkpoint was best for every
    # class -- see emotion_adapter.PER_CLASS_ADAPTER_PATH for the measured
    # evidence (hybrid best for sad/happy/neutral/surprise, a separately
    # trained "discriminative" checkpoint is what makes fear's sharpening
    # work at all). The global default (hybrid) is trained on AffectNet-HQ
    # references filtered against an independent pretrained expression
    # classifier (trpakov/vit-face-expression): kept only where it agreed
    # with the human label, or disagreed without much confidence (< 0.6) --
    # dropped the ones it confidently called something else (mostly
    # "neutral" for photos humans had labeled "sad", at 0.9+ confidence).
    # 'disgust' is exempt from this filter and stays 100% human-labeled,
    # since the pretrained model has ~zero usable signal for it (1/99
    # agreement) -- filtering by its opinion there would just delete the
    # class, not clean it.
    cfg.setdefault('emotion_adapter_path', default_emotion_adapter_path(cfg.get('emotion_class', None)))
    cfg.setdefault('emotion_sharpen_top_k', 8)
    cfg.setdefault('emotion_sharpen_boost', 2.5)
    # Default True for UNSTYLED requests only, as of the 196-run (14
    # identities x 7 classes, no style) validation: every class improved,
    # zero regressions -- mean confidence 34.3% -> 62.1%, match rate
    # 39.8% -> 67.3%. Notably overturned the earlier conclusion that anger
    # was a hard rendering ceiling (19.0% -> 84.4%, match 21.4% -> 92.9%):
    # the direction wasn't unreachable, it just needed to be
    # identity-specific rather than a bigger push in one fixed direction.
    # Disgust barely moved (0.2% -> 5.7%) and remains the one real outlier,
    # consistent with it having no strong distinctive direction to converge
    # toward in the first place. Cost is small (measured +3.7s/request at the
    # default 30 steps -- the expensive backbone pass is computed once and
    # reused, same as optimize_style_colors).
    #
    # Left OFF by default when a style image is actually present. A follow-up
    # fix threaded the style-blended face SHAPE into the optimizer (it was
    # previously correcting against the content's own unstyled shape, which
    # measurably regressed styled requests -- e.g. one 'neutral' case dropped
    # 96.8% -> 0.9%). That fix helped inside the optimizer's own proxy render,
    # but a second, separate mismatch surfaced testing it: the proxy renders
    # with style COLOR zeroed out (matching how optimize_style_colors isolates
    # geometry), while the real final image has the fully per-instance-
    # optimized style color applied afterward -- so the optimizer can converge
    # against a face that doesn't look like what actually ships. Measured
    # directly: the 96.8% -> 0.9% regression case above, re-tested with the
    # shape fix in place, still ended up wrong in the final render (0.3%
    # neutral, top1 'happy' 99.4%) despite the in-loop proxy reporting 98.1%
    # neutral. Not safe to default on for styled requests until that's
    # resolved too -- see STYLE_GEOMETRY_PIPELINE.md.
    cfg.setdefault('emotion_optimize', not _user_requested_style)
    # Raised 30 -> 100 after 'sad' was measured to need it: its loss barely
    # moves for the first ~30 steps then transitions sharply around step
    # 40-60 -- 30 steps alone stops before that transition and 'sad' stays
    # stuck near its bad starting point regardless of emotion_optimize_reg.
    # 'happy' (already fast-converging by step 20) is unaffected by the
    # extra steps -- confirmed stable at 98%+ from step 30 through 149, no
    # drift or new artifact from running longer.
    cfg.setdefault('emotion_optimize_steps', 100)
    # Projected gradient: constrain the test-time optimiser DURING the search
    # rather than clipping its answer afterwards. See optimize_emotion_expr.
    cfg.setdefault('emotion_optimize_project', True)
    cfg.setdefault('emotion_optimize_lr', 0.05)
    # No single value works for every class -- see PER_CLASS_OPTIMIZE_REG in
    # emotion_adapter.py for the measured evidence (a real user-caught
    # artifact on 'happy' that a uniform high value fixed, and a regression
    # on 'sad' that the SAME high value caused, requiring per-class values
    # the same way PER_CLASS_DEFAULT_STRENGTH already does for strength).
    cfg.setdefault('emotion_optimize_reg', default_emotion_optimize_reg(cfg.get('emotion_class', None)))
    # Separate, stronger-by-default penalty on jaw_delta specifically -- see
    # lambda_reg_jaw's docstring in lam/stylization/emotion_optimize.py.
    # jaw_delta is only 3 numbers with outsized visual leverage (controls
    # how wide the mouth opens across the whole face), so sharing one
    # combined penalty with expr_delta's 100 components let the optimizer
    # dump most of its "budget" into jaw and lean on a shared classifier
    # bias (mouth-open reads as more intense for almost every non-neutral
    # class) instead of finding each class's own distinctive shape.
    cfg.setdefault('emotion_optimize_reg_jaw', default_emotion_optimize_reg_jaw(cfg.get('emotion_class', None)))
    # Damps the driving clip's own left/right lateral bias, which every
    # avatar otherwise replays verbatim (see lam/models/motion_symmetry.py
    # for the per-clip measurements). 0.0 = off, 1.0 = fully symmetric
    # replayed expression. Default set from the visual + measured sweep
    # recorded in STYLE_GEOMETRY_PIPELINE.md.
    cfg.setdefault('motion_symmetry', 0.0)
    # Damps the lateral half of the learned emotion delta (35-72% of
    # each class's norm is asymmetric residual). 0.0 = off, 1.0 =
    # fully symmetric emotion. See the emotion block in infer_single.
    cfg.setdefault('emotion_symmetry', 0.0)
    # Final guard rail applied after the adapter and optional test-time
    # optimizer. It bounds exactly the two user-visible failure modes we keep
    # seeing: jaw opening too far, and lateral mouth/jaw drift. This is not a
    # learned emotion fix; it is an output safety clamp so a single bad
    # optimizer step cannot make the mouth unusable.
    cfg.setdefault('frontalize', False)
    cfg.setdefault('frontalize_eyes', True)
    cfg.setdefault('frontalize_pitch', 0.0)
    cfg.setdefault('render_scale', 1.0)
    cfg.setdefault('render_zoom', 1.0)
    cfg.setdefault('emotion_safety', True)
    cfg.setdefault('emotion_final_symmetry', 0.5)
    cfg.setdefault('emotion_max_expr_delta_norm', {
        'default': 6.0,
        'neutral': 0.8,
        'surprise': 7.0,
    })
    cfg.setdefault('emotion_max_jaw_open_delta', {
        'default': 0.04,
        'neutral': 0.01,
        'fear': 0.06,
        'surprise': 0.12,
    })
    cfg.setdefault('emotion_max_jaw_close_delta', 0.03)
    cfg.setdefault('emotion_max_jaw_lateral_delta', 0.01)
    # Damps the lateral asymmetry the Gaussian decoder invents in its own
    # per-point xyz offsets (measured: 37.3% of offset magnitude in the mouth
    # region, with style and emotion both disabled). Acts on the learned
    # residual only, never the FLAME surface, so the subject's real geometry
    # is preserved. 0.0 = off, 1.0 = fully symmetric offsets.
    cfg.setdefault('gs_mouth_symmetry', 0.0)
    cfg.setdefault('gs_symmetry_mouth_only', True)

    """
    [required]
    model_name: str
    image_input: str
    export_video: bool
    export_mesh: bool

    [special]
    source_size: int
    render_size: int
    video_dump: str
    mesh_dump: str

    [default]
    render_views: int
    render_fps: int
    mesh_size: int
    mesh_thres: float
    frame_size: int
    logger: str
    """

    cfg.setdefault('logger', 'INFO')

    if cfg.get('emotion_class', None):
        assert cfg.emotion_class in EMOTION_CLASSES, \
            f"emotion_class must be one of {EMOTION_CLASSES} or unset, got {cfg.emotion_class!r}"

    # assert not (args.config is not None and args.infer is not None), "Only one of config and infer should be provided"
    assert cfg.model_name is not None, "model_name is required"
    if not os.environ.get('APP_ENABLED', None):
        assert cfg.image_input is not None, "image_input is required"
        assert cfg.export_video or cfg.export_mesh, \
            "At least one of export_video or export_mesh should be True"
        cfg.app_enabled = False
    else:
        cfg.app_enabled = True

    return cfg


@REGISTRY_RUNNERS.register('infer.lam')
class LAMInferrer(Inferrer):

    EXP_TYPE: str = 'lam'

    def __init__(self):
        super().__init__()

        self.cfg = parse_configs()
        _seed_everything(self.cfg.get('seed', 42))
        """
        configure_logger(
            stream_level=self.cfg.logger,
            log_level=self.cfg.logger,
        )
        """

        self.model: LAMInferrer = self._build_model(self.cfg).to(self.device)

        # Decoder-offset symmetrization is read off the renderer at
        # forward_gs time; set it here rather than threading a config through
        # the whole model signature, since it is an inference-time correction
        # and not part of the trained architecture.
        self.model.renderer.gs_mouth_symmetry = float(self.cfg.get('gs_mouth_symmetry', 0.0))
        self.model.renderer.gs_symmetry_mouth_only = bool(self.cfg.get('gs_symmetry_mouth_only', True))

        self.flametracking = FlameTrackingSingleImage(output_dir='tracking_output',
                                             alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
                                             vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',
                                             human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
                                             facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',
                                             detect_iris_landmarks=True,
                                             args = self.cfg)
        self.style_autoencoder = None
        self._load_style_autoencoder()
        self.emotion_adapter = None
        self._load_emotion_adapter()


    def _build_model(self, cfg):
        """
        from lam.models import model_dict
        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])
        model = hf_model_cls.from_pretrained(cfg.model_name)
        """
        from lam.models import ModelLAM
        model = ModelLAM(**cfg.model)

        if os.path.isdir(cfg.model_name):
            resume = os.path.join(cfg.model_name, "model.safetensors")
        else:
            resume = cfg.model_name
        print("==="*16*3)
        print("loading pretrained weight from:", resume)
        if resume.endswith('safetensors'):
            ckpt = load_file(resume, device='cpu')
        else:
            ckpt = torch.load(resume, map_location='cpu')
            if isinstance(ckpt, dict) and 'model' in ckpt:
                ckpt = ckpt['model']
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict:
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    print(f"WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.")
            else:
                print(f"WARN] unexpected param {k}: {v.shape}")
        print("finish loading pretrained weight from:", resume)
        print("==="*16*3)
        return model

    def _load_style_autoencoder(self):
        style_autoencoder_path = self.cfg.get('style_autoencoder_path', None)
        if not style_autoencoder_path:
            return
        if not os.path.exists(style_autoencoder_path):
            print(f"Warning: style autoencoder checkpoint not found: {style_autoencoder_path}")
            return

        base_channels = int(self.cfg.get('style_autoencoder_base_channels', 48))
        latent_channels = int(self.cfg.get('style_autoencoder_latent_channels', 128))
        model = TinyStyleAutoEncoder(base_channels=base_channels, latent_channels=latent_channels)
        ckpt = torch.load(style_autoencoder_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys in style autoencoder checkpoint: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys in style autoencoder checkpoint: {unexpected}")
        model.eval()
        self.style_autoencoder = model.to(self.device)
        print(f"Loaded style autoencoder from {style_autoencoder_path}")

    def _load_emotion_adapter(self):
        # Only needed if a run actually requests emotion_class -- avoid
        # failing startup for users who never touch this feature just
        # because the checkpoint hasn't been trained yet.
        if not self.cfg.get('emotion_class', None):
            return
        emotion_adapter_path = self.cfg.get('emotion_adapter_path', None)
        if not emotion_adapter_path or not os.path.exists(emotion_adapter_path):
            raise FileNotFoundError(
                f"emotion_class={self.cfg.get('emotion_class')} requested but "
                f"emotion_adapter checkpoint not found at {emotion_adapter_path} "
                "(train it with tools/train_emotion_adapter.py first)."
            )
        model = EmotionAdapter()
        state_dict = torch.load(emotion_adapter_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        self.emotion_adapter = model.to(self.device)
        print(f"Loaded emotion adapter from {emotion_adapter_path}")

    def _style_optimize_cache_path(self, image_path, style_image_path, style_opt_kwargs):
        # Style color optimization has no dependency on emotion_class (it
        # operates on shape/color; emotion only ever touches expr/jaw_pose --
        # see modeling_lam.py's blend functions) or on any other per-request
        # setting outside this dict, so the exact same (content, style,
        # settings) triple always converges to the same result -- re-running
        # it is redundant work, not fresh signal. Key on absolute paths (not
        # mtimes) deliberately: this cache is meant to survive across CLI
        # invocations in the same session/benchmark run, and content churn on
        # these files mid-run isn't a case worth handling.
        import hashlib
        key_obj = {
            "image_path": os.path.abspath(image_path),
            "style_image_path": os.path.abspath(style_image_path),
            **style_opt_kwargs,
        }
        key = hashlib.sha1(json.dumps(key_obj, sort_keys=True).encode()).hexdigest()[:16]
        cache_dir = self.cfg.get("style_optimize_cache_dir", "exps/style_color_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{key}.pt")

    def _load_style_optimize_cache(self, cache_path):
        if not cache_path or not os.path.exists(cache_path):
            return None
        try:
            data = torch.load(cache_path, map_location=self.device)
            return data["color_gamma"], data["color_beta"]
        except Exception as exc:
            print(f"Warning: failed to load style-optimize cache {cache_path}, re-optimizing: {exc}")
            return None

    def _save_style_optimize_cache(self, cache_path, color_gamma, color_beta):
        torch.save({"color_gamma": color_gamma.detach().cpu(), "color_beta": color_beta.detach().cpu()}, cache_path)
        print(f"cached per-instance style colors to {cache_path}")

    def _apply_style_autoencoder(self, style_image: torch.Tensor) -> torch.Tensor:
        if self.style_autoencoder is None:
            return style_image
        with torch.no_grad():
            if style_image.dim() == 4:
                style_image = style_image.to(self.device)
            else:
                raise ValueError('Style image tensor must be 4D [B,C,H,W]')
            reconstructed = self.style_autoencoder(style_image)
            return reconstructed.clamp(0.0, 1.0)

    def _apply_emotion_safety(self, emotion_class, expr_delta, jaw_delta,
                             clamp_only=False):
        """Constrain an emotion delta to the safe region.

        `clamp_only=True` applies ONLY the hard clamps (expr norm, jaw box).
        Those are true projections -- idempotent, so re-applying them every
        optimiser step is well defined, which is what projected gradient
        descent needs. The symmetry blend is deliberately excluded from that
        path: it is a contraction, not a projection
        (x <- (1-a)x + a*Mx), so iterating it 100 times drives the delta into
        the fully symmetric subspace instead of holding it at the requested
        50% blend. It stays a single post-optimisation step.
        """
        if not bool(self.cfg.get("emotion_safety", True)):
            return expr_delta, jaw_delta

        expr_delta = expr_delta.clone()
        jaw_delta = jaw_delta.clone()

        final_symmetry = 0.0 if clamp_only else float(self.cfg.get("emotion_final_symmetry", 0.0))
        if final_symmetry:
            sym_matrix = expr_symmetry_matrix(
                self.model.renderer.flame_model, expr_delta.device
            ).to(expr_delta.dtype)
            expr_delta_sym = expr_delta @ sym_matrix.transpose(0, 1)
            expr_delta = expr_delta * (1.0 - final_symmetry) + expr_delta_sym * final_symmetry
            jaw_delta[..., 1] *= (1.0 - final_symmetry)
            jaw_delta[..., 2] *= (1.0 - final_symmetry)

        max_expr_norm = _cfg_float_by_class(
            self.cfg, "emotion_max_expr_delta_norm", emotion_class, 6.0
        )
        if max_expr_norm > 0:
            norm = expr_delta.norm(dim=-1, keepdim=True)
            scale = torch.clamp(max_expr_norm / norm.clamp_min(1e-8), max=1.0)
            expr_delta = expr_delta * scale

        max_open = _cfg_float_by_class(
            self.cfg, "emotion_max_jaw_open_delta", emotion_class, 0.04
        )
        max_close = _cfg_float_by_class(
            self.cfg, "emotion_max_jaw_close_delta", emotion_class, 0.03
        )
        max_lateral = _cfg_float_by_class(
            self.cfg, "emotion_max_jaw_lateral_delta", emotion_class, 0.01
        )
        if max_open > 0 or max_close > 0:
            jaw_delta[..., 0] = jaw_delta[..., 0].clamp(min=-max_close, max=max_open)
        if max_lateral >= 0:
            jaw_delta[..., 1] = jaw_delta[..., 1].clamp(min=-max_lateral, max=max_lateral)
            jaw_delta[..., 2] = jaw_delta[..., 2].clamp(min=-max_lateral, max=max_lateral)

        return expr_delta, jaw_delta

    def _default_source_camera(self, dist_to_center: float = 2.0, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        # return: (N, D_cam_raw)
        canonical_camera_extrinsics = torch.tensor([[
            [1, 0, 0, 0],
            [0, 0, -1, -dist_to_center],
            [0, 1, 0, 0],
        ]], dtype=torch.float32, device=device)
        canonical_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0)
        source_camera = build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)
        return source_camera.repeat(batch_size, 1)

    def _default_render_cameras(self, n_views: int, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        # return: (N, M, D_cam_render)
        render_camera_extrinsics = surrounding_views_linspace(n_views=n_views, device=device)
        render_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0).repeat(render_camera_extrinsics.shape[0], 1, 1)
        render_cameras = build_camera_standard(render_camera_extrinsics, render_camera_intrinsics)
        return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)

    def infer_planes(self, image: torch.Tensor, source_cam_dist: float):
        N = image.shape[0]
        source_camera = self._default_source_camera(dist_to_center=source_cam_dist, batch_size=N, device=self.device)
        planes = self.model.forward_planes(image, source_camera)
        assert N == planes.shape[0]
        return planes

    def infer_video(self, planes: torch.Tensor, frame_size: int, render_size: int, render_views: int, render_fps: int, dump_video_path: str):
        N = planes.shape[0]
        render_cameras = self._default_render_cameras(n_views=render_views, batch_size=N, device=self.device)
        render_anchors = torch.zeros(N, render_cameras.shape[1], 2, device=self.device)
        render_resolutions = torch.ones(N, render_cameras.shape[1], 1, device=self.device) * render_size
        render_bg_colors = torch.ones(N, render_cameras.shape[1], 1, device=self.device, dtype=torch.float32) * 0. # 1.

        frames = []
        for i in range(0, render_cameras.shape[1], frame_size):
            frames.append(
                self.model.synthesizer(
                    planes=planes,
                    cameras=render_cameras[:, i:i+frame_size],
                    anchors=render_anchors[:, i:i+frame_size],
                    resolutions=render_resolutions[:, i:i+frame_size],
                    bg_colors=render_bg_colors[:, i:i+frame_size],
                    region_size=render_size,
                )
            )
        # merge frames
        frames = {
            k: torch.cat([r[k] for r in frames], dim=1)
            for k in frames[0].keys()
        }
        # dump
        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)
        for k, v in frames.items():
            if k == 'images_rgb':
                images_to_video(
                    images=v[0],
                    output_path=dump_video_path,
                    fps=render_fps,
                    gradio_codec=self.cfg.app_enabled,
                )

    def infer_mesh(self, planes: torch.Tensor, mesh_size: int, mesh_thres: float, dump_mesh_path: str):
        grid_out = self.model.synthesizer.forward_grid(
            planes=planes,
            grid_size=mesh_size,
        )
        
        vtx, faces = mcubes.marching_cubes(grid_out['sigma'].squeeze(0).squeeze(-1).cpu().numpy(), mesh_thres)
        vtx = vtx / (mesh_size - 1) * 2 - 1

        vtx_tensor = torch.tensor(vtx, dtype=torch.float32, device=self.device).unsqueeze(0)
        vtx_colors = self.model.synthesizer.forward_points(planes, vtx_tensor)['rgb'].squeeze(0).cpu().numpy()  # (0, 1)
        vtx_colors = (vtx_colors * 255).astype(np.uint8)
        
        mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)

        # dump
        os.makedirs(os.path.dirname(dump_mesh_path), exist_ok=True)
        mesh.export(dump_mesh_path)

    def add_audio_to_video(self, video_path, out_path, audio_path):
        from moviepy.editor import VideoFileClip, AudioFileClip
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        video_clip_with_audio = video_clip.set_audio(audio_clip)
        video_clip_with_audio.write_videofile(out_path, codec='libx264', audio_codec='aac')
        print(f"Audio added successfully at {out_path}")

    def save_imgs_2_video(self, img_lst, v_pth, fps):
        from moviepy.editor import ImageSequenceClip
        images = [image.astype(np.uint8) for image in img_lst]
        clip = ImageSequenceClip(images, fps=fps)
        clip.write_videofile(v_pth, codec='libx264')
        print(f"Video saved successfully at {v_pth}")
    
    def infer_single(self, image_path: str,
                     motion_seqs_dir, 
                     motion_img_dir,
                     motion_video_read_fps,
                     export_video: bool, 
                     export_mesh: bool, 
                     dump_tmp_dir:str,  # require by extracting motion seq from video, to save some results
                     dump_image_dir:str,
                     dump_video_path: str, 
                     dump_mesh_path: str,
                     gaga_track_type: str):
        source_size = self.cfg.source_size
        # The rasterised resolution is set by the camera intrinsics (see
        # modeling_lam.py:485, render_h = cy*2), not by cfg.render_size, so
        # raising the output resolution means scaling the intrinsics and
        # render_size together -- if they disagree the rasteriser renders a
        # crop instead of a larger image.
        render_scale = float(self.cfg.get("render_scale", 1.0))
        render_size = int(round(self.cfg.render_size * render_scale))
        render_fps = self.cfg.render_fps
        aspect_standard = 1.0/1.0
        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)  # False
        vis_motion = self.cfg.get("vis_motion", False)  # False
        save_ply = self.cfg.get("save_ply", False)  # False
        save_img = self.cfg.get("save_img", False)  # False
        rendered_bg = 1.
        ref_bg = 1.
        mask_path = image_path.replace("/images/", "/fg_masks/").replace(".jpg", ".png")
        if ref_bg < 1.:
            if "VFHQ_TEST" in image_path:
                mask_path = image_path.replace("/VFHQ_TEST/", "/mask/").replace("/images/", "/mask/").replace(".png", ".jpg")
            else:
                mask_path = image_path.replace("/vfhq_test_nooffset_export/", "/mask/").replace("/images/", "/mask/").replace(".png", ".jpg")
        if not os.path.exists(mask_path):
            print("Warning: Mask path not exists:", mask_path)
            mask_path = None
        else:
            print("load mask from:", mask_path)

        image, _, _, shape_param = preprocess_image(image_path, mask_path=mask_path, intr=None, pad_ratio=0, bg_color=ref_bg, 
                                             max_tgt_size=None, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
                                             render_tgt_size=source_size, multiply=14, need_mask=True, get_shape_param=True)
        # save masked image for vis
        save_ref_img_path = os.path.join(dump_tmp_dir, "refer_" + os.path.basename(image_path))
        vis_ref_img = (image[0].permute(1, 2 ,0).cpu().detach().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_ref_img).save(save_ref_img_path)
        # prepare motion seq
        test_sample=self.cfg.get("test_sample", False)
        # test_sample=True
        src = image_path.split('/')[-3]
        driven = motion_seqs_dir.split('/')[-2]
        src_driven = [src, driven]
        motion_seq = prepare_motion_seqs(motion_seqs_dir, motion_img_dir, save_root=dump_tmp_dir, fps=motion_video_read_fps,
                                            bg_color=rendered_bg, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1,0],
                                            render_image_res=render_size,  multiply=16, 
                                            need_mask=motion_img_need_mask, vis_motion=vis_motion, 
                                            shape_param=shape_param, test_sample=test_sample, cross_id=self.cfg.get("cross_id", False), src_driven=src_driven)

        # return

        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)

        # Zoom is the focal length alone (intrinsics [0,0] and [1,1]); cx/cy
        # are left untouched because render_h/render_w are derived from them,
        # so touching those would resize the canvas instead of zooming in it.
        # Needed because the frontalised head sits high enough to clip at the
        # top edge; pulling back leaves margin on every side, after which the
        # figure builder can centre the content exactly.
        render_zoom = float(self.cfg.get("render_zoom", 1.0))
        if render_zoom != 1.0:
            motion_seq["render_intrs"][..., 0, 0] *= render_zoom
            motion_seq["render_intrs"][..., 1, 1] *= render_zoom
            print(f"render_zoom={render_zoom}")

        if render_scale != 1.0:
            # rows 0 and 1 carry (fl_x, cx) and (fl_y, cy)
            motion_seq["render_intrs"][..., 0, :] *= render_scale
            motion_seq["render_intrs"][..., 1, :] *= render_scale
            print(f"render_scale={render_scale} -> {render_size}x{render_size}")

        # Neutralize the driving clip's head orientation so the rendered face
        # looks straight at the camera. The driving sequence carries its own
        # head pose in `rotation` (FLAME root) and `neck_pose`; on
        # Look_In_My_Eyes frame 0 that is 20.8 deg combined (pitch 14.7, yaw
        # 5.8, roll 13.5), which is why every still came out tilted and turned
        # slightly to one side. Picking a different frame does not fix it --
        # the most frontal frame in that clip is still 8.2 deg off -- so the
        # rotation is zeroed outright. Expression and jaw are untouched, so
        # the emotion is unaffected; only the rigid head transform changes.
        # eyes_pose is zeroed too by default, otherwise the gaze stays off-axis
        # on an otherwise frontal face, which reads as a squint.
        if bool(self.cfg.get("frontalize", False)):
            _fp = motion_seq["flame_params"]
            _zeroed = []
            for _k in ("rotation", "neck_pose"):
                if _k in _fp:
                    _fp[_k] = torch.zeros_like(_fp[_k]); _zeroed.append(_k)
            # Zero rotation is canonical FLAME, which under this clip's camera
            # sits pitched slightly back (nostrils visible). A small positive
            # pitch on the root brings the chin down to a natural straight-on
            # framing; axis 0 is pitch, positive = chin toward chest.
            _pitch = float(self.cfg.get("frontalize_pitch", 0.0))
            if _pitch and "rotation" in _fp:
                _fp["rotation"][..., 0] = _pitch
                _zeroed.append(f"pitch<-{_pitch:+.3f}")
            if bool(self.cfg.get("frontalize_eyes", True)) and "eyes_pose" in _fp:
                _fp["eyes_pose"] = torch.zeros_like(_fp["eyes_pose"]); _zeroed.append("eyes_pose")
            print(f"frontalize: zeroed {', '.join(_zeroed)}")

        # Strip the driving clip's own left/right bias before anything else
        # reads these params -- the emotion block below (and the per-instance
        # optimizer inside it) should correct against motion that is already
        # free of it, not inherit it and then optimize on top. Applied here
        # rather than inside prepare_motion_seqs so it stays one visible,
        # configurable step; see lam/models/motion_symmetry.py for the
        # measurement that traced the visible "lower lip drifts to one side"
        # artifact to the driving clips themselves.
        motion_symmetry_strength = float(self.cfg.get("motion_symmetry", 0.0))
        if motion_symmetry_strength:
            motion_seq["flame_params"] = symmetrize_motion(
                motion_seq["flame_params"],
                self.model.renderer.flame_model,
                motion_symmetry_strength,
            )

        # Moved ahead of the emotion block (was originally right before the
        # style-color-optimize section below) so emotion_optimize can correct
        # expr/jaw against the SAME (possibly style-blended) face shape the
        # final render will actually use, not just the content's own shape --
        # see optimize_emotion_expr's style_shape_params/style_geometry_strength
        # args and STYLE_GEOMETRY_PIPELINE.md's "known limitation" note this
        # closes. Only the geometry reference is needed this early; the
        # (expensive) color/appearance side of style still happens below,
        # unchanged, since color has no bearing on emotion.
        style_image_path = self.cfg.get("style_image_path", "./style_image.png")
        style_track_geometry = self.cfg.get("style_track_geometry", True)
        style_geometry_strength = self.cfg.get("style_geometry_strength", 0.85)
        style_opacity_strength = self.cfg.get("style_opacity_strength", 0.45)
        style_image, style_shape_param = prepare_style_reference(
            style_image_path,
            self.flametracking,
            source_size=source_size,
            aspect_standard=aspect_standard,
            dump_tmp_dir=dump_tmp_dir,
            bg_color=ref_bg,
            track_geometry=style_track_geometry,
        )
        # Snapshot the tracked head crop BEFORE the autoencoder touches it --
        # this is what the LAB post-process below should measure its target
        # colors from, not the raw style file (see color_transfer.py).
        style_ref_rgb = None
        if style_image is not None:
            style_ref_rgb = (style_image[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        # Also moved ahead of the emotion block, for the same reason as the
        # shape reference above: emotion_optimize's proxy render needs the
        # SAME color the final render will actually use, not the raw
        # (unstyled) network prediction. Previously this ran after the
        # emotion block and the optimizer's proxy always used
        # style_appearance_strength=0.0 (zeroed color) -- measured to cause
        # a second, separate mismatch on top of the shape one: a case whose
        # in-loop proxy reported 98.1% confidence scored 0.3% on the actual
        # final (real-color) render. See STYLE_GEOMETRY_PIPELINE.md.
        if style_image is not None and self.style_autoencoder is not None:
            style_image = self._apply_style_autoencoder(style_image.to(self.device))
        device = self.device
        dtype=torch.float32
        # dtype=torch.bfloat16
        self.model.to(dtype)

        color_style_override = None
        if style_image is not None and self.cfg.get("style_optimize_color", False):
            # style_adapter.appearance_mlp tries to learn one network that
            # generalizes a color transform across many (content, style) pairs from
            # limited training data; for a single content+style pair at inference
            # time, directly optimizing per-point Gaussian color against this exact
            # style image's Gram/AdaIN statistics converges to the real target
            # instead of hoping the network generalized -- see
            # lam/stylization/color_optimize.py.
            opt_steps = int(self.cfg.get("style_optimize_steps", 400))
            num_total_views = motion_seq["render_c2ws"].shape[1]
            num_opt_views = min(int(self.cfg.get("style_optimize_num_views", 6)), num_total_views)
            # Spread the sampled views across the whole motion sequence (different
            # head poses/angles), not just the first frame -- see the comment in
            # color_optimize.py on why a single view causes per-point color noise.
            view_indices = [round(i * (num_total_views - 1) / max(1, num_opt_views - 1)) for i in range(num_opt_views)]

            style_opt_kwargs = dict(
                style_geometry_strength=style_geometry_strength,
                style_opacity_strength=style_opacity_strength,
                num_steps=opt_steps,
                lr=float(self.cfg.get("style_optimize_lr", 0.03)),
                max_color_scale=float(self.cfg.get("style_optimize_max_color_scale", 0.9)),
                max_darken_ratio=float(self.cfg.get("style_optimize_max_darken_ratio", 0.5)),
                lambda_style=float(self.cfg.get("style_optimize_style_weight", 2.0e3)),
                lambda_adain=float(self.cfg.get("style_optimize_adain_weight", 5.0)),
                lambda_content=float(self.cfg.get("style_optimize_content_weight", 0.3)),
                lambda_tv=float(self.cfg.get("style_optimize_tv_weight", 1.0e-4)),
                lambda_color_reg=float(self.cfg.get("style_optimize_color_reg_weight", 0.3)),
                lambda_color_uniformity=float(self.cfg.get("style_optimize_color_uniformity_weight", 3.0)),
                lambda_anatomy=float(self.cfg.get("style_optimize_anatomy_weight", 60.0)),
                anatomy_symmetrize=float(self.cfg.get("style_optimize_symmetrize", 1.0)),
                lambda_color_smoothness=float(self.cfg.get("style_optimize_smoothness_weight", 150.0)),
                smoothness_k=int(self.cfg.get("style_optimize_smoothness_k", 8)),
                num_opt_views=num_opt_views,
            )
            cache_path = None
            if bool(self.cfg.get("style_optimize_cache", True)):
                cache_path = self._style_optimize_cache_path(image_path, style_image_path, style_opt_kwargs)
            cached = self._load_style_optimize_cache(cache_path) if cache_path else None
            if cached is not None:
                print(f"reusing cached per-instance style colors from {cache_path} "
                      "(same content+style+settings already optimized this session -- "
                      "style color doesn't depend on emotion_class, so this is exact, not an approximation)")
                color_gamma, color_beta = cached
            else:
                print("optimizing per-instance style colors...................")
                def _log_opt(step, loss_dict):
                    # Gram-matrix MSE runs around 1e-5, which is why lambda_style
                    # defaults to 2e3; printed as %.4f it always read "0.0000" and
                    # looked like a dead loss term.
                    print(f"  [style-optimize] step={step} content={loss_dict['content']:.4f} "
                          f"style={loss_dict['style']:.3e} adain={loss_dict['adain']:.4f} "
                          f"anatomy={loss_dict.get('anatomy', 0.0):.4f} "
                          f"smoothness={loss_dict.get('color_smoothness', 0.0):.4f} "
                          f"total={loss_dict['total']:.4f}")
                color_gamma, color_beta = optimize_style_colors(
                    self.model,
                    image.to(device, dtype),
                    style_image.to(device, dtype),
                    flame_params={k: v.to(device) for k, v in motion_seq["flame_params"].items()},
                    render_c2ws=motion_seq["render_c2ws"][:, view_indices].to(device),
                    render_intrs=motion_seq["render_intrs"][:, view_indices].to(device),
                    render_bg_colors=motion_seq["render_bg_colors"][:, view_indices].to(device),
                    view_indices=view_indices,
                    render_h=render_size,
                    render_w=render_size,
                    style_shape_params=style_shape_param.unsqueeze(0).to(device) if style_shape_param is not None else None,
                    log_fn=_log_opt,
                    **{k: v for k, v in style_opt_kwargs.items() if k != "num_opt_views"},
                )
                if cache_path:
                    self._save_style_optimize_cache(cache_path, color_gamma, color_beta)
            # style_internal_appearance_strength previously had no effect here --
            # the optimized color unconditionally replaced style_adapter's own
            # (weak/unreliable) prediction regardless of this value, silently
            # ignoring it. Reusing it as a 0-1 blend factor on the optimized
            # result restores the expected "lower = weaker effect" control.
            blend = float(self.cfg.get("style_internal_appearance_strength", 0.75))
            color_style_override = (color_gamma * blend, color_beta * blend)

        emotion_class = self.cfg.get("emotion_class", None)
        if emotion_class:
            # A single additive edit to the whole driving clip's expr/jaw_pose,
            # applied once here -- independent of style (disjoint parameter
            # subspaces: style only ever touches betas/color, never expr/jaw)
            # and needs zero changes to modeling_lam.py/gs_renderer.py, since
            # flame_params flows through model.infer_single_view unsliced.
            # See lam/models/emotion_adapter.py.
            with torch.no_grad():
                expr = motion_seq["flame_params"]["expr"].to(self.device)
                jaw = motion_seq["flame_params"]["jaw_pose"].to(self.device)
                emotion_id = torch.tensor(
                    [EMOTION_CLASSES.index(emotion_class)], device=self.device
                )
                # One context vector for the whole clip (mean expr across
                # frames) -- the adapter's delta is the same for every frame
                # of this request, so the emotion reads as one consistent
                # expression shift rather than flickering per frame.
                base_expr = expr.mean(dim=1)
                expr_delta, jaw_delta = self.emotion_adapter(emotion_id, base_expr)
                if emotion_class in CLASSES_WITH_DISTINCTIVE_SHARPENING:
                    # Uniformly scaling the whole 100-dim delta (emotion_strength
                    # alone) dilutes a class's one real distinctive signal across
                    # every component, most of which it shares with other classes
                    # -- see distinctive_component_mask. Only applied to classes
                    # confirmed to have a real distinctive direction to sharpen.
                    sharpen_mask = distinctive_component_mask(
                        self.emotion_adapter, emotion_class, base_expr,
                        top_k=int(self.cfg.get('emotion_sharpen_top_k', 8)),
                        boost=float(self.cfg.get('emotion_sharpen_boost', 2.5)),
                    )
                    expr_delta = expr_delta * sharpen_mask[None, :]

                # Damp the LATERAL half of the learned delta. Measured on the
                # default checkpoint at the mean driving context, the
                # asymmetric residual (d - M@d) is 35-72% of each class's
                # delta norm -- anger 59.5%, happy 52.6%, surprise 51.6%,
                # disgust 44.6%, fear 39.1%, sad 35.5% -- and each class also
                # carries its own jaw yaw (surprise -0.0123, happy -0.0064).
                # A prototypical anger/happy/surprise is anatomically
                # symmetric, so that residual is mostly the idiosyncratic
                # asymmetry of whichever reference photos survived trimming,
                # and it is why the visible sideways mouth pull CHANGES
                # DIRECTION with the emotion class rather than being one
                # fixed offset. Symmetrizing the adapter's prior (before the
                # per-instance optimizer, so the optimizer's own pull-toward-
                # init regularization then keeps it near a symmetric prior).
                emotion_symmetry = float(self.cfg.get("emotion_symmetry", 0.0))
                if emotion_symmetry:
                    sym_matrix = expr_symmetry_matrix(
                        self.model.renderer.flame_model, expr_delta.device
                    ).to(expr_delta.dtype)
                    expr_delta_sym = expr_delta @ sym_matrix.transpose(0, 1)
                    expr_delta = (
                        expr_delta * (1.0 - emotion_symmetry) + expr_delta_sym * emotion_symmetry
                    )
                    jaw_delta = jaw_delta.clone()
                    jaw_delta[..., 1] *= (1.0 - emotion_symmetry)
                    jaw_delta[..., 2] *= (1.0 - emotion_symmetry)
            # Fallback values here are unreachable in practice -- parse_configs
            # always sets these via default_emotion_strength() before this
            # runs -- kept only so this line doesn't lie if that ever changes.
            emotion_strength = float(self.cfg.get("emotion_strength", 1.0))
            emotion_jaw_influence = float(self.cfg.get("emotion_jaw_influence", 0.5))
            scaled_expr_delta = emotion_strength * expr_delta
            scaled_jaw_delta = emotion_strength * emotion_jaw_influence * jaw_delta

            emotion_optimize = bool(self.cfg.get("emotion_optimize", False))
            if emotion_optimize:
                # Per-instance correction for THIS face -- the trained
                # adapter has no notion of identity, so its raw prediction
                # can read as the wrong expression on some faces at any
                # strength/sign (measured directly: see
                # lam/stylization/emotion_optimize.py's module docstring).
                # Mirrors optimize_style_colors's per-instance pattern.
                print(f"optimizing per-instance emotion correction for '{emotion_class}'...................")
                # Fallback values here are unreachable in practice --
                # parse_configs always sets these (the reg one via
                # default_emotion_optimize_reg()) before this runs -- kept
                # only so these lines don't lie if that ever changes.
                opt_steps = int(self.cfg.get("emotion_optimize_steps", 100))
                opt_lr = float(self.cfg.get("emotion_optimize_lr", 0.05))
                opt_reg = float(self.cfg.get("emotion_optimize_reg", default_emotion_optimize_reg(emotion_class)))
                opt_reg_jaw = float(
                    self.cfg.get("emotion_optimize_reg_jaw", default_emotion_optimize_reg_jaw(emotion_class))
                )

                def _log_emo_opt(step, loss, prob, expr_drift=None, jaw_drift=None):
                    if step % 10 == 0 or step == opt_steps - 1:
                        extra = f" expr_drift={expr_drift:.3f} jaw_drift={jaw_drift:.3f}" if expr_drift is not None else ""
                        print(f"  [emotion-optimize] step={step} loss={loss:.4f} target_prob={prob*100:.1f}%{extra}")

                image_b = image.unsqueeze(0).to(self.device, torch.float32)
                scaled_expr_delta, scaled_jaw_delta = optimize_emotion_expr(
                    self.model,
                    image_b[:, 0],
                    {k: v.to(self.device) for k, v in motion_seq["flame_params"].items()},
                    motion_seq["render_c2ws"].to(self.device),
                    motion_seq["render_intrs"].to(self.device),
                    motion_seq["render_bg_colors"].to(self.device),
                    view_idx=0,
                    render_h=render_size,
                    render_w=render_size,
                    target_emotion=emotion_class,
                    init_expr_delta=scaled_expr_delta,
                    init_jaw_delta=scaled_jaw_delta,
                    # Keep the search inside the same region the final safety
                    # clamp enforces, instead of clipping its answer afterwards.
                    # clamp_only=True because the symmetry blend is a
                    # contraction and must not be iterated -- see
                    # _apply_emotion_safety.
                    project_fn=(
                        (lambda e, j: self._apply_emotion_safety(
                            emotion_class, e, j, clamp_only=True))
                        if bool(self.cfg.get("emotion_optimize_project", True))
                        else None
                    ),
                    # Correct against the SAME face shape AND color the final
                    # render will actually use -- if style_image_path is set,
                    # that's the style-blended shape/optimized color, not the
                    # content's own shape or a zeroed-color proxy. Two
                    # measured regressions this closes: shape (a 'neutral'
                    # case dropped 96.8% -> 0.9% when optimized against the
                    # content-only shape) and color (the same case, with only
                    # the shape fix, reported 98.1% in-loop but scored 0.3%
                    # on the real color-styled render) -- see
                    # STYLE_GEOMETRY_PIPELINE.md.
                    style_image=(style_image.to(self.device) if style_image is not None else None),
                    style_shape_params=(
                        style_shape_param.unsqueeze(0).to(self.device) if style_shape_param is not None else None
                    ),
                    style_geometry_strength=style_geometry_strength,
                    style_opacity_strength=style_opacity_strength,
                    color_style_override=color_style_override,
                    num_steps=opt_steps,
                    lr=opt_lr,
                    lambda_reg=opt_reg,
                    lambda_reg_jaw=opt_reg_jaw,
                    log_fn=_log_emo_opt,
                )

            scaled_expr_delta, scaled_jaw_delta = self._apply_emotion_safety(
                emotion_class,
                scaled_expr_delta,
                scaled_jaw_delta,
            )
            motion_seq["flame_params"]["expr"] = expr + scaled_expr_delta[:, None, :]
            motion_seq["flame_params"]["jaw_pose"] = jaw + scaled_jaw_delta[:, None, :]
            print(f"applied emotion '{emotion_class}' (strength={emotion_strength}, jaw_influence={emotion_jaw_influence}"
                  f"{', optimized' if emotion_optimize else ''})")

        start_time = time.time()
        print("start to inference...................")
        with torch.no_grad():
            # TODO check device and dtype
            res = self.model.infer_single_view(image.unsqueeze(0).to(device, dtype), None, None,
                                               render_c2ws=motion_seq["render_c2ws"].to(device),
                                               render_intrs=motion_seq["render_intrs"].to(device),
                                               render_bg_colors=motion_seq["render_bg_colors"].to(device),
                                               flame_params={k:v.to(device) for k, v in motion_seq["flame_params"].items()},
                                               style_image=style_image.unsqueeze(0).to(device, dtype) if style_image is not None else None,
                                               style_shape_params=style_shape_param.unsqueeze(0).to(device) if style_shape_param is not None else None,
                                               style_appearance_strength=self.cfg.get("style_internal_appearance_strength", 0.75),
                                               style_geometry_strength=style_geometry_strength,
                                               style_opacity_strength=style_opacity_strength,
                                               color_style_override=color_style_override)

        print(f"time elapsed: {time.time() - start_time}")
        rgb = res["comp_rgb"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)
        mask = res["comp_mask"].detach().cpu().numpy()
        style_strength = self.cfg.get("style_strength", 0.75)
        rgb = stylize_frames_with_reference(
            rgb,
            style_image_path=style_image_path,
            masks=mask,
            strength=style_strength,
            style_rgb=style_ref_rgb,
        )
        only_pred = rgb
        Image.fromarray(only_pred[0]).save(os.path.join(dump_image_dir, "stylized_preview.png"))
        if vis_motion:
            # print(rgb.shape, motion_seq["vis_motion_render"].shape)
            import cv2
            vis_ref_img = np.tile(cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]), interpolation=cv2.INTER_AREA)[None, :, :, :], (rgb.shape[0], 1, 1, 1))
            blend_ratio = 0.7
            blend_res = ((1 -  blend_ratio) * rgb + blend_ratio * motion_seq["vis_motion_render"]).astype(np.uint8)
            # rgb = np.concatenate([rgb, motion_seq["vis_motion_render"], blend_res, vis_ref_img], axis=2)
            rgb = np.concatenate([vis_ref_img, rgb, motion_seq["vis_motion_render"]], axis=2)
            
        # export_video was previously ignored here: encoding and audio muxing ran
        # unconditionally, which is most of the wall time of a run whose only
        # wanted output is the still preview above. Honour the flag.
        if export_video:
            os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)
            # images_to_video(rgb, output_path=dump_video_path, fps=render_fps, gradio_codec=False, verbose=True)
            self.save_imgs_2_video(rgb, dump_video_path, render_fps)
            base_vid = motion_seqs_dir.strip('/').split('/')[-1]
            audio_path = os.path.join(motion_seqs_dir, base_vid+".wav")
            dump_video_path_wa = dump_video_path.replace(".mp4", "_audio.mp4")
            self.add_audio_to_video(dump_video_path, dump_video_path_wa, audio_path)
        if save_img and dump_image_dir is not None:
            for i in range(rgb.shape[0]):
                save_file = os.path.join(dump_image_dir, f"{i:04d}.png")
                Image.fromarray(only_pred[i]).save(save_file)
                if save_ply and dump_mesh_path is not None:
                    res["3dgs"][i][0][0].save_ply(os.path.join(dump_image_dir, f"{i:04d}.ply"))

            dump_cano_dir = "./exps/cano_gs/"
            if not os.path.exists(dump_cano_dir):
                os.system(f"mkdir -p {dump_cano_dir}")
            cano_ply_pth = os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + ".ply")
            # res['cano_gs_lst'][0].save_ply(cano_ply_pth, rgb2sh=True, offset2xyz=False)
            cano_ply_pth = os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + "_gs_offset.ply")
            res['cano_gs_lst'][0].save_ply(cano_ply_pth, rgb2sh=False, offset2xyz=True)
            # res['cano_gs_lst'][0].save_ply("tmp.ply", rgb2sh=False, offset2xyz=True)

            def save_color_points(points, colors, sv_pth, sv_fd="debug_vis/dataloader/"):
                points = points.squeeze().detach().cpu().numpy()
                colors = colors.squeeze().detach().cpu().numpy()
                sv_pth = os.path.join(sv_fd, sv_pth)
                if not os.path.exists(sv_fd):
                    os.system(f"mkdir -p {sv_fd}")
                with open(sv_pth, 'w') as of:
                    for point, color in zip(points, colors):
                        print('v', point[0], point[1], point[2], color[0], color[1], color[2], file=of)
 
            # save canonical color point clouds
            save_color_points(res['cano_gs_lst'][0].xyz, res["cano_gs_lst"][0].shs[:, 0, :], "framework_img.obj", sv_fd=dump_cano_dir) 

            # Export the template mesh to an OBJ file
            import trimesh
            vtxs = res['cano_gs_lst'][0].xyz - res['cano_gs_lst'][0].offset
            vtxs = vtxs.detach().cpu().numpy() 
            faces = self.model.renderer.flame_model.faces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(vertices=vtxs, faces=faces)
            mesh.export(os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + '_shaped_mesh.obj'))

            # Export textured deformed mesh
            import lam.models.rendering.utils.mesh_utils as mesh_utils
            vtxs = res['cano_gs_lst'][0].xyz.detach().cpu()
            faces = self.model.renderer.flame_model.faces.detach().cpu()
            colors = res['cano_gs_lst'][0].shs.squeeze(1).detach().cpu()
            pth = os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + '_textured_mesh.obj')
            print("Save textured mesh to:", pth)
            mesh_utils.save_obj(pth, vtxs, faces, textures=colors, texture_type="vertex")

    def infer(self):
        image_paths = []
        # hard code
        if os.path.isfile(self.cfg.image_input):
            omit_prefix = os.path.dirname(self.cfg.image_input)
            image_paths = [self.cfg.image_input]
        else:
            # ids = sorted(os.listdir(self.cfg.image_input))
            # image_paths = [os.path.join(self.cfg.image_input, e, "images/00000_00.png") for e in ids]
            image_paths = glob(os.path.join(self.cfg.image_input, "*.jpg"))
            omit_prefix = self.cfg.image_input

        # Fail loudly on a bad image_input instead of rendering nothing. A
        # mistyped path is not a file, so it falls to the glob branch above,
        # which returns [] for a directory that doesn't exist -- the loop then
        # runs zero times and the process exits 0 with no output and no error
        # ("0it [00:00, ?it/s]"). That silently burns a whole batch before
        # anyone notices the typo.
        if not image_paths:
            raise FileNotFoundError(
                f"image_input={self.cfg.image_input!r} matched no images: it is neither an "
                f"existing image file nor a directory containing *.jpg. Check the path."
            )

        gaga_track_type = ""

        for image_path in tqdm(image_paths, disable=not self.accelerator.is_local_main_process):
            try:

                # preprocess input image: segmentation, flame params estimation
                return_code = self.flametracking.preprocess(image_path)
                assert (return_code == 0), "flametracking preprocess failed!"
                return_code = self.flametracking.optimize()
                assert (return_code == 0), "flametracking optimize failed!"
                return_code, output_dir = self.flametracking.export()
                assert (return_code == 0), "flametracking export failed!"

                image_path = os.path.join(output_dir, "images/00000_00.png")
                # mask_path = image_path.replace("/images/", "/fg_masks/").replace(".jpg", ".png")

                motion_seqs_dir = self.cfg.motion_seqs_dir
                print("motion_seqs_dir:", motion_seqs_dir)
                # prepare dump paths
                image_name = os.path.basename(image_path)
                uid = image_name.split('.')[0]
                subdir_path = os.path.dirname(image_path).replace(omit_prefix, '')
                subdir_path = subdir_path[1:] if subdir_path.startswith('/') else subdir_path
                # hard code
                subdir_path = gaga_track_type
                uid = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
                print("subdir_path and uid:", subdir_path, uid)
                dump_video_path = os.path.join(
                    self.cfg.video_dump,
                    subdir_path,
                    f'{uid}.mp4',
                )
                dump_image_dir = os.path.join(
                    self.cfg.image_dump,
                    subdir_path,
                    f'{uid}'
                )
                dump_tmp_dir = os.path.join(
                    self.cfg.image_dump,
                    subdir_path,
                    "tmp_res"
                )
                dump_mesh_path = os.path.join(
                    self.cfg.mesh_dump,
                    subdir_path,
                    # f'{uid}.ply',
                )
                os.makedirs(dump_image_dir, exist_ok=True)
                os.makedirs(dump_tmp_dir, exist_ok=True)
                os.makedirs(dump_mesh_path, exist_ok=True)

                # if os.path.exists(dump_video_path):
                #     print(f"skip:{image_path}")
                #     continue

                self.infer_single(
                    image_path,
                    motion_seqs_dir=motion_seqs_dir,
                    motion_img_dir=self.cfg.motion_img_dir,
                    motion_video_read_fps=self.cfg.motion_video_read_fps,
                    export_video=self.cfg.export_video,
                    export_mesh=self.cfg.export_mesh, 
                    dump_tmp_dir=dump_tmp_dir,
                    dump_image_dir=dump_image_dir,
                    dump_video_path=dump_video_path, 
                    dump_mesh_path=dump_mesh_path,
                    gaga_track_type=gaga_track_type
                    )
            except:
                traceback.print_exc()
