import os
import sys

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_train_runner_module_can_be_imported():
    from lam.runners.train.lam import parse_configs

    assert callable(parse_configs)


def test_minimal_config_uses_style_images(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["python", "train.lam", "--config", "configs/train/lam_minimal.yaml"],
    )

    from lam.runners.train.lam import parse_configs

    cfg = parse_configs()

    assert cfg.training_mode == "style_images"
    assert cfg.epochs == 150
    assert cfg.batch_size == 2
    assert cfg.style_image_paths == [
        "./style_image.png",
        "./style_image1.png",
        "./style_image2.png",
        "./style_image3.png",
    ]
