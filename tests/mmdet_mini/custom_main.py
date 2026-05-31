import torch

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet.registry import MODELS


def main():
    cfg = Config.fromfile("tiny_detector_config.py")

    if "custom_imports" in cfg:
        import_modules_from_strings(**cfg.custom_imports)

    print("Loaded config:")
    print(cfg.pretty_text)

    model = MODELS.build(cfg.model)
    print(model)

    x = torch.randn(2, 3, 224, 224)
    logits = model(x)

    print("logits shape:", logits.shape)

    assert logits.shape == (2, 3)
    print("[OK] TinyDetector build from config success")


if __name__ == "__main__":
    main()
