import torch

# 중요:
# MODELS.build 전에 custom module 파일이 import 되어야 registry에 등록된다.
from custom_backbone import TinyCustomBackbone

from mmdet.registry import MODELS


def main():
    cfg = dict(
        type="TinyCustomBackbone",
        in_channels=3,
        out_channels=32,
    )

    model = MODELS.build(cfg)
    print(model)

    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)

    print("num outputs:", len(outputs))
    print("output shape:", outputs[0].shape)

    assert outputs[0].shape == (2, 32, 224, 224)
    print("[OK] TinyCustomBackbone registry/build/forward success")


if __name__ == "__main__":
    main()
