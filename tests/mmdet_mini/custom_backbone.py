import torch
import torch.nn as nn

from mmdet.registry import MODELS


@MODELS.register_module()
class TinyCustomBackbone(nn.Module):
    """MMDetection registry 등록 테스트용 간단한 backbone.

    입력:
        x: Tensor, shape [B, 3, H, W]

    출력:
        tuple(feature,)
        MMDetection backbone은 보통 feature map들을 tuple/list로 반환한다.
    """

    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feat = self.conv(x)
        return (feat,)
