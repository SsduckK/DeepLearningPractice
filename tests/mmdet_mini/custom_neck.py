import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class TinyFPNNeck(nn.Module):
    """간단한 FPN Neck.

    입력:
        inputs: tuple/list of feature maps
            예: [(B, C, H, W), ...]

    출력:
        tuple of feature maps
            예: [(B, out_channels, H, W), ...]
    """

    def __init__(self, in_channels=(16,), out_channels=32):
        super().__init__()

        if isinstance(in_channels, int):
            in_channels = [in_channels]

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for c in in_channels:
            self.lateral_convs.append(nn.Conv2d(c, out_channels, kernel_size=1))

            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, inputs):
        assert isinstance(inputs, (tuple, list))
        assert len(inputs) == len(self.in_channels)

        # 1x1 lateral conv
        laterals = [
            lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, inputs)
        ]

        # top-down path
        # feature가 여러 개 있을 때만 의미 있음
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[-2:]

            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode="nearest",
            )

        # 3x3 fpn conv
        outs = [
            fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)
        ]

        return tuple(outs)
