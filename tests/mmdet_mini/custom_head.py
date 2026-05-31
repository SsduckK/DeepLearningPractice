import torch
import torch.nn as nn

from mmdet.registry import MODELS


@MODELS.register_module()
class TinyClsHead(nn.Module):
    """간단한 3-class classification head.

    입력:
        feats: tuple/list of feature maps
            예: [(B, C, H, W), ...]

    출력:
        logits: Tensor, shape [B, num_classes]
    """

    def __init__(self, in_channels=32, num_classes=3):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, feats):
        assert isinstance(feats, (tuple, list))

        # 여기서는 첫 번째 FPN feature만 사용
        x = feats[0]

        x = self.global_pool(x)  # [B, C, 1, 1]
        x = torch.flatten(x, 1)  # [B, C]
        logits = self.fc(x)  # [B, num_classes]

        return logits
