import torch.nn as nn

from mmdet.registry import MODELS


@MODELS.register_module()
class TinyDetector(nn.Module):
    """Backbone + Neck + Head를 이어붙이는 간단한 모델."""

    def __init__(self, backbone, neck=None, head=None):
        super().__init__()

        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)
        else:
            self.neck = None

        if head is not None:
            self.head = MODELS.build(head)
        else:
            self.head = None

    def forward(self, x):
        feats = self.backbone(x)

        if self.neck is not None:
            feats = self.neck(feats)

        if self.head is not None:
            logits = self.head(feats)
            return logits

        return feats
