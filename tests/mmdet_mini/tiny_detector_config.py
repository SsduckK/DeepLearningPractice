custom_imports = dict(
    imports=[
        "custom_backbone",
        "custom_neck",
        "custom_head",
        "custom_detector",
    ],
    allow_failed_imports=False,
)

model = dict(
    type="TinyDetector",
    backbone=dict(
        type="TinyCustomBackbone",
        in_channels=3,
        out_channels=16,
    ),
    neck=dict(
        type="TinyFPNNeck",
        in_channels=[16],
        out_channels=32,
    ),
    head=dict(
        type="TinyClsHead",
        in_channels=32,
        num_classes=3,
    ),
)
