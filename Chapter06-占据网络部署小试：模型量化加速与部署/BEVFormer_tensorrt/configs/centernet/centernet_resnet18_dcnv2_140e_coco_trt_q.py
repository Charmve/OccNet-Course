_base_ = ["centernet_resnet18_dcnv2_140e_coco_trt.py"]

model = dict(
    backbone=dict(type="ResNetQ", conv_cfg=dict(type="Conv2dQ")),
    # neck=dict(type='CTResNetNeck', conv_cfg=dict(type='Conv2dQ')),
    neck=dict(type="CTResNetNeck", use_dcn=True),
    bbox_head=dict(type="CenterNetHeadQ"),
)


# learning policy
lr_config = dict(_delete_=True, policy="step", step=[10])

runner = dict(type="EpochBasedRunner", max_epochs=1)

optimizer = dict(
    type="SGD",
    lr=0.0001,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)
optimizer_config = dict(grad_clip=None)

load_from = "checkpoints/pytorch/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth"
