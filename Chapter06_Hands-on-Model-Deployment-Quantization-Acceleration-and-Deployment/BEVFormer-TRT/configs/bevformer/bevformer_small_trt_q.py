_base_ = ["bevformer_small_trt.py"]

model = dict(
    img_backbone=dict(
        type="ResNetQ",
        conv_cfg=dict(type="Conv2dQ"),
        dcn=dict(type="DCNv2Q", deform_groups=1, fallback_on_stride=False),
    ),
    img_neck=dict(conv_cfg=dict(type="Conv2dQ")),
    pts_bbox_head=dict(
        linear_cfg=dict(type="LinearQ"),
        transformer=dict(
            linear_cfg=dict(type="LinearQ"),
            encoder=dict(
                transformerlayers=dict(
                    ffn_cfgs=dict(linear_cfg=dict(type="LinearQ")),
                    attn_cfgs=[
                        dict(
                            type="TemporalSelfAttentionTRT",
                            embed_dims={{_base_._dim_}},
                            num_levels=1,
                            linear_cfg=dict(type="LinearQ"),
                        ),
                        dict(
                            type="SpatialCrossAttentionTRT",
                            pc_range={{_base_.point_cloud_range}},
                            deformable_attention=dict(
                                type="MSDeformableAttention3DTRT",
                                embed_dims={{_base_._dim_}},
                                num_points=8,
                                num_levels={{_base_._num_levels_}},
                                linear_cfg=dict(type="LinearQ"),
                            ),
                            embed_dims={{_base_._dim_}},
                            linear_cfg=dict(type="LinearQ"),
                        ),
                    ],
                )
            ),
            decoder=dict(
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims={{_base_._dim_}},
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="CustomMSDeformableAttentionTRT",
                            embed_dims={{_base_._dim_}},
                            num_levels=1,
                            linear_cfg=dict(type="LinearQ"),
                        ),
                    ],
                    ffn_cfgs=dict(linear_cfg=dict(type="LinearQ")),
                )
            ),
        ),
    ),
)


# learning policy
lr_config = dict(_delete_=True, policy="step", step=[10])

total_epochs = 1
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)

optimizer = dict(
    type="SGD",
    lr=1e-6,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)
optimizer_config = dict(grad_clip=None)

load_from = "checkpoints/pytorch/bevformer_small_epoch_24.pth"
