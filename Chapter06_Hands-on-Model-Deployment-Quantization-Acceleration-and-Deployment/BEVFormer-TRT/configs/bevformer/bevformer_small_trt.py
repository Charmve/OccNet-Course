_base_ = ["bevformer_small.py", "../_base_/det2trt.py"]

model = dict(
    type="BEVFormerTRT",
    pts_bbox_head=dict(
        type="BEVFormerHeadTRT",
        transformer=dict(
            type="PerceptionTransformerTRT",
            encoder=dict(
                type="BEVFormerEncoderTRT",
                transformerlayers=dict(
                    type="BEVFormerLayerTRT",
                    attn_cfgs=[
                        dict(
                            type="TemporalSelfAttentionTRT",
                            embed_dims={{_base_._dim_}},
                            num_levels=1,
                        ),
                        dict(
                            type="SpatialCrossAttentionTRT",
                            pc_range={{_base_.point_cloud_range}},
                            deformable_attention=dict(
                                type="MSDeformableAttention3DTRT",
                                embed_dims={{_base_._dim_}},
                                num_points=8,
                                num_levels={{_base_._num_levels_}},
                            ),
                            embed_dims={{_base_._dim_}},
                        ),
                    ],
                    ffn_cfgs=dict(type="FFNTRT"),
                ),
            ),
            decoder=dict(
                type="DetectionTransformerDecoder",
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
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
                        ),
                    ],
                    ffn_cfgs=dict(type="FFNTRT"),
                ),
            ),
        ),
    ),
)

data = dict(
    samples_per_gpu=1,
    quant=dict(
        type={{_base_.dataset_type}},
        data_root={{_base_.data_root}},
        ann_file={{_base_.data.train.ann_file}},
        pipeline={{_base_.test_pipeline}},
        bev_size={{_base_.data.train.bev_size}},
        classes={{_base_.class_names}},
        modality={{_base_.input_modality}},
        test_mode=True,
        box_type_3d="LiDAR",
    ),
)

# batch_size, num_classes, img_h, img_w
default_shapes = dict(
    batch_size=1,
    img_h=736,
    img_w=1280,
    bev_h={{_base_.bev_h_}},
    bev_w={{_base_.bev_w_}},
    dim={{_base_._dim_}},
    num_query={{_base_.model.pts_bbox_head.num_query}},
    num_classes={{_base_.model.pts_bbox_head.num_classes}},
    code_size={{_base_.model.pts_bbox_head.code_size}},
    cameras=6,
)

input_shapes = dict(
    image=["batch_size", "cameras", 3, "img_h", "img_w"],
    prev_bev=["bev_h*bev_w", "batch_size", "dim"],
    use_prev_bev=[1],
    can_bus=[18],
    lidar2img=["batch_size", "cameras", 4, 4],
)

output_shapes = dict(
    bev_embed=["bev_h*bev_w", "batch_size", "dim"],
    outputs_classes=["cameras", "batch_size", "num_query", "num_classes"],
    outputs_coords=["cameras", "batch_size", "num_query", "code_size"],
)
