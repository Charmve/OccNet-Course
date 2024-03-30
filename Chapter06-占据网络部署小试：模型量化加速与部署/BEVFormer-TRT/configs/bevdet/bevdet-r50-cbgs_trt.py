_base_ = ["bevdet-r50-cbgs.py", "../_base_/det2trt.py"]

model = dict(
    type="BEVDetTRT",
    img_view_transformer=dict(type="LSSViewTransformerTRT"),
    pts_bbox_head=dict(type="BEVDetCenterHeadTRT"),
)

data = dict(
    samples_per_gpu=1,
    quant=dict(
        type={{_base_.data.val.type}},
        pipeline={{_base_.data.val.pipeline}},
        ann_file={{_base_.data.train.dataset.ann_file}},
        classes={{_base_.data.val.classes}},
        modality={{_base_.data.val.modality}},
        img_info_prototype={{_base_.data.val.img_info_prototype}},
        data_root={{_base_.data.val.data_root}},
        test_mode=True,
        box_type_3d="LiDAR",
    ),
)

# batch_size, num_classes, img_h, img_w
default_shapes = dict(
    batch_size=1,
    img_hw={{_base_.data_config.input_size}},
    bev_y={{_base_.grid_config.y}},
    bev_x={{_base_.grid_config.x}},
    cameras={{_base_.data_config.Ncams}},
)

input_shapes = dict(
    image=["batch_size", "cameras", 3, "img_hw[0]", "img_hw[1]"],
    ranks_bev=[179520],
    ranks_depth=[179520],
    ranks_feat=[179520],
    interval_starts=[11398],
    interval_lengths=[11398],
)

output_shapes = dict(
    reg=[
        "batch_size",
        2,
        "int((bev_y[1]-bev_y[0])/bev_y[2])",
        "int((bev_x[1]-bev_x[0])/bev_x[2])",
    ],
    height=[
        "batch_size",
        1,
        "int((bev_y[1]-bev_y[0])/bev_y[2])",
        "int((bev_x[1]-bev_x[0])/bev_x[2])",
    ],
    dim=[
        "batch_size",
        3,
        "int((bev_y[1]-bev_y[0])/bev_y[2])",
        "int((bev_x[1]-bev_x[0])/bev_x[2])",
    ],
    rot=[
        "batch_size",
        2,
        "int((bev_y[1]-bev_y[0])/bev_y[2])",
        "int((bev_x[1]-bev_x[0])/bev_x[2])",
    ],
    vel=[
        "batch_size",
        2,
        "int((bev_y[1]-bev_y[0])/bev_y[2])",
        "int((bev_x[1]-bev_x[0])/bev_x[2])",
    ],
    heatmap=[
        "batch_size",
        10,
        "int((bev_y[1]-bev_y[0])/bev_y[2])",
        "int((bev_x[1]-bev_x[0])/bev_x[2])",
    ],
)
