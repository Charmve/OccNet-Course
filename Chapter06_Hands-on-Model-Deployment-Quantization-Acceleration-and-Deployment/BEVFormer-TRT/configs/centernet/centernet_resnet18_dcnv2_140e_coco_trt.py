_base_ = [
    "centernet_resnet18_dcnv2_140e_coco.py",
    "../_base_/det2trt.py",
]

model = dict(type="CenterNetTRT")

data = dict(
    samples_per_gpu=1,
    quant=dict(
        type={{_base_.dataset_type}},
        ann_file={{_base_.data.train.dataset.ann_file}},
        img_prefix={{_base_.data.train.dataset.img_prefix}},
        pipeline={{_base_.test_pipeline}},
    ),
)

# batch_size, num_classes, img_h, img_w
default_shapes = dict(batch_size=32, img_h=672, img_w=672, num_classes=80)

input_shapes = dict(image=["batch_size", 3, "img_h", "img_w"])

output_shapes = dict(
    center_heatmap_preds=["batch_size", "num_classes", "img_h//4", "img_w//4"],
    wh_preds=["batch_size", 2, "img_h//4", "img_w//4"],
    offset_preds=["batch_size", 2, "img_h//4", "img_w//4"],
)

dynamic_axes = dict(_delete_=True, image={0: "batch_size", 2: "img_h", 3: "img_w"})

dynamic_input = dict(
    dynamic_input=True,
    min=(1, 3, 336, 336),
    reg=(
        default_shapes["batch_size"],
        3,
        default_shapes["img_h"],
        default_shapes["img_w"],
    ),
    max=(64, 3, 1344, 1344),
)
