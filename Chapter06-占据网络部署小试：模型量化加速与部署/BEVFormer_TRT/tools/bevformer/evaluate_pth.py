import argparse

import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
import torch
from mmcv.parallel import MMDataParallel
import mmcv
import time
import copy

import sys

sys.path.append(".")
from third_party.bev_mmdet3d.models.builder import build_model
from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet3D test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config_file = args.config
    checkpoint_file = args.checkpoint
    config = Config.fromfile(config_file)
    if hasattr(config, "plugin"):
        import importlib
        import sys

        sys.path.append(".")
        if isinstance(config.plugin, list):
            for plu in config.plugin:
                importlib.import_module(plu)
        else:
            importlib.import_module(config.plugin)

    model = build_model(config.model, test_cfg=config.get("test_cfg", None))
    checkpoint = load_checkpoint(model, checkpoint_file, map_location="cpu")

    dataset = build_dataset(cfg=config.data.val)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=6, shuffle=False, dist=False
    )

    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
        # palette for visualization in segmentation tasks
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    elif hasattr(dataset, "PALETTE"):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    model.forward = model.forward_trt
    model.eval()
    model = MMDataParallel(model.cuda())

    ts = []
    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    prev_bev = torch.randn(config.bev_h_ * config.bev_w_, 1, config._dim_).cuda()
    prev_frame_info = {
        "scene_token": None,
        "prev_pos": 0,
        "prev_angle": 0,
    }
    for data in loader:
        img = data["img"][0].data[0].cuda()
        img_metas = data["img_metas"][0].data[0]

        use_prev_bev = torch.tensor([1.0]).cuda()
        if img_metas[0]["scene_token"] != prev_frame_info["scene_token"]:
            use_prev_bev = torch.tensor([0.0]).cuda()
        prev_frame_info["scene_token"] = img_metas[0]["scene_token"]
        tmp_pos = copy.deepcopy(img_metas[0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0]["can_bus"][-1])
        if use_prev_bev[0] == 1:
            img_metas[0]["can_bus"][:3] -= prev_frame_info["prev_pos"]
            img_metas[0]["can_bus"][-1] -= prev_frame_info["prev_angle"]
        else:
            img_metas[0]["can_bus"][-1] = 0
            img_metas[0]["can_bus"][:3] = 0
        can_bus = torch.from_numpy(img_metas[0]["can_bus"]).cuda().float()
        lidar2img = (
            torch.from_numpy(np.stack(img_metas[0]["lidar2img"], axis=0))
            .unsqueeze(0)
            .cuda()
            .float()
        )

        with torch.no_grad():
            torch.cuda.synchronize()
            t1 = time.time()
            bev_embed, outputs_classes, outputs_coords = model(
                img, prev_bev, use_prev_bev, can_bus, lidar2img
            )
            torch.cuda.synchronize()
            t2 = time.time()

        results.extend(
            model.module.post_process(outputs_classes, outputs_coords, img_metas)
        )

        prev_bev = bev_embed
        prev_frame_info["prev_pos"] = tmp_pos
        prev_frame_info["prev_angle"] = tmp_angle

        ts.append(t2 - t1)

        for _ in range(len(img)):
            prog_bar.update()

    metric = dataset.evaluate(results)

    # summary
    print("*" * 50 + " SUMMARY " + "*" * 50)
    for key in metric.keys():
        if key == "pts_bbox_NuScenes/NDS":
            print(f"NDS: {round(metric[key], 3)}")
        elif key == "pts_bbox_NuScenes/mAP":
            print(f"mAP: {round(metric[key], 3)}")

    latency = round(sum(ts[1:-1]) / len(ts[1:-1]) * 1000, 2)
    print(f"Latency: {latency}ms")
    print(f"FPS: {1000 / latency}")


if __name__ == "__main__":
    main()
