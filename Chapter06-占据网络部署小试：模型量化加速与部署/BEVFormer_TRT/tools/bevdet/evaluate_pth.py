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

    ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = (
        None,
        None,
        None,
        None,
        None,
    )
    ts = []
    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in loader:
        img_inputs = data["img_inputs"][0]
        img_metas = data["img_metas"][0].data[0]

        image, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = [
            item.cuda() for item in img_inputs
        ]

        B, N, C, H, W = image.shape
        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global)
        sensor2keyegos = global2keyego @ ego2globals @ sensor2egos
        sensor2keyegos = sensor2keyegos

        if ranks_bev is None:
            (
                ranks_bev,
                ranks_depth,
                ranks_feat,
                interval_starts,
                interval_lengths,
            ) = model.module.get_bev_pool_input(
                sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda
            )
            ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = (
                ranks_bev.float(),
                ranks_depth.float(),
                ranks_feat.float(),
                interval_starts.float(),
                interval_lengths.float(),
            )

        with torch.no_grad():
            torch.cuda.synchronize()
            t1 = time.time()
            out = model(
                image,
                ranks_bev,
                ranks_depth,
                ranks_feat,
                interval_starts,
                interval_lengths,
            )
            torch.cuda.synchronize()
            t2 = time.time()

        results.extend(model.module.post_process(*out, img_metas))

        ts.append(t2 - t1)

        for _ in range(len(image)):
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
