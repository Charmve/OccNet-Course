from pytorch_quantization import nn as quant_nn
import os
import torch
import argparse
from mmcv import Config

import sys

sys.path.append(".")
from det2trt.convert import pytorch2onnx
from det2trt.quantization import get_calibrator

from third_party.bev_mmdet3d.models.builder import build_model
from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch to ONNX")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--opset_version", type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--flag", default="", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config_file = args.config
    checkpoint_file = args.checkpoint

    config = Config.fromfile(config_file)
    if hasattr(config, "plugin"):
        import importlib

        if isinstance(config.plugin, list):
            for plu in config.plugin:
                importlib.import_module(plu)
        else:
            importlib.import_module(config.plugin)

    output = os.path.split(args.checkpoint)[1].split(".")[0]

    if args.int8:
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
    if args.flag:
        output += f"_{args.flag}"
    output_file = os.path.join(config.ONNX_PATH, output + ".onnx")

    dataset = build_dataset(cfg=config.data.val)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=6, shuffle=False, dist=False
    )

    pth_model = build_model(config.model, test_cfg=config.get("test_cfg"))

    _, data = next(enumerate(loader))
    img_inputs = data["img_inputs"][0]

    image, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = img_inputs

    B, N, C, H, W = image.shape
    sensor2egos = sensor2egos.view(B, N, 4, 4)
    ego2globals = ego2globals.view(B, N, 4, 4)

    # calculate the transformation from sweep sensor to key ego
    keyego2global = ego2globals[:, 0, ...].unsqueeze(1)
    global2keyego = torch.inverse(keyego2global)
    sensor2keyegos = global2keyego @ ego2globals @ sensor2egos
    sensor2keyegos = sensor2keyegos

    (
        ranks_bev,
        ranks_depth,
        ranks_feat,
        interval_starts,
        interval_lengths,
    ) = pth_model.get_bev_pool_input(
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda
    )
    ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = (
        ranks_bev.float().numpy(),
        ranks_depth.float().numpy(),
        ranks_feat.float().numpy(),
        interval_starts.float().numpy(),
        interval_lengths.float().numpy(),
    )
    inputs_data = dict(
        ranks_bev=ranks_bev,
        ranks_depth=ranks_depth,
        ranks_feat=ranks_feat,
        interval_starts=interval_starts,
        interval_lengths=interval_lengths,
    )

    pytorch2onnx(
        config,
        checkpoint=checkpoint_file,
        output_file=output_file,
        verbose=False,
        opset_version=args.opset_version,
        cuda=args.cuda,
        inputs_data=inputs_data,
    )


if __name__ == "__main__":
    main()
