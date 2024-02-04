import pycuda.autoinit
import pycuda.driver as cuda
import os
import copy
import mmcv
import torch
import argparse
import tensorrt as trt
import numpy as np
from mmcv import Config
from mmdeploy.backend.tensorrt import load_tensorrt_plugin

import sys

sys.path.append(".")
from det2trt.convert import build_engine
from det2trt.utils.tensorrt import HostDeviceMem, get_logger, create_engine_context

from det2trt.quantization import get_calibrator
from third_party.bev_mmdet3d.models.builder import build_model
from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT")
    parser.add_argument("config", help="config file path")
    parser.add_argument("onnx")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--max_workspace_size", type=int, default=1)
    parser.add_argument(
        "--calibrator", type=str, default=None, help="[legacy, entropy, minmax]"
    )
    parser.add_argument(
        "--length", type=int, default=500, help="length of data to calibrate"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    load_tensorrt_plugin()

    config = Config.fromfile(args.config)
    if hasattr(config, "plugin"):
        import importlib

        if isinstance(config.plugin, list):
            for plu in config.plugin:
                importlib.import_module(plu)
        else:
            importlib.import_module(config.plugin)

    dynamic_input = None
    if config.dynamic_input.dynamic_input:
        dynamic_input = [
            config.dynamic_input.min,
            config.dynamic_input.reg,
            config.dynamic_input.max,
        ]

    output = os.path.split(args.onnx)[1].split(".")[0]
    if args.int8:
        if args.calibrator is not None:
            output += f"_{args.calibrator}"
        output += "_int8"
    if args.fp16:
        output += "_fp16"
    output = os.path.join(config.TENSORRT_PATH, output)

    dataset = build_dataset(cfg=config.data.quant)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=6, shuffle=False, dist=False
    )
    pth_model = build_model(config.model, test_cfg=config.get("test_cfg"))

    calibrator = None
    if args.calibrator is not None:
        config.input_shapes["ranks_bev"] = [188027]
        config.input_shapes["ranks_depth"] = [188027]
        config.input_shapes["ranks_feat"] = [188027]
        config.input_shapes["interval_starts"] = [11416]
        config.input_shapes["interval_lengths"] = [11416]

        class Calibrator(get_calibrator(args.calibrator)):
            def __init__(self, *args, **kwargs):
                super(Calibrator, self).__init__(*args, **kwargs)
                (
                    self.ranks_bev,
                    self.ranks_depth,
                    self.ranks_feat,
                    self.interval_starts,
                    self.interval_lengths,
                ) = (None, None, None, None, None)

            def decode_data(self, data):
                img_inputs = data["img_inputs"][0]

                (
                    image,
                    sensor2egos,
                    ego2globals,
                    intrins,
                    post_rots,
                    post_trans,
                    bda,
                ) = img_inputs

                B, N, C, H, W = image.shape
                sensor2egos = sensor2egos.view(B, N, 4, 4)
                ego2globals = ego2globals.view(B, N, 4, 4)

                # calculate the transformation from sweep sensor to key ego
                keyego2global = ego2globals[:, 0, ...].unsqueeze(1)
                global2keyego = torch.inverse(keyego2global)
                sensor2keyegos = global2keyego @ ego2globals @ sensor2egos
                sensor2keyegos = sensor2keyegos

                if self.ranks_bev is None:
                    (
                        ranks_bev,
                        ranks_depth,
                        ranks_feat,
                        interval_starts,
                        interval_lengths,
                    ) = pth_model.get_bev_pool_input(
                        sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda
                    )
                    (
                        self.ranks_bev,
                        self.ranks_depth,
                        self.ranks_feat,
                        self.interval_starts,
                        self.interval_lengths,
                    ) = (
                        ranks_bev.float().numpy(),
                        ranks_depth.float().numpy(),
                        ranks_feat.float().numpy(),
                        interval_starts.float().numpy(),
                        interval_lengths.float().numpy(),
                    )

                for name in self.names:
                    if name == "image":
                        image = image.numpy().reshape(-1).astype(np.float32)
                        assert (
                            self.host_device_mem_dic[name].host.nbytes == image.nbytes
                        )
                        self.host_device_mem_dic[name].host = image
                    elif name == "ranks_bev":
                        ranks_bev = self.ranks_bev.reshape(-1).astype(np.float32)
                        assert (
                            self.host_device_mem_dic[name].host.nbytes
                            == ranks_bev.nbytes
                        )
                        self.host_device_mem_dic[name].host = ranks_bev
                    elif name == "ranks_depth":
                        ranks_depth = self.ranks_depth.reshape(-1).astype(np.float32)
                        assert (
                            self.host_device_mem_dic[name].host.nbytes
                            == ranks_depth.nbytes
                        )
                        self.host_device_mem_dic[name].host = ranks_depth
                    elif name == "ranks_feat":
                        ranks_feat = self.ranks_feat.reshape(-1).astype(np.float32)
                        assert (
                            self.host_device_mem_dic[name].host.nbytes
                            == ranks_feat.nbytes
                        )
                        self.host_device_mem_dic[name].host = ranks_feat
                    elif name == "interval_starts":
                        interval_starts = self.interval_starts.reshape(-1).astype(
                            np.float32
                        )
                        assert (
                            self.host_device_mem_dic[name].host.nbytes
                            == interval_starts.nbytes
                        )
                        self.host_device_mem_dic[name].host = interval_starts
                    elif name == "interval_lengths":
                        interval_lengths = self.interval_lengths.reshape(-1).astype(
                            np.float32
                        )
                        assert (
                            self.host_device_mem_dic[name].host.nbytes
                            == interval_lengths.nbytes
                        )
                        self.host_device_mem_dic[name].host = interval_lengths
                    else:
                        raise RuntimeError(f"Cannot find input name {name}.")

        calibrator = Calibrator(config, loader, args.length)

    build_engine(
        args.onnx,
        output + ".trt",
        dynamic_input=dynamic_input,
        int8=args.int8,
        fp16=args.fp16,
        max_workspace_size=args.max_workspace_size,
        calibrator=calibrator,
    )


if __name__ == "__main__":
    main()
