import pycuda.autoinit
import os
import argparse
import numpy as np
from mmcv import Config
from mmdeploy.backend.tensorrt import load_tensorrt_plugin

import sys

sys.path.append(".")
from det2trt.convert import build_engine

from det2trt.quantization import get_calibrator
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
    parser.add_argument("--samples_per_gpu", default=32, type=int)
    parser.add_argument("--workers_per_gpu", default=8, type=int)
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
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        shuffle=True,
        dist=False,
    )

    calibrator = None
    if args.calibrator is not None:

        class Calibrator(get_calibrator(args.calibrator)):
            def __init__(self, *args, **kwargs):
                super(Calibrator, self).__init__(*args, **kwargs)

            def decode_data(self, data):
                img = data["img"][0].data[0].numpy()
                for name in self.names:
                    if name == "image":
                        img = img.reshape(-1).astype(np.float32)
                        assert self.host_device_mem_dic[name].host.nbytes == img.nbytes
                        self.host_device_mem_dic[name].host = img
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
