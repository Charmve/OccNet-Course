import os
import argparse
import torch
from mmcv import Config
from mmdet.apis import init_detector
from mmcv.parallel import MMDataParallel
from mmdet.datasets import build_dataloader, build_dataset

import sys

sys.path.append(".")
from det2trt.quantization import calibrator_qdq


def parse_args():
    parser = argparse.ArgumentParser(description="Post-training Quantization")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--calibrator", default="max", help="[max, histogram]")
    parser.add_argument("--pcq", action="store_true", help="per channel quantization")
    parser.add_argument("--d_len", default=500, type=int)
    parser.add_argument("--samples_per_gpu", default=32, type=int)
    parser.add_argument("--workers_per_gpu", default=8, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = Config.fromfile(args.config)
    if hasattr(config, "plugin"):
        import importlib

        if isinstance(config.plugin, list):
            for plu in config.plugin:
                importlib.import_module(plu)
        else:
            importlib.import_module(config.plugin)

    model = init_detector(config=config, checkpoint=args.checkpoint)
    model.eval()
    model = MMDataParallel(model.cuda())

    output = os.path.split(args.checkpoint)[1].split(".")[0] + f"_ptq_{args.calibrator}"
    if args.pcq:
        output += "_pcq"

    dataset = build_dataset(cfg=config.data.quant)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        shuffle=True,
        seed=0,
        dist=False,
    )

    model = calibrator_qdq(
        model=model,
        calibrator=args.calibrator,
        per_channel_quantization=args.pcq,
        data_length=args.d_len,
        samples_per_gpu=args.samples_per_gpu,
        loader=loader,
        return_loss=False,
        rescale=True,
    )

    save_dir = os.path.join(config.PYTORCH_PATH, output + ".pth")
    torch.save(model.state_dict(), save_dir)
    print(f"Post-training quantization model has been saved in {save_dir}")


if __name__ == "__main__":
    main()
