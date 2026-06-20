import os
import argparse
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

import sys

sys.path.append(".")
from det2trt.quantization import calibrator_qdq
from third_party.bev_mmdet3d.models.builder import build_model
from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Post-training Quantization")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--calibrator", default="max", help="[max, histogram]")
    parser.add_argument("--pcq", action="store_true", help="per channel quantization")
    parser.add_argument("--d_len", default=500, type=int)
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

    model = build_model(config.model, test_cfg=config.get("test_cfg"))
    load_checkpoint(model, checkpoint_file, map_location="cpu")

    model.eval()
    model = MMDataParallel(model.cuda())

    output = os.path.split(args.checkpoint)[1].split(".")[0] + f"_ptq_{args.calibrator}"
    if args.pcq:
        output += "_pcq"

    dataset = build_dataset(cfg=config.data.quant)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=6, shuffle=False, dist=False
    )

    model = calibrator_qdq(
        model=model,
        calibrator=args.calibrator,
        per_channel_quantization=args.pcq,
        data_length=args.d_len,
        samples_per_gpu=1,
        loader=loader,
        return_loss=False,
        rescale=True,
    )

    save_dir = os.path.join(config.PYTORCH_PATH, output + ".pth")
    torch.save(model.state_dict(), save_dir)
    print(f"Post-training quantization model has been saved in {save_dir}")


if __name__ == "__main__":
    main()
