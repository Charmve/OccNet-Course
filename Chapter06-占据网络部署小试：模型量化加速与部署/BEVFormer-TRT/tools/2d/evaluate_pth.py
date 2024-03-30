import argparse
from mmdet.apis import init_detector
from mmdet.datasets import build_dataloader, build_dataset
from mmcv import Config
import torch
from mmcv.parallel import MMDataParallel
import mmcv
import time


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--samples_per_gpu", default=32, type=int)
    parser.add_argument("--workers_per_gpu", default=1, type=int)
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

    model = init_detector(config=config_file, checkpoint=checkpoint_file)
    model.forward = model.forward_trt
    model.eval()

    model = MMDataParallel(model.cuda())

    dataset = build_dataset(cfg=config.data.val)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        shuffle=False,
        dist=False,
    )
    ts = []
    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in loader:
        img = data["img"][0].data[0].cuda()
        img_metas = data["img_metas"][0].data[0]

        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta["batch_input_shape"] = batch_input_shape

        with torch.no_grad():
            torch.cuda.synchronize()
            t1 = time.time()
            out = model(img)
            torch.cuda.synchronize()
            t2 = time.time()

        ts.append(t2 - t1)
        results.extend(model.module.post_process(*out, img_metas))

        for _ in range(len(img)):
            prog_bar.update()

    metric = dataset.evaluate(results)

    # summary
    print("*" * 50 + " SUMMARY " + "*" * 50)
    for key in metric.keys():
        if key == "bbox_mAP_copypaste":
            continue
        print(f"{key[5:]}: {metric[key]}")

    latency = round(sum(ts[1:-1]) / len(ts[1:-1]) * 1000, 2)
    print(f"Latency: {latency}ms")
    print(f"FPS: {args.samples_per_gpu * 1000 / latency}")


if __name__ == "__main__":
    main()
