import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda
import argparse
import torch
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.apis import init_detector
from mmdeploy.backend.tensorrt import load_tensorrt_plugin


import sys

sys.path.append(".")
from det2trt.utils.tensorrt import (
    get_logger,
    create_engine_context,
    allocate_buffers,
    do_inference,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("trt_model", help="checkpoint file")
    parser.add_argument("--samples_per_gpu", default=32, type=int)
    parser.add_argument("--workers_per_gpu", default=8, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    load_tensorrt_plugin()

    trt_model = args.trt_model
    config_file = args.config
    TRT_LOGGER = get_logger(trt.Logger.INTERNAL_ERROR)

    engine, context = create_engine_context(trt_model, TRT_LOGGER)

    stream = cuda.Stream()

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

    output_shapes = config.output_shapes
    input_shapes = config.input_shapes
    default_shapes = config.default_shapes

    for key in default_shapes:
        if key in locals():
            raise RuntimeError(f"Variable {key} has been defined.")
        locals()[key] = default_shapes[key]

    dataset = build_dataset(cfg=config.data.val)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        shuffle=False,
        dist=False,
    )

    config.model.bbox_head.update(test_cfg=config.model.test_cfg)
    pth_model = init_detector(config=config_file)

    ts = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    bbox_results = []
    for data in loader:
        img = data["img"][0].data[0].numpy()
        batch_input_shape = tuple(img[0].shape[-2:])
        batch_size, _, img_h, img_w = img.shape

        output_shapes_ = {}
        for key in output_shapes.keys():
            shape = output_shapes[key][:]
            for shape_i in range(len(shape)):
                if isinstance(shape[shape_i], str):
                    shape[shape_i] = eval(shape[shape_i])
            output_shapes_[key] = shape

        input_shapes_ = {}
        for key in input_shapes.keys():
            shape = input_shapes[key][:]
            for shape_i in range(len(shape)):
                if isinstance(shape[shape_i], str):
                    shape[shape_i] = eval(shape[shape_i])
            input_shapes_[key] = shape

        inputs, outputs, bindings = allocate_buffers(
            engine, context, input_shapes=input_shapes_, output_shapes=output_shapes_
        )

        inputs[0].host = img.reshape(-1)

        trt_outputs, t = do_inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )

        trt_outputs = {
            out.name: torch.from_numpy(out.host.reshape(*output_shapes_[out.name]))
            for out in trt_outputs
        }

        img_metas = data["img_metas"][0].data[0]
        for img_meta in img_metas:
            img_meta["batch_input_shape"] = batch_input_shape

        bbox_results.extend(pth_model.post_process(**trt_outputs, img_metas=img_metas))
        ts.append(t)

        for _ in range(len(img)):
            prog_bar.update()

    metric = dataset.evaluate(bbox_results)

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
