import nuscenes.eval.common.loaders
import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda
import argparse
import torch
import mmcv
import copy
import numpy as np
from mmcv import Config
from mmdeploy.backend.tensorrt import load_tensorrt_plugin

import sys

sys.path.append(".")
from det2trt.utils.tensorrt import (
    get_logger,
    create_engine_context,
    allocate_buffers,
    do_inference,
)
from third_party.bev_mmdet3d.models.builder import build_model
from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("trt_model", help="checkpoint file")
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
        dataset, samples_per_gpu=1, workers_per_gpu=6, shuffle=False, dist=False
    )

    pth_model = build_model(config.model, test_cfg=config.get("test_cfg"))

    ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = (
        None,
        None,
        None,
        None,
        None,
    )
    ts = []
    bbox_results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in loader:
        img_inputs = data["img_inputs"][0]
        img_metas = data["img_metas"][0].data[0]

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

        if ranks_bev is None:
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

        image = image.numpy()
        batch_size, cameras, _, img_h, img_w = image.shape

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

        for inp in inputs:
            if inp.name == "image":
                inp.host = image.reshape(-1).astype(np.float32)
            elif inp.name == "ranks_bev":
                inp.host = ranks_bev.reshape(-1).astype(np.float32)
            elif inp.name == "ranks_depth":
                inp.host = ranks_depth.reshape(-1).astype(np.float32)
            elif inp.name == "ranks_feat":
                inp.host = ranks_feat.reshape(-1).astype(np.float32)
            elif inp.name == "interval_starts":
                inp.host = interval_starts.reshape(-1).astype(np.float32)
            elif inp.name == "interval_lengths":
                inp.host = interval_lengths.reshape(-1).astype(np.float32)
            else:
                raise RuntimeError(f"Cannot find input name {inp.name}.")

        trt_outputs, t = do_inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )

        trt_outputs = {
            out.name: out.host.reshape(*output_shapes_[out.name]) for out in trt_outputs
        }

        trt_outputs = {k: torch.from_numpy(v) for k, v in trt_outputs.items()}

        bbox_results.extend(pth_model.post_process(**trt_outputs, img_metas=img_metas))
        ts.append(t)

        for _ in range(len(image)):
            prog_bar.update()

    metric = dataset.evaluate(bbox_results)

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
