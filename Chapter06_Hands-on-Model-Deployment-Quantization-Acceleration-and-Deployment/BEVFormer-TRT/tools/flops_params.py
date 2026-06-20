import argparse
from mmcv import Config
from mmdet.models import build_detector
import torch
import torch.nn as nn
from thop.vision.basic_hooks import count_parameters
from thop.profile import register_hooks
from thop.utils import prRed


def parse_args():
    parser = argparse.ArgumentParser(description="Get FLOPs and Params")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    return args


def profile(
    model: nn.Module,
    inputs_kw,
    custom_ops=None,
    verbose=True,
    ret_layer_info=False,
    report_missing=False,
):
    handler_collection = {}
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}
    if report_missing:
        # overwrite `verbose` option when enable report_missing
        verbose = True

    def add_hooks(m: nn.Module):
        m.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))
        m.register_buffer("total_params", torch.zeros(1, dtype=torch.float64))

        # for p in m.parameters():
        #     m.total_params += torch.DoubleTensor([p.numel()])

        m_type = type(m)

        fn = None
        if m_type in custom_ops:
            # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Customize rule %s() %s." % (fn.__qualname__, m_type))
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type))
        else:
            if m_type not in types_collection and report_missing:
                prRed(
                    "[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params."
                    % m_type
                )

        if fn is not None:
            handler_collection[m] = (
                m.register_forward_hook(fn),
                m.register_forward_hook(count_parameters),
            )
        types_collection.add(m_type)

    prev_training_status = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(**inputs_kw)

    def dfs_count(module: nn.Module, prefix="\t") -> (int, int):
        total_ops, total_params = module.total_ops.item(), 0
        ret_dict = {}
        for n, m in module.named_children():
            # if not hasattr(m, "total_ops") and not hasattr(m, "total_params"):  # and len(list(m.children())) > 0:
            #     m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            # else:
            #     m_ops, m_params = m.total_ops, m.total_params
            next_dict = {}
            if m in handler_collection and not isinstance(
                m, (nn.Sequential, nn.ModuleList)
            ):
                m_ops, m_params = m.total_ops.item(), m.total_params.item()
            else:
                m_ops, m_params, next_dict = dfs_count(m, prefix=prefix + "\t")
            ret_dict[n] = (m_ops, m_params, next_dict)
            total_ops += m_ops
            total_params += m_params
        # print(prefix, module._get_name(), (total_ops, total_params))
        return total_ops, total_params, ret_dict

    total_ops, total_params, ret_dict = dfs_count(model)

    # reset model to original status
    model.train(prev_training_status)
    for m, (op_handler, params_handler) in handler_collection.items():
        op_handler.remove()
        params_handler.remove()
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")

    if ret_layer_info:
        return total_ops, total_params, ret_dict
    return total_ops, total_params


def main():
    args = parse_args()

    config_file = args.config
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

    onnx_shapes = config.default_shapes
    input_shapes = config.input_shapes

    for key in onnx_shapes:
        if key in locals():
            raise RuntimeError(f"Variable {key} has been defined.")
        locals()[key] = onnx_shapes[key]

    inputs = {}
    for key in input_shapes.keys():
        for i in range(len(input_shapes[key])):
            if isinstance(input_shapes[key][i], str):
                input_shapes[key][i] = eval(input_shapes[key][i])
        inputs[key] = torch.randn(*input_shapes[key])
        if args.cuda:
            inputs[key] = inputs[key].cuda()

    model = build_detector(config.model, test_cfg=config.get("test_cfg", None))
    model.forward = model.forward_trt
    model.eval()
    if args.cuda:
        model.cuda()

    flops, params = profile(model, inputs_kw=inputs)

    print(f"Config file is {args.config}")
    print(f"FLOPs = {flops / 1024 ** 3}G")
    print(f"Params = {params / 1024 ** 2}M")


if __name__ == "__main__":
    main()
