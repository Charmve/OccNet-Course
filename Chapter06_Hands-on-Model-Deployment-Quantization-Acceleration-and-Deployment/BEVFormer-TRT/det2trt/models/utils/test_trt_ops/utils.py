import pycuda.driver as cuda
import os

import torch
import torch.nn as nn
import numpy as np
import tensorrt as trt
import tempfile
from torch.onnx import OperatorExportTypes
import time


TRT_LOGGER = trt.Logger(trt.Logger.INTERNAL_ERROR)

EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


class Calibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, input_shapes):
        super(Calibrator, self).__init__()
        self.inputs = {}
        for name in input_shapes.keys():
            shape = input_shapes[name]
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, np.float32)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.inputs[name] = HostDeviceMem(name, host_mem, device_mem)
        self.iter_num = 1

    def set_inputs(self, inputs):
        for name in self.inputs.keys():
            np_input = inputs[name]
            self.inputs[name].host = np_input.reshape(-1)
            cuda.memcpy_htod(
                self.inputs[name].device, self.inputs[name].host,
            )

    def get_batch(self, names):
        if self.iter_num == 0:
            return None
        self.iter_num -= 1
        return [int(self.inputs[name].device) for name in names]

    def get_batch_size(self):
        return 1

    def read_calibration_cache(self):
        pass

    def write_calibration_cache(self, cache):
        pass


def createModel(
    module,
    input_shapes,
    output_shapes,
    device="cuda",
    fp16=False,
    int8=False,
    model_class=None,
    model_kwargs=None,
    **kwargs
):
    assert device in ["cuda", "cpu"]
    assert isinstance(input_shapes, dict) and isinstance(output_shapes, dict)

    if model_class is None:

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.module = module
                self.kwargs = kwargs
                self.expand = False
                if int8:
                    assert len(output_shapes["output"]) in [3, 4]
                    if len(output_shapes["output"]) == 3:
                        channel = output_shapes["output"][0]
                        self.expand = True
                    else:
                        channel = output_shapes["output"][1]
                    self.conv = nn.Conv2d(
                        channel, channel, 1, groups=channel, bias=False
                    )
                    self.conv.weight = nn.Parameter(torch.ones_like(self.conv.weight))

            def forward(self, *inputs):
                output = self.module(*inputs, **self.kwargs)
                if int8:
                    if self.expand:
                        output = output.unsqueeze(0)
                    return self.conv(output)
                return output

        model = Model().half() if fp16 else Model()
    else:
        model_kwargs = {} if model_kwargs is None else model_kwargs
        model = (
            model_class(**model_kwargs).half() if fp16 else model_class(**model_kwargs)
        )
    model.eval()
    model.input_shapes = input_shapes
    model.output_shapes = output_shapes
    return model.to(torch.device(device))


def build_engine(
    onnx_file,
    int8=False,
    fp16=False,
    int8_fp16=False,
    max_workspace_size=1,
    calibrator=None,
):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        EXPLICIT_BATCH
    ) as network, builder.create_builder_config() as config, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser, trt.Runtime(
        TRT_LOGGER
    ) as runtime:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        # max_workspace_size GB
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, (1 << 32) * max_workspace_size
        )
        # Parse model file
        if not parser.parse(onnx_file.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator
            if int8_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        return engine


def pth2trt(
    module,
    inputs,
    opset_version=13,
    fp16=False,
    int8=False,
    int8_fp16=False,
    calibrator=None,
):
    fd, path = tempfile.mkstemp()
    args = tuple([inputs[key] for key in inputs.keys()])
    try:
        torch.onnx.export(
            module.half() if fp16 or int8_fp16 else module.float(),
            args,
            path,
            input_names=list(inputs.keys()),
            output_names=list(module.output_shapes.keys()),
            opset_version=opset_version,
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
        )
        with open(path, "rb") as f:
            engine = build_engine(
                f, fp16=fp16, int8=int8, int8_fp16=int8_fp16, calibrator=calibrator
            )
    finally:
        os.remove(path)
    return engine


class HostDeviceMem(object):
    def __init__(self, name, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.name = name
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return (
            "Name:\n"
            + str(self.name)
            + "\nHost:\n"
            + str(self.host)
            + "\nDevice:\n"
            + str(self.device)
            + "\n"
        )

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine, input_shapes, output_shapes):
    inputs = []
    outputs = []
    bindings = []
    for binding_id, binding in enumerate(engine):
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            dims = input_shapes[binding]
        elif engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
            dims = output_shapes[binding]
        else:
            raise RuntimeError

        size = trt.volume(dims)
        # The maximum batch size which can be used for inference.
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(binding, host_mem, device_mem))
        elif engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
            outputs.append(HostDeviceMem(binding, host_mem, device_mem))
        else:
            raise RuntimeError
    return inputs, outputs, bindings


def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    t1 = time.time()
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    t2 = time.time()
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

    stream.synchronize()

    return outputs, t2 - t1


def trt_forward(engine, inputs_np, input_shapes, output_shapes, stream):
    context = engine.create_execution_context()
    inputs, outputs, bindings = allocate_buffers(engine, input_shapes, output_shapes)
    assert inputs_np.keys() == set([inp.name for inp in inputs])
    for inp in inputs:
        inp.host = inputs_np[inp.name].reshape(-1)
    outputs, t = do_inference(context, bindings, inputs, outputs, stream)
    outputs_dict = {}
    for out in outputs:
        outputs_dict[out.name] = out.host.reshape(*output_shapes[out.name])
    return outputs_dict, t
