import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import time


def get_logger(level=trt.Logger.INTERNAL_ERROR):
    TRT_LOGGER = trt.Logger(level)
    return TRT_LOGGER


def create_engine_context(trt_model, trt_logger):
    with open(trt_model, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    return engine, context


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


def allocate_buffers(engine, context, input_shapes, output_shapes):
    inputs = []
    outputs = []
    bindings = []
    for binding_id, binding in enumerate(engine):
        if engine.binding_is_input(binding):
            dims = input_shapes[binding]
            context.set_binding_shape(binding_id, dims)
        else:
            dims = output_shapes[binding]

        size = trt.volume(dims)
        # The maximum batch size which can be used for inference.
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        assert dtype == np.float32, "Engine's inputs/outputs only support FP32."
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(binding, host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(binding, host_mem, device_mem))
    return inputs, outputs, bindings


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    stream.synchronize()
    t1 = time.time()
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    stream.synchronize()
    t2 = time.time()
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()

    return outputs, t2 - t1
