import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
from ..utils.tensorrt import HostDeviceMem

CALIBRATORS = {
    "minmax": trt.IInt8MinMaxCalibrator,
    "entropy": trt.IInt8EntropyCalibrator2,
    "legacy": trt.IInt8LegacyCalibrator,
}


def get_calibrator(calibrator):
    assert (
        calibrator in CALIBRATORS
    ), "calibrator_qdq should be in [minmax, entropy, legacy]"

    class Calibrator(CALIBRATORS[calibrator]):
        def __init__(self, config, dataloader, length):
            super(Calibrator, self).__init__()
            self.config = config
            self.iter = dataloader.__iter__()
            self.length = length
            self.current_batch = 0
            self.batch_size = dataloader.batch_size
            self.num_batch = (length + self.batch_size - 1) // self.batch_size
            self.num_batch = min(
                len(dataloader.dataset) // self.batch_size, self.num_batch
            )
            self.input_shapes = {}
            self.get_input_shapes()
            self.names = set(self.input_shapes.keys())
            self.host_device_mem_dic = {}
            for name in self.names:
                shape = self.input_shapes[name]
                size = trt.volume(shape)
                host_mem = cuda.pagelocked_empty(size, np.float32)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.host_device_mem_dic[name] = HostDeviceMem(
                    name, host_mem, device_mem
                )

        def get_input_shapes(self):
            for key in self.config.default_shapes:
                if key in locals():
                    raise RuntimeError(f"Variable {key} has been defined.")
                locals()[key] = self.config.default_shapes[key]
            batch_size = self.batch_size

            for key in self.config.input_shapes.keys():
                shape = self.config.input_shapes[key][:]
                for shape_i in range(len(shape)):
                    if isinstance(shape[shape_i], str):
                        shape[shape_i] = eval(shape[shape_i])
                self.input_shapes[key] = shape

        def get_batch(self, names):
            try:
                if self.current_batch >= self.num_batch:
                    return None
                assert set(names) == self.names
                # Assume self.batches is a generator that provides batch data.
                data = self.iter.next()
                self.decode_data(data)
                # Assume that self.device_input is a device buffer allocated by the constructor.
                [
                    cuda.memcpy_htod(
                        self.host_device_mem_dic[name].device,
                        self.host_device_mem_dic[name].host,
                    )
                    for name in names
                ]
                self.current_batch += 1
                return [int(self.host_device_mem_dic[name].device) for name in names]
            except StopIteration:
                # When we're out of batches, we return either [] or None.
                # This signals to TensorRT that there is no calibration data remaining.
                return None

        def decode_data(self, data):
            raise NotImplementedError

        def get_batch_size(self):
            return self.batch_size

        def read_calibration_cache(self):
            pass

        def write_calibration_cache(self, cache):
            pass

    return Calibrator
