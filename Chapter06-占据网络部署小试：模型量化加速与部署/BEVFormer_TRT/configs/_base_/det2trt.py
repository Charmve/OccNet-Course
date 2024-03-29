ONNX_PATH = "checkpoints/onnx"
PYTORCH_PATH = "checkpoints/pytorch"
TENSORRT_PATH = "checkpoints/tensorrt"

onnx_shapes = dict()
input_shapes = dict()
output_shapes = dict()
dynamic_axes = None
dynamic_input = dict(dynamic_input=False, min=None, reg=None, max=None)
