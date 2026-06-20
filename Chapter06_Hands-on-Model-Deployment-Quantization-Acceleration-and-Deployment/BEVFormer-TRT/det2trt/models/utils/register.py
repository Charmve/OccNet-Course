from mmcv.cnn.bricks.registry import CONV_LAYERS
from mmcv.utils import Registry
from pytorch_quantization import nn as quant_nn
from torch import nn
import os
import ctypes


class FuncRegistry:
    def __init__(self, name, build_func=None, parent=None, scope=None):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = (
            self.__class__.__name__ + f"(name={self._name}, "
            f"items={self._module_dict})"
        )
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key)

    def _register_module(self, module, module_name=None, force=False):
        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f"{name} is already registered " f"in {self.name}")
            self._module_dict[name] = module

    def register_module(self, name=None, force=False, module=None):
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(
                "name must be either of None, an instance of str or a sequence"
                f"  of str, but got {type(name)}"
            )

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(module=cls, module_name=name, force=force)
            return cls

        return _register


OS_PATH = "TensorRT/lib/libtensorrt_ops.so"
OS_PATH = os.path.realpath(OS_PATH)
ctypes.CDLL(OS_PATH)
print(f"Loaded tensorrt plugins from {OS_PATH}")

CONV_LAYERS.register_module("Conv1dQ", module=quant_nn.Conv1d)
CONV_LAYERS.register_module("Conv2dQ", module=quant_nn.Conv2d)
CONV_LAYERS.register_module("Conv3dQ", module=quant_nn.Conv3d)
CONV_LAYERS.register_module("ConvQ", module=quant_nn.Conv2d)

LINEAR_LAYERS = Registry("linear layer")
LINEAR_LAYERS.register_module("Linear", module=nn.Linear)
LINEAR_LAYERS.register_module("LinearQ", module=quant_nn.Linear)

TRT_FUNCTIONS = FuncRegistry("tensorrt functions")
