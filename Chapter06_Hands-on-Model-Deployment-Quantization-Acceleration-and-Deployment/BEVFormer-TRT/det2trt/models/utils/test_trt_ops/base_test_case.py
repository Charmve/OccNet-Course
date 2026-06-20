import pycuda.driver as cuda
import torch
import numpy as np
from ..register import TRT_FUNCTIONS
from .utils import createModel, pth2trt, trt_forward, Calibrator


class BaseTestCase:
    def __init__(
        self,
        model_name,
        input_shapes,
        output_shapes,
        func_name,
        params=None,
        device="cuda",
        delta=1e-5,
        model_class=None,
        model_kwargs=None,
    ):
        assert "fp16" in func_name or "fp32" in func_name or "int8" in func_name
        self.int8_fp16 = False
        self.fp16 = False
        self.int8 = False
        if "int8" in func_name and "fp16" in func_name:
            self.int8_fp16 = True
            self.int8 = True
            self.fp16 = True
        elif "fp16" in func_name:
            self.fp16 = True
        elif "int8" in func_name:
            self.int8 = True
        self.stream = cuda.Stream()
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        assert len(output_shapes) == 1
        self.device = device
        self.delta = delta
        if params is None:
            params = [{}]
        assert isinstance(params, list)
        module = TRT_FUNCTIONS.get(model_name)
        self.createInputs()
        self.models = self.createModels(
            params, module, input_shapes, output_shapes, model_class, model_kwargs
        )

    def createModels(
        self, params, module, input_shapes, output_shapes, model_class, model_kwargs
    ):
        models = []
        for param in params:
            dic = {"param": param}
            if self.fp16:
                model_fp16 = createModel(
                    module,
                    input_shapes=input_shapes,
                    output_shapes=output_shapes,
                    fp16=self.fp16,
                    int8=self.int8,
                    model_class=model_class,
                    model_kwargs=model_kwargs,
                    **param
                )
                if self.int8:
                    dic["model_pth_int8"] = model_fp16
                else:
                    dic["model_pth_fp16"] = model_fp16
            else:
                model_fp32 = createModel(
                    module,
                    input_shapes=input_shapes,
                    output_shapes=output_shapes,
                    fp16=self.fp16,
                    int8=self.int8,
                    model_class=model_class,
                    model_kwargs=model_kwargs,
                    **param
                )
                if self.int8:
                    dic["model_pth_int8"] = model_fp32
                else:
                    dic["model_pth_fp32"] = model_fp32
            models.append(dic)
        return models

    def createInputs(self):
        if self.fp16:
            inputs_pth = self.getInputs()
            self.inputs_pth_fp16 = {key: val.half() for key, val in inputs_pth.items()}
            self.inputs_np_fp16 = {
                key: val.cpu().numpy() for key, val in self.inputs_pth_fp16.items()
            }
        elif self.int8:
            self.inputs_pth_int8 = self.getInputs()
            self.inputs_np_int8 = {
                key: val.cpu().numpy() for key, val in self.inputs_pth_int8.items()
            }
        else:
            self.inputs_pth_fp32 = self.getInputs()
            self.inputs_np_fp32 = {
                key: val.cpu().numpy() for key, val in self.inputs_pth_fp32.items()
            }

    def getInputs(self):
        torch.random.manual_seed(0)
        torch_type = torch.float32
        inputs = {}
        for key in self.input_shapes:
            inputs[key] = torch.randn(
                *self.input_shapes[key], dtype=torch_type, device=self.device
            )
        return inputs

    def buildEngine(self, opset_version):
        for dic in self.models:
            if self.int8:
                calibrator = Calibrator(self.input_shapes)
                calibrator.set_inputs(
                    self.inputs_np_fp16 if self.int8_fp16 else self.inputs_np_int8,
                )

                engine = pth2trt(
                    dic["model_pth_int8"],
                    self.inputs_pth_fp16 if self.int8_fp16 else self.inputs_pth_int8,
                    opset_version=opset_version,
                    int8=True,
                    int8_fp16=self.int8_fp16,
                    calibrator=calibrator,
                )
                dic["engine_int8"] = engine

            elif self.fp16:
                engine = pth2trt(
                    dic["model_pth_fp16"],
                    self.inputs_pth_fp16,
                    opset_version=opset_version,
                    fp16=self.fp16,
                )
                dic["engine_fp16"] = engine
            else:
                engine = pth2trt(
                    dic["model_pth_fp32"],
                    self.inputs_pth_fp32,
                    opset_version=opset_version,
                    fp16=self.fp16,
                )
                dic["engine_fp32"] = engine

    def engineForward(self, engine, fp16=False, int8=False):
        if fp16:
            outputs_trt, t = trt_forward(
                engine,
                self.inputs_np_fp16,
                self.input_shapes,
                self.output_shapes,
                self.stream,
            )
        elif int8:
            outputs_trt, t = trt_forward(
                engine,
                self.inputs_np_int8,
                self.input_shapes,
                self.output_shapes,
                self.stream,
            )
        else:
            outputs_trt, t = trt_forward(
                engine,
                self.inputs_np_fp32,
                self.input_shapes,
                self.output_shapes,
                self.stream,
            )
        for val in outputs_trt.values():
            return val, t

    def torchForward(self, model, fp16=False, int8=False):
        args = []
        if fp16:
            for val in self.inputs_pth_fp16.values():
                args.append(val)
        elif int8:
            for val in self.inputs_pth_int8.values():
                args.append(val)
        else:
            for val in self.inputs_pth_fp32.values():
                args.append(val)
        outputs_pth = model(*args)
        return outputs_pth.cpu().detach().numpy()

    @staticmethod
    def getCost(array_1, array_2):
        delta_arr = np.abs(array_1 - array_2).mean()
        return delta_arr
