import torch
import numpy as np
import unittest
from .base_test_case import BaseTestCase

torch.random.manual_seed(0)
value_shape = [6, 30825, 8, 32]
reference_points_shape = [6, 40000, 1, 4 * 2]
sampling_offsets_shape = [6, 40000, 8, 4 * 8 * 2]
attention_weights_shape = [6, 40000, 8, 4 * 8]
spatial_shapes_shape = [4, 2]

output_shape = [6, 40000, 8, 32]


class MultiScaleDeformableAttnTestCase(BaseTestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func_name = args[0]

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(0)
        print("\n" + "#" * 20 + f" Running {cls.__name__} " + "#" * 20)
        cls.input_data = dict(
            value_spatial_shapes=torch.tensor(
                [[116, 200], [58, 100], [29, 50], [15, 25]], device="cuda"
            ),
            value=torch.randn(value_shape, device="cuda"),
            reference_points=torch.rand(reference_points_shape, device="cuda"),
            sampling_offsets=torch.randn(sampling_offsets_shape, device="cuda"),
            attention_weights=torch.randn(attention_weights_shape, device="cuda"),
        )

    @classmethod
    def tearDownClass(cls):
        del cls.input_data
        torch.cuda.empty_cache()

    def setUp(self):
        params = [dict()]

        BaseTestCase.__init__(
            self,
            "multi_scale_deformable_attn",
            input_shapes={
                "value": value_shape,
                "value_spatial_shapes": spatial_shapes_shape,
                "reference_points": reference_points_shape,
                "sampling_offsets": sampling_offsets_shape,
                "attention_weights": attention_weights_shape,
            },
            output_shapes={"output": output_shape},
            params=params,
            device="cuda",
            func_name=self.func_name,
        )
        self.buildEngine(opset_version=13)

    def tearDown(self):
        del self
        torch.cuda.empty_cache()

    def createInputs(self):
        f_keys = ["value", "reference_points", "sampling_offsets", "attention_weights"]
        if self.fp16:
            inputs_pth = self.getInputs()
            self.inputs_pth_fp16 = {
                key: (val.half() if key in f_keys else val)
                for key, val in inputs_pth.items()
            }
            self.inputs_np_fp16 = {
                key: (
                    val.cpu().numpy()
                    if key in f_keys
                    else val.cpu().numpy().astype(np.int32)
                )
                for key, val in self.inputs_pth_fp16.items()
            }
        elif self.int8:
            self.inputs_pth_int8 = self.getInputs()
            self.inputs_np_int8 = {
                key: (
                    val.cpu().numpy()
                    if key in f_keys
                    else val.cpu().numpy().astype(np.int32)
                )
                for key, val in self.inputs_pth_int8.items()
            }
        else:
            self.inputs_pth_fp32 = self.getInputs()
            self.inputs_np_fp32 = {
                key: (
                    val.cpu().numpy()
                    if key in f_keys
                    else val.cpu().numpy().astype(np.int32)
                )
                for key, val in self.inputs_pth_fp32.items()
            }

    def getInputs(self):
        inputs = {}
        for key in self.input_shapes:
            inputs[key] = self.__class__.input_data[key].to(torch.device(self.device))
        return inputs

    def fp32_case(self, delta=None):
        delta = self.delta if delta is None else delta
        for dic in self.models:
            output_pth = self.torchForward(dic["model_pth_fp32"], fp16=False)
            output_trt, t = self.engineForward(dic["engine_fp32"], fp16=False)
            cost = self.getCost(output_trt, output_pth)
            self.assertLessEqual(cost, delta)

    def fp16_case(self, delta=None):
        delta = self.delta if delta is None else delta
        for dic in self.models:
            output_pth = self.torchForward(dic["model_pth_fp16"], fp16=True)
            output_trt, t = self.engineForward(dic["engine_fp16"], fp16=True)
            cost = self.getCost(output_trt, output_pth)
            self.assertLessEqual(cost, delta)

    def int8_case(self, delta=None):
        delta = self.delta if delta is None else delta
        for dic in self.models:
            output_pth = self.torchForward(dic["model_pth_int8"], int8=True)
            output_trt, t = self.engineForward(dic["engine_int8"], int8=True)
            cost = self.getCost(output_trt, output_pth)
            self.assertLessEqual(cost, delta)

    def int8_fp16_case(self, delta=None):
        delta = self.delta if delta is None else delta
        for dic in self.models:
            output_pth = self.torchForward(dic["model_pth_int8"], int8=True, fp16=True)
            output_trt, t = self.engineForward(dic["engine_int8"], int8=True, fp16=True)
            cost = self.getCost(output_trt, output_pth)
            self.assertLessEqual(cost, delta)

    def test_fp32(self):
        self.fp32_case()

    def test_fp16(self):
        self.fp16_case(0.01)

    def test_int8_fp16(self):
        self.int8_fp16_case(0.01)

    def test_int8(self):
        self.int8_case(0.01)


class MultiScaleDeformableAttnTestCase2(BaseTestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func_name = args[0]

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(0)
        print("\n" + "#" * 20 + f" Running {cls.__name__} " + "#" * 20)
        cls.input_data = dict(
            value_spatial_shapes=torch.tensor(
                [[116, 200], [58, 100], [29, 50], [15, 25]], device="cuda"
            ),
            value=torch.randn(value_shape, device="cuda"),
            reference_points=torch.rand(reference_points_shape, device="cuda"),
            sampling_offsets=torch.randn(sampling_offsets_shape, device="cuda"),
            attention_weights=torch.randn(attention_weights_shape, device="cuda"),
        )

    @classmethod
    def tearDownClass(cls):
        del cls.input_data
        torch.cuda.empty_cache()

    def setUp(self):
        params = [dict()]

        BaseTestCase.__init__(
            self,
            "multi_scale_deformable_attn2",
            input_shapes={
                "value": value_shape,
                "value_spatial_shapes": spatial_shapes_shape,
                "reference_points": reference_points_shape,
                "sampling_offsets": sampling_offsets_shape,
                "attention_weights": attention_weights_shape,
            },
            output_shapes={"output": output_shape},
            params=params,
            device="cuda",
            func_name=self.func_name,
        )
        self.buildEngine(opset_version=13)

    def tearDown(self):
        del self
        torch.cuda.empty_cache()

    def createInputs(self):
        f_keys = ["value", "reference_points", "sampling_offsets", "attention_weights"]
        if self.int8_fp16:
            self.inputs_pth_int8 = self.getInputs()
            self.inputs_pth_fp16 = {
                key: (val.half() if key in f_keys else val)
                for key, val in self.inputs_pth_int8.items()
            }
            self.inputs_np_fp16 = {
                key: (
                    val.cpu().numpy()
                    if key in f_keys
                    else val.cpu().numpy().astype(np.int32)
                )
                for key, val in self.inputs_pth_fp16.items()
            }
            self.inputs_np_int8 = {
                key: (
                    val.cpu().numpy()
                    if key in f_keys
                    else val.cpu().numpy().astype(np.int32)
                )
                for key, val in self.inputs_pth_int8.items()
            }
        elif self.fp16:
            inputs_pth = self.getInputs()
            self.inputs_pth_fp16 = {
                key: (val.half() if key in f_keys else val)
                for key, val in inputs_pth.items()
            }
            self.inputs_np_fp16 = {
                key: (
                    val.cpu().numpy()
                    if key in f_keys
                    else val.cpu().numpy().astype(np.int32)
                )
                for key, val in self.inputs_pth_fp16.items()
            }
        elif self.int8:
            self.inputs_pth_int8 = self.getInputs()
            self.inputs_np_int8 = {
                key: (
                    val.cpu().numpy()
                    if key in f_keys
                    else val.cpu().numpy().astype(np.int32)
                )
                for key, val in self.inputs_pth_int8.items()
            }
        else:
            self.inputs_pth_fp32 = self.getInputs()
            self.inputs_np_fp32 = {
                key: (
                    val.cpu().numpy()
                    if key in f_keys
                    else val.cpu().numpy().astype(np.int32)
                )
                for key, val in self.inputs_pth_fp32.items()
            }

    def getInputs(self):
        inputs = {}
        for key in self.input_shapes:
            inputs[key] = self.__class__.input_data[key].to(torch.device(self.device))
        return inputs

    def fp32_case(self, delta=None):
        delta = self.delta if delta is None else delta
        for dic in self.models:
            output_pth = self.torchForward(dic["model_pth_fp32"], fp16=False)
            output_trt, t = self.engineForward(dic["engine_fp32"], fp16=False)
            cost = self.getCost(output_trt, output_pth)
            self.assertLessEqual(cost, delta)

    def fp16_case(self, delta=None):
        delta = self.delta if delta is None else delta
        for dic in self.models:
            output_pth = self.torchForward(dic["model_pth_fp16"], fp16=True)
            output_trt, t = self.engineForward(dic["engine_fp16"], fp16=True)
            cost = self.getCost(output_trt, output_pth)
            self.assertLessEqual(cost, delta)

    def int8_case(self, delta=None):
        delta = self.delta if delta is None else delta
        for dic in self.models:
            output_pth = self.torchForward(dic["model_pth_int8"], int8=True)
            output_trt, t = self.engineForward(dic["engine_int8"], int8=True)
            cost = self.getCost(output_trt, output_pth)
            self.assertLessEqual(cost, delta)

    def int8_fp16_case(self, delta=None):
        delta = self.delta if delta is None else delta
        for dic in self.models:
            output_pth = self.torchForward(dic["model_pth_int8"], int8=True, fp16=True)
            output_trt, t = self.engineForward(dic["engine_int8"], int8=True, fp16=True)
            cost = self.getCost(output_trt, output_pth)
            self.assertLessEqual(cost, delta)

    def test_fp32(self):
        self.fp32_case()

    def test_fp16(self):
        self.fp16_case(0.01)

    def test_int8_fp16(self):
        self.int8_fp16_case(0.01)

    def test_int8(self):
        self.int8_case(0.01)
