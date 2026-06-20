import torch
import unittest
from .base_test_case import BaseTestCase

torch.random.manual_seed(0)
input_shape = [8, 256, 256, 256]
offset_shape = [8, 2 * 2 * 3 * 3, 256, 256]
mask_shape = [8, 2 * 3 * 3, 256, 256]
weight_shape = [256, 256 // 2, 3, 3]
bias_shape = [256]
output_shape = [8, 256, 256, 256]


class ModulatedDeformableConv2dTestCase(BaseTestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func_name = args[0]

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(0)
        print("\n" + "#" * 20 + f" Running {cls.__name__} " + "#" * 20)
        cls.input_data = dict(
            input=torch.randn(input_shape, device="cuda"),
            offset=torch.randn(offset_shape, device="cuda"),
            mask=torch.randn(mask_shape, device="cuda"),
            weight=torch.randn(weight_shape, device="cuda"),
            bias=torch.randn(bias_shape, device="cuda"),
        )

    @classmethod
    def tearDownClass(cls):
        del cls.input_data
        torch.cuda.empty_cache()

    def setUp(self):
        params = [dict(stride=1, padding=1, dilation=1, groups=2, deform_groups=2)]

        BaseTestCase.__init__(
            self,
            "modulated_deformable_conv2d",
            input_shapes={
                "input": input_shape,
                "offset": offset_shape,
                "mask": mask_shape,
                "weight": weight_shape,
                "bias": bias_shape,
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
        self.fp16_case(0.05)

    def test_int8(self):
        self.int8_case(1.5)

    def test_int8_fp16(self):
        self.int8_fp16_case(1.5)


class ModulatedDeformableConv2dTestCase2(BaseTestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func_name = args[0]

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(0)
        print("\n" + "#" * 20 + f" Running {cls.__name__} " + "#" * 20)
        cls.input_data = dict(
            input=torch.randn(input_shape, device="cuda"),
            offset=torch.randn(offset_shape, device="cuda"),
            mask=torch.randn(mask_shape, device="cuda"),
            weight=torch.randn(weight_shape, device="cuda"),
            bias=torch.randn(bias_shape, device="cuda"),
        )

    @classmethod
    def tearDownClass(cls):
        del cls.input_data
        torch.cuda.empty_cache()

    def setUp(self):
        params = [dict(stride=1, padding=1, dilation=1, groups=2, deform_groups=2)]

        BaseTestCase.__init__(
            self,
            "modulated_deformable_conv2d2",
            input_shapes={
                "input": input_shape,
                "offset": offset_shape,
                "mask": mask_shape,
                "weight": weight_shape,
                "bias": bias_shape,
            },
            output_shapes={"output": output_shape},
            params=params,
            device="cuda",
            func_name=self.func_name,
        )
        self.buildEngine(opset_version=13)

    def tearDown(self):
        del self

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
        self.fp16_case(0.05)

    def test_int8(self):
        self.int8_case(1.5)

    def test_int8_fp16(self):
        self.int8_fp16_case(1.5)
