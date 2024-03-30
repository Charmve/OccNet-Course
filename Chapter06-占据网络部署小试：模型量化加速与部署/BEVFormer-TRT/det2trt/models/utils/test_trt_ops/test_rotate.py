import torch
import unittest
from .base_test_case import BaseTestCase


img_shape = [256, 512, 512]
angle_shape = [1]
center_shape = [2]
output_shape = [256, 512, 512]


class RotateTestCase(BaseTestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func_name = args[0]

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(0)
        print("\n" + "#" * 20 + f" Running {cls.__name__} " + "#" * 20)
        cls.input_data = dict(
            img=torch.randn(*img_shape, device="cuda"),
            angle=torch.randn(*angle_shape, device="cuda") * 360,
            center=torch.tensor([500.0, 500.0], device="cuda"),
        )

    @classmethod
    def tearDownClass(cls):
        del cls.input_data
        torch.cuda.empty_cache()

    def setUp(self):
        if "bilinear" in self.func_name:
            interpolation_lst = ["bilinear"]
        elif "nearest" in self.func_name:
            interpolation_lst = ["nearest"]
        else:
            raise RuntimeError(f"Not support method {self.func_name}")

        params = []
        for interpolation in interpolation_lst:
            params.append(dict(interpolation=interpolation,))

        BaseTestCase.__init__(
            self,
            "rotate",
            input_shapes={
                "img": img_shape,
                "angle": angle_shape,
                "center": center_shape,
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

    def test_fp32_bilinear(self):
        self.fp32_case(1e-4)

    def test_fp32_nearest(self):
        self.fp32_case(1e-4)

    def test_fp16_bilinear(self):
        self.fp16_case(0.5)

    def test_fp16_nearest(self):
        self.fp16_case(0.6)

    def test_int8_bilinear(self):
        self.int8_case(0.3)

    def test_int8_nearest(self):
        self.int8_case(0.5)

    def test_int8_fp16_bilinear(self):
        self.int8_fp16_case(0.3)

    def test_int8_fp16_nearest(self):
        self.int8_fp16_case(0.5)


class RotateTestCase2(BaseTestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func_name = args[0]

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(0)
        print("\n" + "#" * 20 + f" Running {cls.__name__} " + "#" * 20)
        cls.input_data = dict(
            img=torch.randn(*img_shape, device="cuda"),
            angle=torch.randn(*angle_shape, device="cuda") * 360,
            center=torch.tensor([500.0, 500.0], device="cuda"),
        )

    @classmethod
    def tearDownClass(cls):
        del cls.input_data
        torch.cuda.empty_cache()

    def setUp(self):
        if "bilinear" in self.func_name:
            interpolation_lst = ["bilinear"]
        elif "nearest" in self.func_name:
            interpolation_lst = ["nearest"]
        else:
            raise RuntimeError(f"Not support method {self.func_name}")

        params = []
        for interpolation in interpolation_lst:
            params.append(dict(interpolation=interpolation,))

        BaseTestCase.__init__(
            self,
            "rotate2",
            input_shapes={
                "img": img_shape,
                "angle": angle_shape,
                "center": center_shape,
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

    def test_fp32_bilinear(self):
        self.fp32_case(1e-4)

    def test_fp32_nearest(self):
        self.fp32_case(1e-4)

    def test_fp16_bilinear(self):
        self.fp16_case(0.5)

    def test_fp16_nearest(self):
        self.fp16_case(0.6)

    def test_int8_bilinear(self):
        self.int8_case(0.3)

    def test_int8_nearest(self):
        self.int8_case(0.5)

    def test_int8_fp16_bilinear(self):
        self.int8_fp16_case(0.3)

    def test_int8_fp16_nearest(self):
        self.int8_fp16_case(0.5)
