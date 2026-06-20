import torch
import unittest
from .base_test_case import BaseTestCase

input_shape_2d = [8, 32, 100, 100]
grid_shape_2d = [8, 2, 1001, 1001]
output_shape_2d = [8, 32, 1001, 1001]

input_shape_3d = [8, 32, 10, 10, 10]
grid_shape_3d = [8, 3, 101, 101, 101]
output_shape_3d = [8, 32, 101, 101, 101]


class GridSampler2DTestCase(BaseTestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func_name = args[0]

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(0)
        print("\n" + "#" * 20 + f" Running {cls.__name__} " + "#" * 20)
        new_h = (
            torch.linspace(-15, 15, grid_shape_2d[2], device="cuda")
            .view(-1, 1)
            .repeat(1, grid_shape_2d[3])
        )
        new_w = torch.linspace(-15, 15, grid_shape_2d[3], device="cuda").repeat(
            grid_shape_2d[2], 1
        )
        cls.input_data = dict(
            input=torch.randn(*input_shape_2d, device="cuda"),
            grid=torch.cat((new_h.unsqueeze(0), new_w.unsqueeze(0)), dim=0)
            .unsqueeze(0)
            .repeat(grid_shape_2d[0], 1, 1, 1),
        )

    @classmethod
    def tearDownClass(cls):
        del cls.input_data
        torch.cuda.empty_cache()

    def setUp(self):
        if "bilinear" in self.func_name:
            interpolation_mode_lst = ["bilinear"]
        elif "nearest" in self.func_name:
            interpolation_mode_lst = ["nearest"]
        elif "bicubic" in self.func_name:
            interpolation_mode_lst = ["bicubic"]
        else:
            raise RuntimeError(f"Not support method {self.func_name}")

        if "zeros" in self.func_name:
            padding_mode_lst = ["zeros"]
        elif "border" in self.func_name:
            padding_mode_lst = ["border"]
        elif "reflection" in self.func_name:
            padding_mode_lst = ["reflection"]
        else:
            raise RuntimeError(f"Not support method {self.func_name}")

        if "alignCorners" in self.func_name:
            align_corners_lst = [True]
        elif "NoAlignCorners" in self.func_name:
            align_corners_lst = [False]
        else:
            raise RuntimeError(f"Not support method {self.func_name}")

        params = []
        for interpolation_mode in interpolation_mode_lst:
            for padding_mode in padding_mode_lst:
                for align_corners in align_corners_lst:
                    params.append(
                        dict(
                            interpolation_mode=interpolation_mode,
                            padding_mode=padding_mode,
                            align_corners=align_corners,
                        )
                    )

        BaseTestCase.__init__(
            self,
            "grid_sampler",
            input_shapes={"input": input_shape_2d, "grid": grid_shape_2d},
            output_shapes={"output": output_shape_2d},
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

    def test_fp32_bilinear_zeros_alignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_zeros_NoAlignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_border_alignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_border_NoAlignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_reflection_alignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_reflection_NoAlignCorners(self):
        self.fp32_case()

    def test_fp32_nearest_zeros_alignCorners(self):
        self.fp32_case()

    def test_fp32_nearest_zeros_NoAlignCorners(self):
        self.fp32_case(0.1)

    def test_fp32_nearest_border_alignCorners(self):
        self.fp32_case()

    def test_fp32_nearest_border_NoAlignCorners(self):
        self.fp32_case(0.1)

    def test_fp32_nearest_reflection_alignCorners(self):
        self.fp32_case()

    def test_fp32_nearest_reflection_NoAlignCorners(self):
        self.fp32_case(0.1)

    def test_fp32_bicubic_zeros_alignCorners(self):
        self.fp32_case()

    def test_fp32_bicubic_zeros_NoAlignCorners(self):
        self.fp32_case()

    def test_fp32_bicubic_border_alignCorners(self):
        self.fp32_case()

    def test_fp32_bicubic_border_NoAlignCorners(self):
        self.fp32_case()

    def test_fp32_bicubic_reflection_alignCorners(self):
        self.fp32_case()

    def test_fp32_bicubic_reflection_NoAlignCorners(self):
        self.fp32_case()

    def test_fp16_bilinear_zeros_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_zeros_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_border_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_border_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_reflection_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_reflection_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_nearest_zeros_alignCorners(self):
        self.fp16_case(0.1)

    def test_fp16_nearest_zeros_NoAlignCorners(self):
        self.fp16_case(0.2)

    def test_fp16_nearest_border_alignCorners(self):
        self.fp16_case(0.1)

    def test_fp16_nearest_border_NoAlignCorners(self):
        self.fp16_case(0.2)

    def test_fp16_nearest_reflection_alignCorners(self):
        self.fp16_case(0.1)

    def test_fp16_nearest_reflection_NoAlignCorners(self):
        self.fp16_case(0.2)

    def test_fp16_bicubic_zeros_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bicubic_zeros_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bicubic_border_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bicubic_border_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bicubic_reflection_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bicubic_reflection_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_int8_bilinear_zeros_alignCorners(self):
        self.int8_case(0.1)

    def test_int8_bilinear_zeros_NoAlignCorners(self):
        self.int8_case(0.1)

    def test_int8_bilinear_border_alignCorners(self):
        self.int8_case(0.2)

    def test_int8_bilinear_border_NoAlignCorners(self):
        self.int8_case(0.2)

    def test_int8_bilinear_reflection_alignCorners(self):
        self.int8_case(0.25)

    def test_int8_bilinear_reflection_NoAlignCorners(self):
        self.int8_case(0.25)

    def test_int8_nearest_zeros_alignCorners(self):
        self.int8_case(0.2)

    def test_int8_nearest_zeros_NoAlignCorners(self):
        self.int8_case(0.2)

    def test_int8_nearest_border_alignCorners(self):
        self.int8_case(0.3)

    def test_int8_nearest_border_NoAlignCorners(self):
        self.int8_case(0.3)

    def test_int8_nearest_reflection_alignCorners(self):
        self.int8_case(0.4)

    def test_int8_nearest_reflection_NoAlignCorners(self):
        self.int8_case(0.4)

    def test_int8_bicubic_zeros_alignCorners(self):
        self.int8_case(0.15)

    def test_int8_bicubic_zeros_NoAlignCorners(self):
        self.int8_case(0.15)

    def test_int8_bicubic_border_alignCorners(self):
        self.int8_case(0.25)

    def test_int8_bicubic_border_NoAlignCorners(self):
        self.int8_case(0.25)

    def test_int8_bicubic_reflection_alignCorners(self):
        self.int8_case(0.3)

    def test_int8_bicubic_reflection_NoAlignCorners(self):
        self.int8_case(0.3)


class GridSampler3DTestCase(BaseTestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func_name = args[0]

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(0)
        print("\n" + "#" * 20 + f" Running {cls.__name__} " + "#" * 20)
        new_d = (
            torch.linspace(-15, 15, grid_shape_3d[2], device="cuda")
            .view(-1, 1, 1)
            .repeat(1, grid_shape_3d[3], grid_shape_3d[4])
        )
        new_h = (
            torch.linspace(-15, 15, grid_shape_3d[3], device="cuda")
            .view(1, -1, 1)
            .repeat(grid_shape_3d[2], 1, grid_shape_3d[4])
        )
        new_w = (
            torch.linspace(-15, 15, grid_shape_3d[4], device="cuda")
            .view(1, 1, -1)
            .repeat(grid_shape_3d[2], grid_shape_3d[3], 1)
        )
        cls.input_data = dict(
            input=torch.randn(*input_shape_3d, device="cuda"),
            grid=torch.cat(
                (new_d.unsqueeze(0), new_h.unsqueeze(0), new_w.unsqueeze(0)), dim=0
            )
            .unsqueeze(0)
            .repeat(grid_shape_3d[0], 1, 1, 1, 1),
        )

    @classmethod
    def tearDownClass(cls):
        del cls.input_data
        torch.cuda.empty_cache()

    def setUp(self):
        if "bilinear" in self.func_name:
            interpolation_mode_lst = ["bilinear"]
        elif "nearest" in self.func_name:
            interpolation_mode_lst = ["nearest"]
        else:
            raise RuntimeError(f"Not support method {self.func_name}")

        if "zeros" in self.func_name:
            padding_mode_lst = ["zeros"]
        elif "border" in self.func_name:
            padding_mode_lst = ["border"]
        elif "reflection" in self.func_name:
            padding_mode_lst = ["reflection"]
        else:
            raise RuntimeError(f"Not support method {self.func_name}")

        if "alignCorners" in self.func_name:
            align_corners_lst = [True]
        elif "NoAlignCorners" in self.func_name:
            align_corners_lst = [False]
        else:
            raise RuntimeError(f"Not support method {self.func_name}")

        params = []
        for interpolation_mode in interpolation_mode_lst:
            for padding_mode in padding_mode_lst:
                for align_corners in align_corners_lst:
                    params.append(
                        dict(
                            interpolation_mode=interpolation_mode,
                            padding_mode=padding_mode,
                            align_corners=align_corners,
                        )
                    )

        BaseTestCase.__init__(
            self,
            "grid_sampler",
            input_shapes={"input": input_shape_3d, "grid": grid_shape_3d},
            output_shapes={"output": output_shape_3d},
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

    def test_fp32_bilinear_zeros_alignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_zeros_NoAlignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_border_alignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_border_NoAlignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_reflection_alignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_reflection_NoAlignCorners(self):
        self.fp32_case()

    def test_fp32_nearest_zeros_alignCorners(self):
        self.fp32_case()

    def test_fp32_nearest_zeros_NoAlignCorners(self):
        self.fp32_case(0.1)

    def test_fp32_nearest_border_alignCorners(self):
        self.fp32_case()

    def test_fp32_nearest_border_NoAlignCorners(self):
        self.fp32_case(0.1)

    def test_fp32_nearest_reflection_alignCorners(self):
        self.fp32_case()

    def test_fp32_nearest_reflection_NoAlignCorners(self):
        self.fp32_case(0.1)

    def test_fp16_bilinear_zeros_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_zeros_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_border_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_border_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_reflection_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_reflection_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_nearest_zeros_alignCorners(self):
        self.fp16_case(0.1)

    def test_fp16_nearest_zeros_NoAlignCorners(self):
        self.fp16_case(0.2)

    def test_fp16_nearest_border_alignCorners(self):
        self.fp16_case(0.1)

    def test_fp16_nearest_border_NoAlignCorners(self):
        self.fp16_case(0.2)

    def test_fp16_nearest_reflection_alignCorners(self):
        self.fp16_case(0.1)

    def test_fp16_nearest_reflection_NoAlignCorners(self):
        self.fp16_case(0.2)


class GridSampler2DTestCase2(BaseTestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func_name = args[0]

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(0)
        print("\n" + "#" * 20 + f" Running {cls.__name__} " + "#" * 20)
        new_h = (
            torch.linspace(-15, 15, grid_shape_2d[2], device="cuda")
            .view(-1, 1)
            .repeat(1, grid_shape_2d[3])
        )
        new_w = torch.linspace(-15, 15, grid_shape_2d[3], device="cuda").repeat(
            grid_shape_2d[2], 1
        )
        cls.input_data = dict(
            input=torch.randn(*input_shape_2d, device="cuda"),
            grid=torch.cat((new_h.unsqueeze(0), new_w.unsqueeze(0)), dim=0)
            .unsqueeze(0)
            .repeat(grid_shape_2d[0], 1, 1, 1),
        )

    @classmethod
    def tearDownClass(cls):
        del cls.input_data
        torch.cuda.empty_cache()

    def setUp(self):
        if "bilinear" in self.func_name:
            interpolation_mode_lst = ["bilinear"]
        elif "nearest" in self.func_name:
            interpolation_mode_lst = ["nearest"]
        elif "bicubic" in self.func_name:
            interpolation_mode_lst = ["bicubic"]
        else:
            raise RuntimeError(f"Not support method {self.func_name}")

        if "zeros" in self.func_name:
            padding_mode_lst = ["zeros"]
        elif "border" in self.func_name:
            padding_mode_lst = ["border"]
        elif "reflection" in self.func_name:
            padding_mode_lst = ["reflection"]
        else:
            raise RuntimeError(f"Not support method {self.func_name}")

        if "alignCorners" in self.func_name:
            align_corners_lst = [True]
        elif "NoAlignCorners" in self.func_name:
            align_corners_lst = [False]
        else:
            raise RuntimeError(f"Not support method {self.func_name}")

        params = []
        for interpolation_mode in interpolation_mode_lst:
            for padding_mode in padding_mode_lst:
                for align_corners in align_corners_lst:
                    params.append(
                        dict(
                            interpolation_mode=interpolation_mode,
                            padding_mode=padding_mode,
                            align_corners=align_corners,
                        )
                    )

        BaseTestCase.__init__(
            self,
            "grid_sampler2",
            input_shapes={"input": input_shape_2d, "grid": grid_shape_2d},
            output_shapes={"output": output_shape_2d},
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

    def test_fp32_bilinear_zeros_alignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_zeros_NoAlignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_border_alignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_border_NoAlignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_reflection_alignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_reflection_NoAlignCorners(self):
        self.fp32_case()

    def test_fp32_nearest_zeros_alignCorners(self):
        self.fp32_case()

    def test_fp32_nearest_zeros_NoAlignCorners(self):
        self.fp32_case(0.1)

    def test_fp32_nearest_border_alignCorners(self):
        self.fp32_case()

    def test_fp32_nearest_border_NoAlignCorners(self):
        self.fp32_case(0.1)

    def test_fp32_nearest_reflection_alignCorners(self):
        self.fp32_case()

    def test_fp32_nearest_reflection_NoAlignCorners(self):
        self.fp32_case(0.1)

    def test_fp32_bicubic_zeros_alignCorners(self):
        self.fp32_case()

    def test_fp32_bicubic_zeros_NoAlignCorners(self):
        self.fp32_case()

    def test_fp32_bicubic_border_alignCorners(self):
        self.fp32_case()

    def test_fp32_bicubic_border_NoAlignCorners(self):
        self.fp32_case()

    def test_fp32_bicubic_reflection_alignCorners(self):
        self.fp32_case()

    def test_fp32_bicubic_reflection_NoAlignCorners(self):
        self.fp32_case()

    def test_fp16_bilinear_zeros_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_zeros_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_border_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_border_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_reflection_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_reflection_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_nearest_zeros_alignCorners(self):
        self.fp16_case(0.1)

    def test_fp16_nearest_zeros_NoAlignCorners(self):
        self.fp16_case(0.2)

    def test_fp16_nearest_border_alignCorners(self):
        self.fp16_case(0.1)

    def test_fp16_nearest_border_NoAlignCorners(self):
        self.fp16_case(0.2)

    def test_fp16_nearest_reflection_alignCorners(self):
        self.fp16_case(0.1)

    def test_fp16_nearest_reflection_NoAlignCorners(self):
        self.fp16_case(0.2)

    def test_fp16_bicubic_zeros_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bicubic_zeros_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bicubic_border_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bicubic_border_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bicubic_reflection_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bicubic_reflection_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_int8_bilinear_zeros_alignCorners(self):
        self.int8_case(0.1)

    def test_int8_bilinear_zeros_NoAlignCorners(self):
        self.int8_case(0.1)

    def test_int8_bilinear_border_alignCorners(self):
        self.int8_case(0.2)

    def test_int8_bilinear_border_NoAlignCorners(self):
        self.int8_case(0.2)

    def test_int8_bilinear_reflection_alignCorners(self):
        self.int8_case(0.25)

    def test_int8_bilinear_reflection_NoAlignCorners(self):
        self.int8_case(0.25)

    def test_int8_nearest_zeros_alignCorners(self):
        self.int8_case(0.2)

    def test_int8_nearest_zeros_NoAlignCorners(self):
        self.int8_case(0.2)

    def test_int8_nearest_border_alignCorners(self):
        self.int8_case(0.3)

    def test_int8_nearest_border_NoAlignCorners(self):
        self.int8_case(0.3)

    def test_int8_nearest_reflection_alignCorners(self):
        self.int8_case(0.4)

    def test_int8_nearest_reflection_NoAlignCorners(self):
        self.int8_case(0.4)

    def test_int8_bicubic_zeros_alignCorners(self):
        self.int8_case(0.15)

    def test_int8_bicubic_zeros_NoAlignCorners(self):
        self.int8_case(0.15)

    def test_int8_bicubic_border_alignCorners(self):
        self.int8_case(0.25)

    def test_int8_bicubic_border_NoAlignCorners(self):
        self.int8_case(0.25)

    def test_int8_bicubic_reflection_alignCorners(self):
        self.int8_case(0.3)

    def test_int8_bicubic_reflection_NoAlignCorners(self):
        self.int8_case(0.3)


class GridSampler3DTestCase2(BaseTestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func_name = args[0]

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(0)
        print("\n" + "#" * 20 + f" Running {cls.__name__} " + "#" * 20)
        new_d = (
            torch.linspace(-15, 15, grid_shape_3d[2], device="cuda")
            .view(-1, 1, 1)
            .repeat(1, grid_shape_3d[3], grid_shape_3d[4])
        )
        new_h = (
            torch.linspace(-15, 15, grid_shape_3d[3], device="cuda")
            .view(1, -1, 1)
            .repeat(grid_shape_3d[2], 1, grid_shape_3d[4])
        )
        new_w = (
            torch.linspace(-15, 15, grid_shape_3d[4], device="cuda")
            .view(1, 1, -1)
            .repeat(grid_shape_3d[2], grid_shape_3d[3], 1)
        )
        cls.input_data = dict(
            input=torch.randn(*input_shape_3d, device="cuda"),
            grid=torch.cat(
                (new_d.unsqueeze(0), new_h.unsqueeze(0), new_w.unsqueeze(0)), dim=0
            )
            .unsqueeze(0)
            .repeat(grid_shape_3d[0], 1, 1, 1, 1),
        )

    @classmethod
    def tearDownClass(cls):
        del cls.input_data
        torch.cuda.empty_cache()

    def setUp(self):
        if "bilinear" in self.func_name:
            interpolation_mode_lst = ["bilinear"]
        elif "nearest" in self.func_name:
            interpolation_mode_lst = ["nearest"]
        else:
            raise RuntimeError(f"Not support method {self.func_name}")

        if "zeros" in self.func_name:
            padding_mode_lst = ["zeros"]
        elif "border" in self.func_name:
            padding_mode_lst = ["border"]
        elif "reflection" in self.func_name:
            padding_mode_lst = ["reflection"]
        else:
            raise RuntimeError(f"Not support method {self.func_name}")

        if "alignCorners" in self.func_name:
            align_corners_lst = [True]
        elif "NoAlignCorners" in self.func_name:
            align_corners_lst = [False]
        else:
            raise RuntimeError(f"Not support method {self.func_name}")

        params = []
        for interpolation_mode in interpolation_mode_lst:
            for padding_mode in padding_mode_lst:
                for align_corners in align_corners_lst:
                    params.append(
                        dict(
                            interpolation_mode=interpolation_mode,
                            padding_mode=padding_mode,
                            align_corners=align_corners,
                        )
                    )

        BaseTestCase.__init__(
            self,
            "grid_sampler2",
            input_shapes={"input": input_shape_3d, "grid": grid_shape_3d},
            output_shapes={"output": output_shape_3d},
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

    def test_fp32_bilinear_zeros_alignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_zeros_NoAlignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_border_alignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_border_NoAlignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_reflection_alignCorners(self):
        self.fp32_case()

    def test_fp32_bilinear_reflection_NoAlignCorners(self):
        self.fp32_case()

    def test_fp32_nearest_zeros_alignCorners(self):
        self.fp32_case()

    def test_fp32_nearest_zeros_NoAlignCorners(self):
        self.fp32_case(0.1)

    def test_fp32_nearest_border_alignCorners(self):
        self.fp32_case()

    def test_fp32_nearest_border_NoAlignCorners(self):
        self.fp32_case(0.1)

    def test_fp32_nearest_reflection_alignCorners(self):
        self.fp32_case()

    def test_fp32_nearest_reflection_NoAlignCorners(self):
        self.fp32_case(0.1)

    def test_fp16_bilinear_zeros_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_zeros_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_border_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_border_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_reflection_alignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_bilinear_reflection_NoAlignCorners(self):
        self.fp16_case(0.05)

    def test_fp16_nearest_zeros_alignCorners(self):
        self.fp16_case(0.1)

    def test_fp16_nearest_zeros_NoAlignCorners(self):
        self.fp16_case(0.2)

    def test_fp16_nearest_border_alignCorners(self):
        self.fp16_case(0.1)

    def test_fp16_nearest_border_NoAlignCorners(self):
        self.fp16_case(0.2)

    def test_fp16_nearest_reflection_alignCorners(self):
        self.fp16_case(0.1)

    def test_fp16_nearest_reflection_NoAlignCorners(self):
        self.fp16_case(0.2)
