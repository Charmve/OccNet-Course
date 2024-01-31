import torch
import unittest
from .base_test_case import BaseTestCase

input_shape = [8, 256, 32, 32]
output_shape = [8, 256, 32, 32]


class InverseTestCase(BaseTestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func_name = args[0]

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(0)
        print("\n" + "#" * 20 + f" Running {cls.__name__} " + "#" * 20)
        cls.input_data = dict(
            inputs=torch.eye(input_shape[-1], device="cuda").repeat(
                *input_shape[:-2], 1, 1
            ),
        )

    @classmethod
    def tearDownClass(cls):
        del cls.input_data
        torch.cuda.empty_cache()

    def setUp(self):
        params = [{}]
        BaseTestCase.__init__(
            self,
            "inverse",
            input_shapes={"inputs": input_shape,},
            output_shapes={"output": output_shape},
            params=params,
            device="cpu",
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

    def test_fp32(self):
        self.fp32_case()
