import torch
import unittest
from ..register import TRT_FUNCTIONS
from .base_test_case import BaseTestCase

# q_len, batch_size, embed_dim
query_shape = [960, 8, 256]
key_shape = [960, 8, 256]
value_shape = [960, 8, 256]
output_shape = [960, 8, 256]

transformer_kwargs = {"embed_dim": 256, "num_heads": 8}


class Transformer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Transformer, self).__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.qkv = TRT_FUNCTIONS.get("qkv")

        self.q_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, query, key, value):
        batch_size = query.shape[1]

        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = query.view(
            -1, batch_size * self.num_heads, self.embed_dim // self.num_heads
        ).permute(1, 0, 2)
        key = key.view(
            -1, batch_size * self.num_heads, self.embed_dim // self.num_heads
        ).permute(1, 0, 2)
        value = value.view(
            -1, batch_size * self.num_heads, self.embed_dim // self.num_heads
        ).permute(1, 0, 2)

        qkv = self.qkv(query, key, value)

        qkv = qkv.permute(1, 0, 2).reshape(-1, self.embed_dim)
        output = self.out_proj(qkv).view(-1, batch_size, self.embed_dim)

        return output


class Transformer2(Transformer):
    def __init__(self, embed_dim, num_heads):
        super(Transformer2, self).__init__(embed_dim, num_heads)
        self.qkv = TRT_FUNCTIONS.get("qkv2")


class TransformerTestCase(BaseTestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func_name = args[0]

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(0)
        print("\n" + "#" * 20 + f" Running {cls.__name__} " + "#" * 20)
        cls.input_data = dict(
            query=torch.randn(query_shape, device="cuda"),
            key=torch.randn(key_shape, device="cuda"),
            value=torch.randn(value_shape, device="cuda"),
        )

    @classmethod
    def tearDownClass(cls):
        del cls.input_data
        torch.cuda.empty_cache()

    def setUp(self):
        params = [{}]
        BaseTestCase.__init__(
            self,
            "",
            input_shapes={"query": query_shape, "key": key_shape, "value": value_shape},
            output_shapes={"output": output_shape},
            params=params,
            device="cuda",
            func_name=self.func_name,
            model_class=Transformer,
            model_kwargs=transformer_kwargs,
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

    def test_fp32(self):
        self.fp32_case()

    def test_fp16(self):
        self.fp16_case(1e-4)

    def test_int8(self):
        self.int8_case(3e-3)


class TransformerTestCase2(BaseTestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func_name = args[0]

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(0)
        print("\n" + "#" * 20 + f" Running {cls.__name__} " + "#" * 20)
        cls.input_data = dict(
            query=torch.randn(query_shape, device="cuda"),
            key=torch.randn(key_shape, device="cuda"),
            value=torch.randn(value_shape, device="cuda"),
        )

    @classmethod
    def tearDownClass(cls):
        del cls.input_data
        torch.cuda.empty_cache()

    def setUp(self):
        params = [{}]
        BaseTestCase.__init__(
            self,
            "",
            input_shapes={"query": query_shape, "key": key_shape, "value": value_shape},
            output_shapes={"output": output_shape},
            params=params,
            device="cuda",
            func_name=self.func_name,
            model_class=Transformer2,
            model_kwargs=transformer_kwargs,
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

    def test_fp32(self):
        self.fp32_case()

    def test_fp16(self):
        self.fp16_case(1e-4)

    def test_int8(self):
        self.int8_case(3e-3)
