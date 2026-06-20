import torch
from torch.autograd import Function


class _Inverse(Function):
    @staticmethod
    def symbolic(g, inputs):
        return g.op("InverseTRT", inputs)

    @staticmethod
    def forward(ctx, inputs):
        outputs = torch.linalg.inv(inputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


_inverse = _Inverse.apply


def inverse(inputs):
    return _inverse(inputs)
