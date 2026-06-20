import torch
import math
from torch.autograd import Function


class _QKV(Function):
    @staticmethod
    def symbolic(g, query, key, value):
        return g.op("QKVTRT", query, key, value)

    @staticmethod
    def forward(ctx, query, key, value):
        query = query / math.sqrt(query.shape[-1])
        qk = torch.matmul(query, key.permute(0, 2, 1)).softmax(-1)
        qkv = torch.matmul(qk, value)
        return qkv


class _QKV2(_QKV):
    @staticmethod
    def symbolic(g, query, key, value):
        return g.op("QKVTRT2", query, key, value)


_qkv = _QKV.apply
_qkv2 = _QKV2.apply


def qkv(query, key, value):
    """

    Args:
        query: [batch_size, q_len, embed_dim]
        key: [batch_size, kv_len, embed_dim]
        value: [batch_size, kv_len, embed_dim]

    Returns: [batch_size, q_len, embed_dim]

    """
    return _qkv(query, key, value)


def qkv2(query, key, value):
    """

    Args:
        query: [batch_size, q_len, embed_dim]
        key: [batch_size, kv_len, embed_dim]
        value: [batch_size, kv_len, embed_dim]

    Returns: [batch_size, q_len, embed_dim]

    """
    return _qkv2(query, key, value)
