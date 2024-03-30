import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from mmcv.utils import ext_loader

ext_module = ext_loader.load_ext(
    "_ext", ["modulated_deform_conv_forward", "modulated_deform_conv_backward"]
)


class _ModulatedDeformableConv2dFunction(Function):
    @staticmethod
    def symbolic(
        g,
        input,
        offset,
        mask,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        deform_groups,
    ):
        input_tensors = [input, offset, mask, weight]
        if bias is not None:
            input_tensors.append(bias)
        return g.op(
            "ModulatedDeformableConv2dTRT",
            *input_tensors,
            stride_i=_pair(stride),
            padding_i=_pair(padding),
            dilation_i=_pair(dilation),
            groups_i=groups,
            deform_groups_i=deform_groups,
        )

    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        mask,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deform_groups=1,
    ):
        if input is not None and input.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead."
            )
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deform_groups = deform_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor
        # When pytorch version >= 1.6.0, amp is adopted for fp16 mode;
        # amp won't cast the type of model (float32), but "offset" is cast
        # to float16 by nn.Conv2d automatically, leading to the type
        # mismatch with input (when it is float32) or weight.
        # The flag for whether to use fp16 or amp is the type of "offset",
        # we cast weight and input to temporarily support fp16 and amp
        # whatever the pytorch version is.
        input = input.type_as(offset)
        weight = weight.type_as(input)
        bias = bias.type_as(input)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(
            _ModulatedDeformableConv2dFunction._output_size(ctx, input, weight)
        )

        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        ext_module.modulated_deform_conv_forward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            output,
            ctx._bufs[1],
            kernel_h=weight.size(2),
            kernel_w=weight.size(3),
            stride_h=ctx.stride[0],
            stride_w=ctx.stride[1],
            pad_h=ctx.padding[0],
            pad_w=ctx.padding[1],
            dilation_h=ctx.dilation[0],
            dilation_w=ctx.dilation[1],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            with_bias=ctx.with_bias,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        grad_output = grad_output.contiguous()
        ext_module.modulated_deform_conv_backward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            ctx._bufs[1],
            grad_input,
            grad_weight,
            grad_bias,
            grad_offset,
            grad_mask,
            grad_output,
            kernel_h=weight.size(2),
            kernel_w=weight.size(3),
            stride_h=ctx.stride[0],
            stride_w=ctx.stride[1],
            pad_h=ctx.padding[0],
            pad_w=ctx.padding[1],
            dilation_h=ctx.dilation[0],
            dilation_w=ctx.dilation[1],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            with_bias=ctx.with_bias,
        )
        if not ctx.with_bias:
            grad_bias = None

        return (
            grad_input,
            grad_offset,
            grad_mask,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def _output_size(ctx, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = ctx.padding[d]
            kernel = ctx.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = ctx.stride[d]
            output_size += (
                torch.div(in_size + (2 * pad) - kernel, stride_, rounding_mode="trunc")
                + 1,
            )
        return output_size


class _ModulatedDeformableConv2dFunction2(_ModulatedDeformableConv2dFunction):
    @staticmethod
    def symbolic(
        g,
        input,
        offset,
        mask,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        deform_groups,
    ):
        input_tensors = [input, offset, mask, weight]
        if bias is not None:
            input_tensors.append(bias)
        return g.op(
            "ModulatedDeformableConv2dTRT2",
            *input_tensors,
            stride_i=_pair(stride),
            padding_i=_pair(padding),
            dilation_i=_pair(dilation),
            groups_i=groups,
            deform_groups_i=deform_groups,
        )


_modulated_deformable_conv2d = _ModulatedDeformableConv2dFunction.apply
_modulated_deformable_conv2d2 = _ModulatedDeformableConv2dFunction2.apply


def modulated_deformable_conv2d(
    input,
    offset,
    mask,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    deform_groups=1,
):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Support TensorRT plugin ModulatedDeformableConv2dTRT: FP32 and FP16(nv_half).

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """
    return _modulated_deformable_conv2d(
        input,
        offset,
        mask,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        deform_groups,
    )


def modulated_deformable_conv2d2(
    input,
    offset,
    mask,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    deform_groups=1,
):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Support TensorRT plugin ModulatedDeformableConv2dTRT2: FP32 and FP16(nv_half2).

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """
    return _modulated_deformable_conv2d2(
        input,
        offset,
        mask,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        deform_groups,
    )
