import torch
from torch import Tensor
from torch.types import _int, _bool
from torch.autograd import Function


class _GridSampler2D(Function):
    @staticmethod
    def symbolic(g, input, grid, interpolation_mode, padding_mode, align_corners):
        return g.op(
            "GridSampler2DTRT",
            input,
            grid,
            interpolation_mode_i=interpolation_mode,
            padding_mode_i=padding_mode,
            align_corners_i=align_corners,
        )

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        grid: Tensor,
        interpolation_mode: _int,
        padding_mode: _int,
        align_corners: _bool,
    ):
        grid = grid.permute(0, 2, 3, 1)
        grid_ = grid / 10
        output = torch.ops.aten.grid_sampler(
            input, grid_, interpolation_mode, padding_mode, align_corners
        )
        ctx.save_for_backward(input, grid_)
        ctx.interpolation_mode = interpolation_mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        input, grid = ctx.saved_tensors
        interpolation_mode = ctx.interpolation_mode
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        input_grad, grid_grad = torch.ops.aten.grid_sampler_2d_backward(
            grad_outputs,
            input,
            grid,
            interpolation_mode,
            padding_mode,
            align_corners,
            [input.requires_grad, grid.requires_grad],
        )
        grid_grad = grid_grad.permute(0, 3, 1, 2)
        return input_grad, grid_grad * 10, None, None, None


class _GridSampler3D(Function):
    @staticmethod
    def symbolic(g, input, grid, interpolation_mode, padding_mode, align_corners):
        return g.op(
            "GridSampler3DTRT",
            input,
            grid,
            interpolation_mode_i=interpolation_mode,
            padding_mode_i=padding_mode,
            align_corners_i=align_corners,
        )

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        grid: Tensor,
        interpolation_mode: _int,
        padding_mode: _int,
        align_corners: _bool,
    ):
        grid = grid.permute(0, 2, 3, 4, 1)
        grid_ = grid / 10
        output = torch.ops.aten.grid_sampler(
            input, grid_, interpolation_mode, padding_mode, align_corners
        )
        ctx.save_for_backward(input, grid_)
        ctx.interpolation_mode = interpolation_mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        input, grid = ctx.saved_tensors
        interpolation_mode = ctx.interpolation_mode
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        input_grad, grid_grad = torch.ops.aten.grid_sampler_3d_backward(
            grad_outputs,
            input,
            grid,
            interpolation_mode,
            padding_mode,
            align_corners,
            [input.requires_grad, grid.requires_grad],
        )
        grid_grad = grid_grad.permute(0, 4, 1, 2, 3)
        return input_grad, grid_grad * 10, None, None, None


class _GridSampler2D2(_GridSampler2D):
    @staticmethod
    def symbolic(g, input, grid, interpolation_mode, padding_mode, align_corners):
        return g.op(
            "GridSampler2DTRT2",
            input,
            grid,
            interpolation_mode_i=interpolation_mode,
            padding_mode_i=padding_mode,
            align_corners_i=align_corners,
        )


class _GridSampler3D2(_GridSampler3D):
    @staticmethod
    def symbolic(g, input, grid, interpolation_mode, padding_mode, align_corners):
        return g.op(
            "GridSampler3DTRT2",
            input,
            grid,
            interpolation_mode_i=interpolation_mode,
            padding_mode_i=padding_mode,
            align_corners_i=align_corners,
        )


_grid_sampler2d = _GridSampler2D.apply
_grid_sampler2d2 = _GridSampler2D2.apply
_grid_sampler3d = _GridSampler3D.apply
_grid_sampler3d2 = _GridSampler3D2.apply
_MODE = {"bilinear": 0, "nearest": 1, "bicubic": 2}

_PAD = {"zeros": 0, "border": 1, "reflection": 2}


def grid_sampler(
    input: Tensor,
    grid: Tensor,
    interpolation_mode: str,
    padding_mode: str,
    align_corners: _bool,
):
    r"""Currently, only spatial (4-D) and volumetric (5-D) :attr:`input` are
    supported.

    Support TensorRT plugin GridSamplerTRT: FP32 and FP16(nv_half).

    In the spatial (4-D) case, for :attr:`input` with shape
    :math:`(N, C, H_\text{in}, W_\text{in})` and :attr:`grid` with shape
    :math:`(N, 2, H_\text{out}, W_\text{out})`, the output will have shape
    :math:`(N, C, H_\text{out}, W_\text{out})`.

    For each output location ``output[n, :, h, w]``, the size-2 vector
    ``grid[n, :, h, w]`` specifies :attr:`input` pixel locations ``x`` and ``y``,
    which are used to interpolate the output value ``output[n, :, h, w]``.
    In the case of 5D inputs, ``grid[n, :, d, h, w]`` specifies the
    ``x``, ``y``, ``z`` pixel locations for interpolating
    ``output[n, :, d, h, w]``. :attr:`mode` argument specifies ``nearest`` or
    ``bilinear`` interpolation method to sample the input pixels.

    :attr:`grid` specifies the sampling pixel locations normalized by the
    :attr:`input` spatial dimensions. Therefore, it should have most values in
    the range of ``[-10, 10]``. For example, values ``x = -10, y = -10`` is the
    left-top pixel of :attr:`input`, and values  ``x = 10, y = 10`` is the
    right-bottom pixel of :attr:`input`.

    If :attr:`grid` has values outside the range of ``[-10, 10]``, the corresponding
    outputs are handled as defined by :attr:`padding_mode`. Options are

        * ``padding_mode="zeros"``: use ``0`` for out-of-bound grid locations,
        * ``padding_mode="border"``: use border values for out-of-bound grid locations,
        * ``padding_mode="reflection"``: use values at locations reflected by
          the border for out-of-bound grid locations. For location far away
          from the border, it will keep being reflected until becoming in bound,
          e.g., (normalized) pixel location ``x = -35`` reflects by border ``-10``
          and becomes ``x' = 15``, then reflects by border ``10`` and becomes
          ``x'' = 5``.

    Args:
        input (Tensor): input of shape :math:`(N, C, H_\text{in}, W_\text{in})` (4-D case)
                        or :math:`(N, C, D_\text{in}, H_\text{in}, W_\text{in})` (5-D case)
        grid (Tensor): flow-field of shape :math:`(N, 2, H_\text{out}, W_\text{out})` (4-D case)
                       or :math:`(N, 3, D_\text{out}, H_\text{out}, W_\text{out})` (5-D case)
        interpolation_mode (str): interpolation mode to calculate output values
            ``'bilinear'`` | ``'nearest'`` | ``'bicubic'``. Default: ``'bilinear'``
            When ``mode='bilinear'`` and the input is 5-D, the interpolation mode
            used internally will actually be trilinear. However, when the input is 4-D,
            the interpolation mode will legitimately be bilinear.
            Note: ``mode='bicubic'`` supports only 4-D input.
        padding_mode (str): padding mode for outside grid values
            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input  as squares rather than points.
            If set to ``True``, the extrema (``-1`` and ``1``) are considered as referring
            to the center points of the input's corner pixels. If set to ``False``, they
            are instead considered as referring to the corner points of the input's corner
            pixels, making the sampling more resolution agnostic.
            This option parallels the ``align_corners`` option in
            :func:`interpolate`, and so whichever option is used here
            should also be used there to resize the input image before grid sampling.
            Default: ``False``

    Returns:
        output (Tensor): output Tensor
    """
    if grid.dim() == 4:
        return _grid_sampler2d(
            input, grid, _MODE[interpolation_mode], _PAD[padding_mode], align_corners
        )
    elif grid.dim() == 5:
        return _grid_sampler3d(
            input, grid, _MODE[interpolation_mode], _PAD[padding_mode], align_corners
        )
    else:
        raise RuntimeError


def grid_sampler2(
    input: Tensor,
    grid: Tensor,
    interpolation_mode: str,
    padding_mode: str,
    align_corners: _bool,
):
    r"""Currently, only spatial (4-D) and volumetric (5-D) :attr:`input` are
    supported.

    Support TensorRT plugin GridSamplerTRT2: FP32 and FP16(nv_half2).

    In the spatial (4-D) case, for :attr:`input` with shape
    :math:`(N, C, H_\text{in}, W_\text{in})` and :attr:`grid` with shape
    :math:`(N, 2, H_\text{out}, W_\text{out})`, the output will have shape
    :math:`(N, C, H_\text{out}, W_\text{out})`.

    For each output location ``output[n, :, h, w]``, the size-2 vector
    ``grid[n, :, h, w]`` specifies :attr:`input` pixel locations ``x`` and ``y``,
    which are used to interpolate the output value ``output[n, :, h, w]``.
    In the case of 5D inputs, ``grid[n, :, d, h, w]`` specifies the
    ``x``, ``y``, ``z`` pixel locations for interpolating
    ``output[n, :, d, h, w]``. :attr:`mode` argument specifies ``nearest`` or
    ``bilinear`` interpolation method to sample the input pixels.

    :attr:`grid` specifies the sampling pixel locations normalized by the
    :attr:`input` spatial dimensions. Therefore, it should have most values in
    the range of ``[-10, 10]``. For example, values ``x = -10, y = -10`` is the
    left-top pixel of :attr:`input`, and values  ``x = 10, y = 10`` is the
    right-bottom pixel of :attr:`input`.

    If :attr:`grid` has values outside the range of ``[-10, 10]``, the corresponding
    outputs are handled as defined by :attr:`padding_mode`. Options are

        * ``padding_mode="zeros"``: use ``0`` for out-of-bound grid locations,
        * ``padding_mode="border"``: use border values for out-of-bound grid locations,
        * ``padding_mode="reflection"``: use values at locations reflected by
          the border for out-of-bound grid locations. For location far away
          from the border, it will keep being reflected until becoming in bound,
          e.g., (normalized) pixel location ``x = -35`` reflects by border ``-10``
          and becomes ``x' = 15``, then reflects by border ``10`` and becomes
          ``x'' = 5``.

    Args:
        input (Tensor): input of shape :math:`(N, C, H_\text{in}, W_\text{in})` (4-D case)
                        or :math:`(N, C, D_\text{in}, H_\text{in}, W_\text{in})` (5-D case)
        grid (Tensor): flow-field of shape :math:`(N, 2, H_\text{out}, W_\text{out})` (4-D case)
                       or :math:`(N, 3, D_\text{out}, H_\text{out}, W_\text{out})` (5-D case)
        interpolation_mode (str): interpolation mode to calculate output values
            ``'bilinear'`` | ``'nearest'`` | ``'bicubic'``. Default: ``'bilinear'``
            When ``mode='bilinear'`` and the input is 5-D, the interpolation mode
            used internally will actually be trilinear. However, when the input is 4-D,
            the interpolation mode will legitimately be bilinear.
            Note: ``mode='bicubic'`` supports only 4-D input.
        padding_mode (str): padding mode for outside grid values
            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input  as squares rather than points.
            If set to ``True``, the extrema (``-1`` and ``1``) are considered as referring
            to the center points of the input's corner pixels. If set to ``False``, they
            are instead considered as referring to the corner points of the input's corner
            pixels, making the sampling more resolution agnostic.
            This option parallels the ``align_corners`` option in
            :func:`interpolate`, and so whichever option is used here
            should also be used there to resize the input image before grid sampling.
            Default: ``False``

    Returns:
        output (Tensor): output Tensor
    """
    if grid.dim() == 4:
        return _grid_sampler2d2(
            input, grid, _MODE[interpolation_mode], _PAD[padding_mode], align_corners
        )
    elif grid.dim() == 5:
        return _grid_sampler3d2(
            input, grid, _MODE[interpolation_mode], _PAD[padding_mode], align_corners
        )
    else:
        raise RuntimeError
