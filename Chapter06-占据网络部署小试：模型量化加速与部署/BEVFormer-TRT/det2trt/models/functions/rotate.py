import numpy as np
import torch
from torch.autograd import Function


class _Rotate(Function):
    @staticmethod
    def symbolic(g, img, angle, center, interpolation):
        return g.op("RotateTRT", img, angle, center, interpolation_i=interpolation)

    @staticmethod
    def forward(ctx, img, angle, center, interpolation):
        assert img.ndim == 3
        oh, ow = img.shape[-2:]
        cx = center[0] - center[0].new_tensor(ow * 0.5)
        cy = center[1] - center[1].new_tensor(oh * 0.5)

        angle = -angle * np.pi / 180
        theta = torch.stack(
            [
                torch.cos(angle),
                torch.sin(angle),
                -cx * torch.cos(angle) - cy * torch.sin(angle) + cx,
                -torch.sin(angle),
                torch.cos(angle),
                cx * torch.sin(angle) - cy * torch.cos(angle) + cy,
            ]
        ).view(1, 2, 3)

        # grid will be generated on the same device as theta and img
        d = 0.5
        base_grid = torch.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
        x_grid = torch.linspace(
            -ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device
        )
        base_grid[..., 0] = x_grid.expand(1, oh, ow)
        y_grid = torch.linspace(
            -oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device
        ).unsqueeze_(-1)
        base_grid[..., 1] = y_grid.expand(1, oh, ow)
        base_grid[..., 2].fill_(1)

        rescaled_theta = 2 * theta.transpose(1, 2)
        rescaled_theta[..., 0] /= ow
        rescaled_theta[..., 1] /= oh

        output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
        grid = output_grid.view(1, oh, ow, 2)

        req_dtypes = [
            grid.dtype,
        ]
        # make image NCHW
        img = img.unsqueeze(dim=0)

        out_dtype = img.dtype
        need_cast = False
        if out_dtype not in req_dtypes:
            need_cast = True
            req_dtype = req_dtypes[0]
            img = img.to(req_dtype)

        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])
        img = torch.grid_sampler(img, grid, interpolation, 0, False)

        img = img.squeeze(dim=0)

        if need_cast:
            if out_dtype in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                # it is better to round before cast
                img = torch.round(img)
            img = img.to(out_dtype)

        return img

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class _Rotate2(_Rotate):
    @staticmethod
    def symbolic(g, img, angle, center, interpolation):
        return g.op("RotateTRT2", img, angle, center, interpolation_i=interpolation)


_rotate = _Rotate.apply
_rotate2 = _Rotate2.apply

_MODE = {"bilinear": 0, "nearest": 1}


def rotate(img, angle, center, interpolation="nearest"):
    """Rotate the image by angle.

    Support TensorRT plugin RotateTRT: FP32 and FP16(nv_half).

    Args:
        img (Tensor): image to be rotated.
        angle (Tensor): rotation angle value in degrees, counter-clockwise.
        center (Tensor): Optional center of rotation.
        interpolation (str): interpolation mode to calculate output values
            ``'bilinear'`` | ``'nearest'``. Default: ``'nearest'``
    Returns:
        Tensor: Rotated image.
    """
    if torch.onnx.is_in_onnx_export():
        return _rotate(img, angle, center, _MODE[interpolation])
    return _Rotate.forward(None, img, angle, center, _MODE[interpolation])


def rotate2(img, angle, center, interpolation="nearest"):
    """Rotate the image by angle.

    Support TensorRT plugin RotateTRT: FP32 and FP16(nv_half2).

    Args:
        img (Tensor): image to be rotated.
        angle (Tensor): rotation angle value in degrees, counter-clockwise.
        center (Tensor): Optional center of rotation.
        interpolation (str): interpolation mode to calculate output values
            ``'bilinear'`` | ``'nearest'``. Default: ``'nearest'``
    Returns:
        Tensor: Rotated image.
    """
    if torch.onnx.is_in_onnx_export():
        return _rotate2(img, angle, center, _MODE[interpolation])
    return _Rotate2.forward(None, img, angle, center, _MODE[interpolation])
