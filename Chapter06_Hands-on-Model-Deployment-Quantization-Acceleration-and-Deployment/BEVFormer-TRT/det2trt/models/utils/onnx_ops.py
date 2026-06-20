import math
import torch
import numpy as np
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
from torch.onnx.symbolic_helper import parse_args
from det2trt.models.functions import grid_sampler, grid_sampler2


def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    rot = angle * np.pi / 180
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    a = torch.cos(rot - sy) / np.cos(sy)
    b = -torch.cos(rot - sy) * np.tan(sx) / np.cos(sy) - torch.sin(rot)
    c = torch.sin(rot - sy) / np.cos(sy)
    d = -torch.sin(rot - sy) * np.tan(sx) / np.cos(sy) + torch.cos(rot)

    return (
        torch.stack(
            [
                d,
                -b,
                d * (-cx - tx) - b * (-cy - ty) + cx,
                -c,
                a,
                -c * (-cx - tx) + a * (-cy - ty) + cy,
            ]
        )
        / scale
    )


def rotate(img, angle, interpolation=InterpolationMode.NEAREST, center=None, fill=None):
    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")

    if center is not None:
        img_size = [img.shape[-1], img.shape[-2]]
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, img_size)]
        center_f = torch.tensor(center_f, device=img.device)
    else:
        center_f = torch.tensor([0.0, 0.0], device=img.device)

    # due to current incoherence of rotation angle direction between affine and rotate implementations
    # we need to set -angle.
    matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])

    w, h = img.shape[-1], img.shape[-2]
    ow, oh = int(w), int(h)
    theta = matrix.reshape(1, 2, 3)

    # grid will be generated on the same device as theta and img
    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
    x_grid = torch.linspace(
        -ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device
    )
    # base_grid[..., 0].copy_(x_grid)
    base_grid[..., 0] = x_grid.expand(1, oh, ow)
    y_grid = torch.linspace(
        -oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device
    ).unsqueeze_(-1)
    # base_grid[..., 1].copy_(y_grid)
    base_grid[..., 1] = y_grid.expand(1, oh, ow)
    base_grid[..., 2].fill_(1)

    rescaled_theta = theta.transpose(1, 2) / torch.tensor(
        [0.5 * w, 0.5 * h], dtype=img.dtype, device=img.device
    )
    output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
    grid = output_grid.view(1, oh, ow, 2)

    req_dtypes = [
        grid.dtype,
    ]
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)

    if img.shape[0] > 1:
        # Apply same grid to a batch of images
        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])

    # Append a dummy mask for customized fill colors, should be faster than grid_sample() twice
    if fill is not None:
        dummy = torch.ones(
            (img.shape[0], 1, img.shape[2], img.shape[3]),
            dtype=img.dtype,
            device=img.device,
        )
        img = torch.cat((img, dummy), dim=1)

    interpolation = interpolation.value
    img = F.grid_sample(
        img, grid, mode=interpolation, padding_mode="zeros", align_corners=False
    )

    # Fill with required color
    if fill is not None:
        mask = img[:, -1:, :, :]  # N * 1 * H * W
        img = img[:, :-1, :, :]  # N * C * H * W
        mask = mask.expand_as(img)
        len_fill = len(fill) if isinstance(fill, (tuple, list)) else 1
        fill_img = (
            torch.tensor(fill, dtype=img.dtype, device=img.device)
            .view(1, len_fill, 1, 1)
            .expand_as(img)
        )
        if interpolation == "nearest":
            mask = mask < 0.5
            img[mask] = fill_img[mask]
        else:  # 'bilinear'
            img = img * mask + (1.0 - mask) * fill_img

    if need_squeeze:
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


def rotate_p(
    img, angle, interpolation=InterpolationMode.NEAREST, center=None, fill=None
):
    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")

    if center is not None:
        img_size = [img.shape[-1], img.shape[-2]]
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, img_size)]
        center_f = torch.tensor(center_f, device=img.device)
    else:
        center_f = torch.tensor([0.0, 0.0], device=img.device)

    # due to current incoherence of rotation angle direction between affine and rotate implementations
    # we need to set -angle.
    matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])

    w, h = img.shape[-1], img.shape[-2]
    ow, oh = int(w), int(h)
    theta = matrix.reshape(1, 2, 3)

    # grid will be generated on the same device as theta and img
    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
    x_grid = torch.linspace(
        -ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device
    )
    # base_grid[..., 0].copy_(x_grid)
    base_grid[..., 0] = x_grid.expand(1, oh, ow)
    y_grid = torch.linspace(
        -oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device
    ).unsqueeze_(-1)
    # base_grid[..., 1].copy_(y_grid)
    base_grid[..., 1] = y_grid.expand(1, oh, ow)
    base_grid[..., 2].fill_(1)

    rescaled_theta = theta.transpose(1, 2) / torch.tensor(
        [0.5 * w, 0.5 * h], dtype=img.dtype, device=img.device
    )
    output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
    grid = output_grid.view(1, oh, ow, 2)

    req_dtypes = [
        grid.dtype,
    ]
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)

    if img.shape[0] > 1:
        # Apply same grid to a batch of images
        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])

    # Append a dummy mask for customized fill colors, should be faster than grid_sample() twice
    if fill is not None:
        dummy = torch.ones(
            (img.shape[0], 1, img.shape[2], img.shape[3]),
            dtype=img.dtype,
            device=img.device,
        )
        img = torch.cat((img, dummy), dim=1)

    interpolation = interpolation.value
    img = grid_sampler(
        img,
        grid.permute(0, 3, 1, 2) * 10,
        interpolation_mode=interpolation,
        padding_mode="zeros",
        align_corners=False,
    )

    # Fill with required color
    if fill is not None:
        mask = img[:, -1:, :, :]  # N * 1 * H * W
        img = img[:, :-1, :, :]  # N * C * H * W
        mask = mask.expand_as(img)
        len_fill = len(fill) if isinstance(fill, (tuple, list)) else 1
        fill_img = (
            torch.tensor(fill, dtype=img.dtype, device=img.device)
            .view(1, len_fill, 1, 1)
            .expand_as(img)
        )
        if interpolation == "nearest":
            mask = mask < 0.5
            img[mask] = fill_img[mask]
        else:  # 'bilinear'
            img = img * mask + (1.0 - mask) * fill_img

    if need_squeeze:
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


def rotate_p2(
    img, angle, interpolation=InterpolationMode.NEAREST, center=None, fill=None
):
    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")

    if center is not None:
        img_size = [img.shape[-1], img.shape[-2]]
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, img_size)]
        center_f = torch.tensor(center_f, device=img.device)
    else:
        center_f = torch.tensor([0.0, 0.0], device=img.device)

    # due to current incoherence of rotation angle direction between affine and rotate implementations
    # we need to set -angle.
    matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])

    w, h = img.shape[-1], img.shape[-2]
    ow, oh = int(w), int(h)
    theta = matrix.reshape(1, 2, 3)

    # grid will be generated on the same device as theta and img
    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
    x_grid = torch.linspace(
        -ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device
    )
    # base_grid[..., 0].copy_(x_grid)
    base_grid[..., 0] = x_grid.expand(1, oh, ow)
    y_grid = torch.linspace(
        -oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device
    ).unsqueeze_(-1)
    # base_grid[..., 1].copy_(y_grid)
    base_grid[..., 1] = y_grid.expand(1, oh, ow)
    base_grid[..., 2].fill_(1)

    rescaled_theta = theta.transpose(1, 2) / torch.tensor(
        [0.5 * w, 0.5 * h], dtype=img.dtype, device=img.device
    )
    output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
    grid = output_grid.view(1, oh, ow, 2)

    req_dtypes = [
        grid.dtype,
    ]
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)

    if img.shape[0] > 1:
        # Apply same grid to a batch of images
        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])

    # Append a dummy mask for customized fill colors, should be faster than grid_sample() twice
    if fill is not None:
        dummy = torch.ones(
            (img.shape[0], 1, img.shape[2], img.shape[3]),
            dtype=img.dtype,
            device=img.device,
        )
        img = torch.cat((img, dummy), dim=1)

    interpolation = interpolation.value
    img = grid_sampler2(
        img,
        grid.permute(0, 3, 1, 2) * 10,
        interpolation_mode=interpolation,
        padding_mode="zeros",
        align_corners=False,
    )

    # Fill with required color
    if fill is not None:
        mask = img[:, -1:, :, :]  # N * 1 * H * W
        img = img[:, :-1, :, :]  # N * C * H * W
        mask = mask.expand_as(img)
        len_fill = len(fill) if isinstance(fill, (tuple, list)) else 1
        fill_img = (
            torch.tensor(fill, dtype=img.dtype, device=img.device)
            .view(1, len_fill, 1, 1)
            .expand_as(img)
        )
        if interpolation == "nearest":
            mask = mask < 0.5
            img[mask] = fill_img[mask]
        else:  # 'bilinear'
            img = img * mask + (1.0 - mask) * fill_img

    if need_squeeze:
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


@parse_args("v", "v", "i", "i", "i")
def grid_sampler_sym(g, input, grid, interpolation_mode, padding_mode, align_corners):
    return g.op(
        "mmdeploy::grid_sampler",
        input,
        grid,
        interpolation_mode_i=interpolation_mode,
        padding_mode_i=padding_mode,
        align_corners_i=align_corners,
    )


torch.onnx.register_custom_op_symbolic("aten::grid_sampler", grid_sampler_sym, 13)
