import torch
from torch.autograd import Function
from mmcv.utils import ext_loader

ext_module = ext_loader.load_ext(
    "_ext", ["ms_deform_attn_backward", "ms_deform_attn_forward"]
)


class _MultiScaleDeformableAttnFunction(Function):
    @staticmethod
    def symbolic(
        g,
        value,
        value_spatial_shapes,
        reference_points,
        sampling_offsets,
        attention_weights,
    ):
        return g.op(
            "MultiScaleDeformableAttnTRT",
            value,
            value_spatial_shapes,
            reference_points,
            sampling_offsets,
            attention_weights,
        )

    @staticmethod
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        reference_points,
        sampling_offsets,
        attention_weights,
    ):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, mum_heads, embed_dims//num_heads, num_keys)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            reference_points (Tensor): The reference points.
            sampling_offsets (Tensor): The offset of sampling points,
                has shape
                (bs, num_heads, num_queries, num_levels*num_points*2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs, num_heads, num_queries, num_levels*num_points).

        Returns:
            Tensor: has shape (bs, embed_dims, num_queries)
        """
        num_heads, channel = value.shape[2:]
        num_level = value_spatial_shapes.shape[0]
        bs, num_queries = reference_points.shape[:2]

        points_per_group = torch.div(
            reference_points.shape[-1], 2, rounding_mode="floor"
        )
        sampling_offsets = sampling_offsets.view(
            bs, num_queries, num_heads, num_level, -1, points_per_group, 2,
        )
        dim = sampling_offsets.shape[4] * num_level * 2 * points_per_group
        offset_normalizer = torch.stack(
            [value_spatial_shapes[..., 1], value_spatial_shapes[..., 0]], -1
        )
        sampling_locations = reference_points.view(
            bs, num_queries, 1, 1, 1, -1, 2
        ) + sampling_offsets / offset_normalizer.view(1, 1, 1, -1, 1, 1, 2)
        sampling_locations = sampling_locations.view(
            bs,
            num_queries,
            num_heads,
            num_level,
            torch.div(dim, num_level * 2, rounding_mode="floor"),
            2,
        )
        attention_weights = attention_weights.view(
            -1, num_level * torch.div(dim, num_level * 2, rounding_mode="floor")
        ).softmax(-1)
        attention_weights = attention_weights.view(
            bs,
            num_queries,
            num_heads,
            num_level,
            torch.div(dim, num_level * 2, rounding_mode="floor"),
        )
        im2col_step = value.shape[0]
        ctx.im2col_step = im2col_step

        ctx.fp16 = False
        if value.dtype == torch.float16:
            ctx.fp16 = True
            value = value.float()
            sampling_locations = sampling_locations.float()
            attention_weights = attention_weights.float()

        value_level_start_index = torch.zeros_like(value_spatial_shapes[:, 0])
        value_level_start_index[1:] = torch.cumsum(
            value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1], dim=0
        )[:-1]

        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step,
        ).view(bs, num_queries, num_heads, channel)
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output.half() if ctx.fp16 else output


class _MultiScaleDeformableAttnFunction2(_MultiScaleDeformableAttnFunction):
    @staticmethod
    def symbolic(
        g,
        value,
        value_spatial_shapes,
        reference_points,
        sampling_offsets,
        attention_weights,
    ):
        return g.op(
            "MultiScaleDeformableAttnTRT2",
            value,
            value_spatial_shapes,
            reference_points,
            sampling_offsets,
            attention_weights,
        )


_multi_scale_deformable_attn_gpu = _MultiScaleDeformableAttnFunction.apply
_multi_scale_deformable_attn_gpu2 = _MultiScaleDeformableAttnFunction2.apply


def multi_scale_deformable_attn(
    value, value_spatial_shapes, reference_points, sampling_offsets, attention_weights
):
    """Multi-scale deformable attention.

    Support TensorRT plugin MultiScaleDeformableAttnTRT: FP32 and FP16(nv_half).

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        reference_points (Tensor): The reference points.
        sampling_offsets (Tensor): The offset of sampling points,
            has shape
            (bs, num_heads, num_queries, num_levels*num_points*2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points).

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """
    assert value.is_cuda
    return _multi_scale_deformable_attn_gpu(
        value,
        value_spatial_shapes,
        reference_points,
        sampling_offsets,
        attention_weights,
    )


def multi_scale_deformable_attn2(
    value, value_spatial_shapes, reference_points, sampling_offsets, attention_weights
):
    """Multi-scale deformable attention.

    Support TensorRT plugin MultiScaleDeformableAttnTRT2: FP32 and FP16(nv_half2).

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        reference_points (Tensor): The reference points.
        sampling_offsets (Tensor): The offset of sampling points,
            has shape
            (bs, num_heads, num_queries, num_levels*num_points*2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points).

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """
    assert value.is_cuda
    return _multi_scale_deformable_attn_gpu2(
        value,
        value_spatial_shapes,
        reference_points,
        sampling_offsets,
        attention_weights,
    )
