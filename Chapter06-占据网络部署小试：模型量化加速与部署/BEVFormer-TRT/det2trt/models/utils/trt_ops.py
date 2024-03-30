import torch.nn.functional as F


def multi_scale_deformable_attn_pytorch(
    value,
    value_spatial_shapes,
    sampling_locations,
    attention_weights,
    num_heads,
    embed_dims,
    num_levels,
    num_points,
    num_queries,
    bs,
):
    """CPU version of multi-scale deformable attention.

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """
    if not isinstance(value, list):
        assert value_spatial_shapes.size(0) == 1
        value = [value]

    embed_dims = embed_dims // num_heads

    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    sampling_grids = 2 * sampling_locations - 1
    output = 0
    for level, value_l_ in enumerate(value):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        H_, W_ = value_spatial_shapes[level]
        value_l_ = (
            value_l_.flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, H_, W_)
        )
        sampling_grids_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)

        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grids_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        sampling_value_l_ *= attention_weights[
            ..., level * num_points : (level + 1) * num_points
        ]
        output += sampling_value_l_.sum(-1)

    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)

    # output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
    #           attention_weights).sum(-1).view(bs, num_heads * embed_dims,
    #                                           num_queries)
    output = output.view(bs, num_heads * embed_dims, num_queries)

    return output.transpose(1, 2).contiguous()
