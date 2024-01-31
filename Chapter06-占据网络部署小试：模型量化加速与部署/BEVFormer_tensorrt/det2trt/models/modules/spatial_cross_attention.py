import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import build_attention
from mmcv.runner import force_fp32

from mmcv.utils import ext_loader
from ..utils import LINEAR_LAYERS, multi_scale_deformable_attn_pytorch
from third_party.bev_mmdet3d import (
    SpatialCrossAttention,
    MSDeformableAttention3D,
    MultiScaleDeformableAttnFunction_fp32,
)
from det2trt.models.functions import (
    multi_scale_deformable_attn,
    multi_scale_deformable_attn2,
)

ext_module = ext_loader.load_ext(
    "_ext", ["ms_deform_attn_backward", "ms_deform_attn_forward"]
)


@ATTENTION.register_module()
class SpatialCrossAttentionTRT(SpatialCrossAttention):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(
        self,
        embed_dims=256,
        num_cams=6,
        pc_range=None,
        dropout=0.1,
        init_cfg=None,
        batch_first=False,
        deformable_attention=dict(
            type="MSDeformableAttention3D", embed_dims=256, num_levels=4
        ),
        linear_cfg=None,
        **kwargs,
    ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        if linear_cfg is None:
            linear_cfg = dict(type="Linear")
        linear = LINEAR_LAYERS.get(linear_cfg["type"])

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    @force_fp32(apply_to=("query", "key", "value", "query_pos", "reference_points_cam"))
    def forward_trt(
        self,
        query,
        key,
        value,
        residual=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        reference_points_cam=None,
        bev_mask=None,
        level_start_index=None,
        flag="encoder",
        **kwargs,
    ):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        # if key is None:
        #     key = query
        # if value is None:
        #     value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3)

        indexes = (bev_mask.sum(-1) > 0).permute(1, 0, 2).unsqueeze(-1)

        max_len = bev_mask.shape[2]

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.clone().view(
            bs, self.num_cams, max_len, D, 2
        )

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                queries_rebatch[j, i, :] = query[j, :]

        value = [
            value[i]
            .permute(2, 0, 1, 3)
            .reshape(
                value[i].shape[2] * self.num_cams, value[i].shape[1], self.embed_dims
            )
            for i in range(len(value))
        ]

        queries = self.deformable_attention.forward_trt(
            query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
            # key=key,
            value=value,
            reference_points=reference_points_rebatch.view(
                bs * self.num_cams, max_len, D, 2
            ),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        ).view(bs, self.num_cams, max_len, self.embed_dims)

        slots = (queries * indexes).sum(1)

        count = bev_mask.sum(-1) > 0
        count = count.sum(0)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@ATTENTION.register_module()
class SpatialCrossAttentionTRTP(SpatialCrossAttentionTRT):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(
        self, *args, **kwargs,
    ):
        super(SpatialCrossAttentionTRTP, self).__init__(*args, **kwargs)

    @force_fp32(apply_to=("query", "key", "value", "query_pos", "reference_points_cam"))
    def forward_trt(
        self,
        query,
        key,
        value,
        residual=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        reference_points_cam=None,
        bev_mask=None,
        level_start_index=None,
        flag="encoder",
        **kwargs,
    ):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        query = query.repeat(self.num_cams, 1, 1)
        reference_points_cam = reference_points_cam.view(
            self.num_cams, -1, int(reference_points_cam.size(3)), 2
        )

        value = value.view(self.num_cams, -1, self.embed_dims)

        queries = self.deformable_attention.forward_trt(
            query=query,
            # key=key,
            value=value,
            reference_points=reference_points_cam,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        slots = (queries * bev_mask).sum(0, keepdims=True)
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@ATTENTION.register_module()
class MSDeformableAttention3DTRT(MSDeformableAttention3D):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=8,
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
        linear_cfg=None,
    ):
        super(MSDeformableAttention3D, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        if linear_cfg is None:
            linear_cfg = dict(type="Linear")
        linear = LINEAR_LAYERS.get(linear_cfg["type"])

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
                )
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = linear(embed_dims, embed_dims)

        self.init_weights()

    def forward_trt(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        assert isinstance(value, list)
        # if value is None:
        #     value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = [value[i].permute(1, 0, 2) for i in range(len(value))]

        bs, num_query, _ = query.shape
        # bs, num_value, _ = value.shape
        num_value = 0
        for i in range(len(value)):
            num_value += value[i].shape[1]

        if len(value) == 1:
            value[0] = self.value_proj(value[0]).view(
                bs, value[0].shape[1], self.num_heads, -1
            )
        else:
            value = torch.cat(value, dim=1)
            value = self.value_proj(value)
            value = [
                *torch.split(
                    value,
                    [
                        spatial_shapes[i, 0] * spatial_shapes[i, 1]
                        for i in range(len(spatial_shapes))
                    ],
                    dim=1,
                )
            ]
            value = [
                value[i].view(bs, value[i].shape[1], self.num_heads, -1)
                for i in range(len(value))
            ]

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert key_padding_mask is None

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = (
                sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
            (
                bs,
                num_query,
                num_heads,
                num_levels,
                num_all_points,
                xy,
            ) = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs,
                num_query,
                num_heads,
                num_levels,
                num_all_points // num_Z_anchors,
                num_Z_anchors,
                xy,
            )
            sampling_locations = reference_points + sampling_offsets
            (
                bs,
                num_query,
                num_heads,
                num_levels,
                num_points,
                num_Z_anchors,
                xy,
            ) = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy
            )

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points

        if (
            torch.cuda.is_available()
            and value[0].is_cuda
            and not torch.onnx.is_in_onnx_export()
        ):
            value = torch.cat(value, 1)
            MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:
            if self.num_levels == 1:
                output = multi_scale_deformable_attn_pytorch(
                    value,
                    spatial_shapes,
                    sampling_locations,
                    attention_weights,
                    num_heads=self.num_heads,
                    embed_dims=self.embed_dims,
                    num_levels=self.num_levels,
                    num_points=self.num_points,
                    num_queries=int(num_query),
                    bs=6,
                )
            elif self.num_levels == 4:
                embed_dims_ = self.embed_dims // self.num_heads

                attention_weights_ = attention_weights.transpose(1, 2).reshape(
                    6 * self.num_heads, 1, num_query, self.num_levels * self.num_points
                )

                sampling_grids_ = 2 * sampling_locations - 1

                output = sampling_locations.new_zeros(
                    6 * self.num_heads, embed_dims_, int(num_query)
                )

                H_0, W_0 = spatial_shapes[0]
                value_l_0 = (
                    value[0]
                    .permute(0, 2, 3, 1)
                    .reshape(6 * self.num_heads, embed_dims_, H_0, W_0)
                )
                sampling_grids_l_0 = (
                    sampling_grids_[:, :, :, 0].transpose(1, 2).flatten(0, 1)
                )
                sampling_value_l_0 = F.grid_sample(
                    value_l_0,
                    sampling_grids_l_0,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                sampling_value_l_0 *= attention_weights_[
                    ..., 0 * self.num_points : (0 + 1) * self.num_points
                ]
                output += sampling_value_l_0.sum(-1)

                H_1, W_1 = spatial_shapes[1]
                value_l_1 = (
                    value[1]
                    .permute(0, 2, 3, 1)
                    .reshape(6 * self.num_heads, embed_dims_, H_1, W_1)
                )
                sampling_grids_l_1 = (
                    sampling_grids_[:, :, :, 1].transpose(1, 2).flatten(0, 1)
                )
                sampling_value_l_1 = F.grid_sample(
                    value_l_1,
                    sampling_grids_l_1,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                sampling_value_l_1 *= attention_weights_[
                    ..., 1 * self.num_points : (1 + 1) * self.num_points
                ]
                output += sampling_value_l_1.sum(-1)

                H_2, W_2 = spatial_shapes[2]
                value_l_2 = (
                    value[2]
                    .permute(0, 2, 3, 1)
                    .reshape(6 * self.num_heads, embed_dims_, H_2, W_2)
                )
                sampling_grids_l_2 = (
                    sampling_grids_[:, :, :, 2].transpose(1, 2).flatten(0, 1)
                )
                sampling_value_l_2 = F.grid_sample(
                    value_l_2,
                    sampling_grids_l_2,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                sampling_value_l_2 *= attention_weights_[
                    ..., 2 * self.num_points : (2 + 1) * self.num_points
                ]
                output += sampling_value_l_2.sum(-1)

                H_3, W_3 = spatial_shapes[3]
                value_l_3 = (
                    value[3]
                    .permute(0, 2, 3, 1)
                    .reshape(6 * self.num_heads, embed_dims_, H_3, W_3)
                )
                sampling_grids_l_3 = (
                    sampling_grids_[:, :, :, 3].transpose(1, 2).flatten(0, 1)
                )
                sampling_value_l_3 = F.grid_sample(
                    value_l_3,
                    sampling_grids_l_3,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                sampling_value_l_3 *= attention_weights_[
                    ..., 3 * self.num_points : (3 + 1) * self.num_points
                ]
                output += sampling_value_l_3.sum(-1)

                output = (
                    output.view(6, self.num_heads * embed_dims_, num_query)
                    .transpose(1, 2)
                    .contiguous()
                )
            else:
                raise RuntimeError

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


@ATTENTION.register_module()
class MSDeformableAttention3DTRTP(MSDeformableAttention3DTRT):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, *args, **kwargs):
        super(MSDeformableAttention3DTRTP, self).__init__(*args, **kwargs)
        self.multi_scale_deformable_attn = multi_scale_deformable_attn

    def forward_trt(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        assert self.batch_first

        value = self.value_proj(value).view(
            6, -1, self.num_heads, self.embed_dims // self.num_heads
        )

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == value.shape[1]
        assert key_padding_mask is None

        sampling_offsets = self.sampling_offsets(query)
        attention_weights = self.attention_weights(query)

        reference_points = reference_points.reshape(6, -1, 1, 8)
        sampling_offsets = sampling_offsets.view(
            *sampling_offsets.shape[:2], self.num_heads, -1
        )
        attention_weights = attention_weights.view(
            *attention_weights.shape[:2], self.num_heads, -1
        )
        output = self.multi_scale_deformable_attn(
            value, spatial_shapes, reference_points, sampling_offsets, attention_weights
        ).flatten(2)

        return output


@ATTENTION.register_module()
class MSDeformableAttention3DTRTP2(MSDeformableAttention3DTRTP):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, *args, **kwargs):
        super(MSDeformableAttention3DTRTP2, self).__init__(*args, **kwargs)
        self.multi_scale_deformable_attn = multi_scale_deformable_attn2
