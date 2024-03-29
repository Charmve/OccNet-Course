import torch
import copy
import warnings
import numpy as np
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER_LAYER
from third_party.bev_mmdet3d import BEVFormerEncoder, BEVFormerLayer


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoderTRT(BEVFormerEncoder):
    def __init__(self, *args, **kwargs):
        super(BEVFormerEncoderTRT, self).__init__(*args, **kwargs)

    def point_sampling_trt(self, reference_points, pc_range, lidar2img, image_shape):
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = (
            reference_points.view(D, B, 1, num_query, 4)
            .repeat(1, 1, num_cam, 1, 1)
            .unsqueeze(-1)
        )

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(
            D, 1, 1, num_query, 1, 1
        )

        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32), reference_points.to(torch.float32)
        ).squeeze(-1)
        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        # reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        #     reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.max(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )

        reference_points_cam[..., 0] /= image_shape[1]
        reference_points_cam[..., 1] /= image_shape[0]

        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )

        bev_mask = torch.nan_to_num(bev_mask).int()

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        # bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4)[..., 0]
        return reference_points_cam, bev_mask

    def forward_trt(
        self,
        bev_query,
        key,
        value,
        *args,
        lidar2img=None,
        bev_h=None,
        bev_w=None,
        bev_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        prev_bev=None,
        shift=0.0,
        image_shape=None,
        use_prev_bev=None,
    ):
        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim="3d",
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )
        ref_2d = self.get_reference_points(
            bev_h,
            bev_w,
            dim="2d",
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )

        reference_points_cam, bev_mask = self.point_sampling_trt(
            ref_3d, self.pc_range, lidar2img, image_shape
        )

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d = shift_ref_2d + shift[:, None, None, :] * use_prev_bev

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape

        prev_bev = prev_bev.permute(1, 0, 2)
        prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(bs * 2, len_bev, -1)
        hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
            bs * 2, len_bev, num_bev_level, 2
        )

        for lid, layer in enumerate(self.layers):
            output = layer.forward_trt(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                use_prev_bev=use_prev_bev,
            )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoderTRTP(BEVFormerEncoderTRT):
    def __init__(self, *args, **kwargs):
        super(BEVFormerEncoderTRTP, self).__init__(*args, **kwargs)

    @staticmethod
    def get_reference_points_3d(
        H, W, Z=8, num_points_in_pillar=4, bs=1, device="cuda", dtype=torch.float,
    ):
        zs = (
            torch.linspace(
                0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device
            )
            .view(-1, 1, 1)
            .repeat(1, H, W)
            / Z
        )
        xs = (
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            .view(1, 1, W)
            .repeat(num_points_in_pillar, H, 1)
            / W
        )
        ys = (
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
            .view(1, H, 1)
            .repeat(num_points_in_pillar, 1, W)
            / H
        )

        ref_3d = torch.stack((xs, ys, zs), -1).view(1, 4, -1, 3)
        return ref_3d

    def point_sampling_trt(self, reference_points, pc_range, lidar2img, image_shape):

        reference_points = reference_points * torch.tensor(
            [
                pc_range[3] - pc_range[0],
                pc_range[4] - pc_range[1],
                pc_range[5] - pc_range[2],
            ],
            dtype=reference_points.dtype,
            device=reference_points.device,
        ).view(1, 1, 1, 3) + torch.tensor(
            pc_range[:3], dtype=reference_points.dtype, device=reference_points.device
        )

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        reference_points = reference_points.view(
            self.num_points_in_pillar, 1, 1, -1, 4, 1
        )
        lidar2img = lidar2img.view(1, 1, 6, 1, 4, 4)
        reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)

        eps = 1e-5
        mask_zeros = reference_points_cam.new_zeros(
            self.num_points_in_pillar,
            1,
            6,
            int(reference_points_cam.shape[3]),
            1,
            dtype=torch.float32,
        )
        mask_ones = mask_zeros + 1
        bev_mask = torch.where(
            reference_points_cam[..., 2:3] > eps, mask_ones, mask_zeros
        )
        reference_points_cam = reference_points_cam[..., 0:2] / torch.max(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )

        reference_points_cam[..., 0] /= image_shape[1]
        reference_points_cam[..., 1] /= image_shape[0]

        bev_mask *= torch.where(
            reference_points_cam[..., 1:2] > 0.0, mask_ones, mask_zeros
        )
        bev_mask *= torch.where(
            reference_points_cam[..., 1:2] < 1.0, mask_ones, mask_zeros
        )
        bev_mask *= torch.where(
            reference_points_cam[..., 0:1] < 1.0, mask_ones, mask_zeros
        )
        bev_mask *= torch.where(
            reference_points_cam[..., 0:1] > 0.0, mask_ones, mask_zeros
        )

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = (1 - (1 - bev_mask).prod(0)).view(6, -1, 1)
        # bev_mask = bev_mask.prod(0).view(6, -1, 1)
        bev_mask = bev_mask / torch.clamp(bev_mask.sum(0, keepdims=True), min=1e-4)
        return reference_points_cam, bev_mask

    def forward_trt(
        self,
        bev_query,
        key,
        value,
        *args,
        lidar2img=None,
        bev_h=None,
        bev_w=None,
        bev_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        prev_bev=None,
        shift=None,
        image_shape=None,
        use_prev_bev=None,
    ):
        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points_3d(
            bev_h,
            bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            bs=1,
            device=bev_query.device,
            dtype=bev_query.dtype,
        )
        ref_2d = ref_3d[0, 0, :, :2].view(1, -1, 1, 2).clone()

        reference_points_cam, bev_mask = self.point_sampling_trt(
            ref_3d, self.pc_range, lidar2img, image_shape
        )

        shift_ref_2d = ref_2d.clone()
        shift_ref_2d = shift_ref_2d + shift.view(1, 1, 1, 2) * use_prev_bev

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.view(1, -1, self.embed_dims)
        bev_pos = bev_pos.view(1, -1, self.embed_dims)
        prev_bev = prev_bev.view(1, -1, self.embed_dims)

        prev_bev = torch.cat([prev_bev, bev_query], dim=0)
        hybird_ref_2d = torch.cat([shift_ref_2d, ref_2d], dim=0)

        for lid, layer in enumerate(self.layers):
            output = layer.forward_trt(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                use_prev_bev=use_prev_bev,
            )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayerTRT(BEVFormerLayer):
    def __init__(
        self,
        attn_cfgs,
        # feedforward_channels,
        # ffn_dropout=0.0,
        operation_order=None,
        act_cfg=dict(type="ReLU", inplace=True),
        norm_cfg=dict(type="LN"),
        # ffn_num_fcs=2,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        **kwargs,
    ):

        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            # feedforward_channels=feedforward_channels,
            # ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            # ffn_num_fcs=ffn_num_fcs,
            ffn_cfgs=ffn_cfgs,
            **kwargs,
        )
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])

    def forward_trt(
        self,
        query,
        key=None,
        value=None,
        bev_pos=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        ref_2d=None,
        ref_3d=None,
        bev_h=None,
        bev_w=None,
        reference_points_cam=None,
        mask=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        use_prev_bev=None,
        **kwargs,
    ):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(
                f"Use same attn_mask in all attentions in "
                f"{self.__class__.__name__} "
            )
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                q_bs, q_len_bev, q_c = query.shape
                prev_bev = use_prev_bev * prev_bev + (1 - use_prev_bev) * torch.stack(
                    [query, query], 1
                ).reshape(q_bs * 2, q_len_bev, q_c)
                query = self.attentions[attn_index].forward_trt(
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == "cross_attn":
                query = self.attentions[attn_index].forward_trt(
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayerTRTP(BEVFormerLayerTRT):
    def __init__(
        self, *args, **kwargs,
    ):
        super(BEVFormerLayerTRTP, self).__init__(*args, **kwargs)

    def forward_trt(
        self,
        query,
        key=None,
        value=None,
        bev_pos=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        ref_2d=None,
        ref_3d=None,
        bev_h=None,
        bev_w=None,
        reference_points_cam=None,
        mask=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        use_prev_bev=None,
        **kwargs,
    ):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        if attn_masks is None:
            attn_masks = [None] * self.num_attn
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(
                f"Use same attn_mask in all attentions in "
                f"{self.__class__.__name__} "
            )
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                prev_bev = use_prev_bev * prev_bev + (1 - use_prev_bev) * query.repeat(
                    2, 1, 1
                )
                query = self.attentions[attn_index].forward_trt(
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == "cross_attn":
                query = self.attentions[attn_index].forward_trt(
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
