
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            # 0.5~7.5中间均匀采样4个点 (4,)-->(4, 1, 1)-->(4, 50, 50) 归一化
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            # 0.5~49.5中间均匀采样50个点 (50,)-->(1, 1, 50)-->(4, 50, 50) 归一化
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            # 0.5~49.5中间均匀采样50个点 (50,)-->(1, 50, 1)-->(4, 50, 50) 归一化
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1) # (4, 50, 50, 3)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1) # (4, 50, 50, 3) -> (4, 3, 50, 50) -> (4, 3, 2500) -> (4, 2500, 3)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1) # (4, 2500, 3) -> (1, 4, 2500, 3) -> (1, 4, 2500, 3)
            return ref_3d # (1, 4, 2500, 3)

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            ) # (50, 50)
            ref_y = ref_y.reshape(-1)[None] / H # (50, 50) -> (2500, ) ->  (1, 2500)
            ref_x = ref_x.reshape(-1)[None] / W # (50, 50) -> (2500, ) ->  (1, 2500)
            ref_2d = torch.stack((ref_x, ref_y), -1) # (1, 2500, 2)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2) # (1, 2500, 2) -> (1, 2500, 1, 2)
            return ref_2d # (1, 2500, 1, 2)

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):
        # reference_points:(1, 4, 2500, 3), lidar2img: lidar -> ego -> camera -> 内参 -> img
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img) 
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4), (1, 6, 4, 4)
        reference_points = reference_points.clone()

        # 将归一化参考点恢复到lidar系下的实际位置
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]
        # ref_3d: (1, 4, 2500, 3)

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1) # cat: (1, 4, 2500, 1), ref_points: (1, 4, 2500, 4)

        reference_points = reference_points.permute(1, 0, 2, 3) # (1, 4, 2500, 4) -> (4, 1, 2500, 4)
        D, B, num_query = reference_points.size()[:3] # 4, 1, 2500
        num_cam = lidar2img.size(1) # 6

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1) # (4, 1, 2500, 4) -> (4, 1, 1, 2500, 4) -> (4, 1, 6, 2500, 4) -> (4, 1, 6, 2500, 4, 1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1) # (1, 6, 4, 4) -> (1, 1, 6, 1, 4, 4) -> (4, 1, 6, 2500, 4, 4)
        
        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1) # 矩阵乘法, (4, 1, 6, 2500, 4)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps) # (4, 1, 6, 2500, 1)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps) # (4, 1, 6, 2500, 2)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        # 用于评判某一 三维坐标点 是否落在了 二维坐标平面上 比例区间(0,1), (4, 1, 6, 2500, 1)
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask) # nan->number, default 0
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4) # (4, 1, 6, 2500, 2) -> (6, 1, 2500, 4, 2)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1) # (4, 1, 6, 2500, 1) -> (6, 1, 2500, 4, 1) -> (6, 1, 2500, 4)

        return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(self,
                bev_query, 
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query # (2500, 1, 256)
        intermediate = []

        # 3d参考点 ref_3d: (1, 4, 2500, 3)
        # self.pc_range[5]-self.pc_range[2] 8
        # self.num_points_in_pillar 4
        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
        # (1, 4, 2500, 3)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)
        # (1, 2500, 1, 2) 

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas']) # ref_3d -> ref_3d_cam, (6, 1, 2500, 4, 2), (6, 1, 2500, 4)

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d  # .clone() # 获取BEV平面上的2D参考点
        shift_ref_2d += shift[:, None, None, :] # 在原始参考点的基础上+偏移量 (1, 2500, 1, 2) 

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2) # (2500, 1, 256) -> (1, 2500, 256)
        bev_pos = bev_pos.permute(1, 0, 2) # (2500, 1, 256) -> (1, 2500, 256)
        bs, len_bev, num_bev_level, _ = ref_2d.shape # 1, 2500, 1, 2
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2) # (2500, 1, 256) -> (1, 2500, 256)
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1) # (2, 2500, 256)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2) # (2, 2500, 1, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2) # prev_bev=None

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query, # (1, 2500, 256)
                key, # (6, 375, 1, 256)
                value, # (6, 375, 1, 256)
                *args,
                bev_pos=bev_pos, # (1, 2500, 256)
                ref_2d=hybird_ref_2d, # (2, 2500, 1, 2)
                ref_3d=ref_3d, # (1, 4, 2500, 3)
                bev_h=bev_h, # 50
                bev_w=bev_w, # 50
                spatial_shapes=spatial_shapes, # [[15, 25]]
                level_start_index=level_start_index, # [0]
                reference_points_cam=reference_points_cam, # (6, 1, 2500, 4, 2)
                bev_mask=bev_mask, # (6, 1, 2500, 4)
                prev_bev=prev_bev, # None or (2, 2500, 256)
                **kwargs)

            bev_query = output # (1, 2500, 256)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
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
                **kwargs):
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
        identity = query # (1, 2500, 256)
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'
        # attn_masks: [None, None]
        # operation_order: ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query, # (1, 2500, 256)
                    prev_bev, # None或(2, 2500, 256)
                    prev_bev, # None或(2, 2500, 256)
                    identity if self.pre_norm else None, # None
                    query_pos=bev_pos, # (1, 2500, 256)
                    key_pos=bev_pos, # (1, 2500, 256)
                    attn_mask=attn_masks[attn_index], # None
                    key_padding_mask=query_key_padding_mask, # None
                    reference_points=ref_2d, # (2, 2500, 1, 2)
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device), # [[50, 50]]
                    level_start_index=torch.tensor([0], device=query.device), # [0]
                    **kwargs)
                attn_index += 1
                identity = query # (1, 2500, 256)

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query, # (1, 2500, 256)
                    key, # (6, 375, 1, 256)
                    value, # (6, 375, 1, 256)
                    identity if self.pre_norm else None, # None
                    query_pos=query_pos, # (1, 2500, 256)
                    key_pos=key_pos, # (1, 2500, 256)
                    reference_points=ref_3d, # (1, 4, 2500, 3)
                    reference_points_cam=reference_points_cam, # (6, 1, 2500, 4, 2)
                    mask=mask, # (6, 1, 2500, 4)
                    attn_mask=attn_masks[attn_index], # None
                    key_padding_mask=key_padding_mask, # None
                    spatial_shapes=spatial_shapes, # [[15, 25]]
                    level_start_index=level_start_index, # [0]
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
