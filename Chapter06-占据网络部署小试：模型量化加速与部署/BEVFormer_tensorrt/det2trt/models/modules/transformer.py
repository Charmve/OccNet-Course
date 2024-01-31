import torch
import torch.nn as nn
import numpy as np
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from third_party.bev_mmdet3d import PerceptionTransformer
from ..utils import rotate, LINEAR_LAYERS
from ..functions import rotate as rotate_trt, rotate2 as rotate_trt2


@TRANSFORMER.register_module()
class PerceptionTransformerTRT(PerceptionTransformer):
    def __init__(
        self,
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        encoder=None,
        decoder=None,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
        linear_cfg=None,
        **kwargs
    ):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals

        if linear_cfg is None:
            linear_cfg = dict(type="Linear")
        self.linear = LINEAR_LAYERS.get(linear_cfg["type"])

        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = self.linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            self.linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            self.linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module("norm", nn.LayerNorm(self.embed_dims))

    def get_bev_features_trt(
        self,
        mlvl_feats,
        bev_queries,
        bev_h,
        bev_w,
        can_bus,
        lidar2img,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        image_shape=None,
        use_prev_bev=None,
    ):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = can_bus[0:1]
        delta_y = can_bus[1:2]
        ego_angle = can_bus[-2:-1] / np.pi * 180

        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = torch.sqrt(delta_x ** 2 + delta_y ** 2)
        # translation_angle = torch.atan2(delta_y, delta_x) / np.pi * 180
        translation_angle = (
            (
                torch.atan(delta_y / (delta_x + 1e-8))
                + ((1 - torch.sign(delta_x)) / 2) * torch.sign(delta_y) * np.pi
            )
            / np.pi
            * 180
        )
        bev_angle = ego_angle - translation_angle
        shift_y = (
            translation_length
            * torch.cos(bev_angle / 180 * np.pi)
            / grid_length_y
            / bev_h
        )
        shift_x = (
            translation_length
            * torch.sin(bev_angle / 180 * np.pi)
            / grid_length_x
            / bev_w
        )
        shift_y = shift_y * int(self.use_shift)
        shift_x = shift_x * int(self.use_shift)
        shift = torch.stack([shift_x, shift_y]).permute(1, 0)

        if self.rotate_prev_bev:
            for i in range(bs):
                rotation_angle = can_bus[-1]
                tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                tmp_prev_bev = rotate(
                    tmp_prev_bev, rotation_angle, center=self.rotate_center
                )
                tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                    bev_h * bev_w, 1, -1
                )
                prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = can_bus.unsqueeze(0)
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * int(self.use_can_bus)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        for i in range(len(feat_flatten)):
            feat_flatten[i] = feat_flatten[i].permute(0, 2, 1, 3)

        bev_embed = self.encoder.forward_trt(
            bev_queries,
            feat_flatten,
            feat_flatten,
            lidar2img=lidar2img,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            image_shape=image_shape,
            use_prev_bev=use_prev_bev,
        )

        return bev_embed

    def forward_trt(
        self,
        mlvl_feats,
        bev_queries,
        object_query_embed,
        bev_h,
        bev_w,
        can_bus,
        lidar2img,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        reg_branches=None,
        cls_branches=None,
        prev_bev=None,
        image_shape=None,
        use_prev_bev=None,
    ):

        bev_embed = self.get_bev_features_trt(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            can_bus,
            lidar2img,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            image_shape=image_shape,
            use_prev_bev=use_prev_bev,
        )
        # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
        )

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out


@TRANSFORMER.register_module()
class PerceptionTransformerTRTP(PerceptionTransformerTRT):
    def __init__(self, *args, **kwargs):
        super(PerceptionTransformerTRTP, self).__init__(*args, **kwargs)
        self.rotate = rotate_trt

    def get_bev_features_trt(
        self,
        mlvl_feats,
        bev_queries,
        bev_h,
        bev_w,
        can_bus,
        lidar2img,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        image_shape=None,
        use_prev_bev=None,
    ):
        bev_queries = bev_queries.unsqueeze(1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = can_bus[0:1]
        delta_y = can_bus[1:2]
        ego_angle = can_bus[-2:-1] / np.pi * 180

        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = torch.sqrt(delta_x ** 2 + delta_y ** 2)
        # translation_angle = torch.atan2(delta_y, delta_x) / np.pi * 180
        translation_angle = (
            (
                torch.atan(delta_y / (delta_x + 1e-8))
                + ((1 - torch.sign(delta_x)) / 2) * torch.sign(delta_y) * np.pi
            )
            / np.pi
            * 180
        )
        bev_angle = ego_angle - translation_angle
        shift_y = (
            translation_length
            * torch.cos(bev_angle / 180 * np.pi)
            / grid_length_y
            / bev_h
        )
        shift_x = (
            translation_length
            * torch.sin(bev_angle / 180 * np.pi)
            / grid_length_x
            / bev_w
        )
        shift_y = shift_y * int(self.use_shift)
        shift_x = shift_x * int(self.use_shift)
        shift = torch.cat([shift_x, shift_y]).unsqueeze(0)

        if self.rotate_prev_bev:
            rotation_angle = can_bus[-1]
            prev_bev = self.rotate(
                prev_bev.view(bev_h, bev_w, -1).permute(2, 0, 1),
                rotation_angle,
                center=prev_bev.new_tensor(self.rotate_center),
            )
            prev_bev = prev_bev.permute(1, 2, 0).reshape(bev_h * bev_w, 1, -1)

        # add can bus signals
        can_bus = can_bus.view(1, -1)
        can_bus = self.can_bus_mlp(can_bus).view(1, 1, -1)
        bev_queries = bev_queries + can_bus * int(self.use_can_bus)

        feat_flatten = mlvl_feats[0].new_zeros(
            6, 1, 0, self.embed_dims, dtype=torch.float32
        )
        spatial_shapes = mlvl_feats[0].new_zeros(len(mlvl_feats), 2, dtype=torch.long)
        for lvl, feat in enumerate(mlvl_feats):
            _, _, _, h, w = feat.shape
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds.view(6, 1, 1, self.embed_dims)
            feat = feat + self.level_embeds[lvl].view(1, 1, 1, -1)
            spatial_shapes[lvl, 0] = int(h)
            spatial_shapes[lvl, 1] = int(w)
            feat_flatten = torch.cat([feat_flatten, feat], dim=2)
        feat_flatten = feat_flatten.view(6, -1, 1, self.embed_dims)

        bev_embed = self.encoder.forward_trt(
            bev_queries,
            feat_flatten,
            feat_flatten,
            lidar2img=lidar2img,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=None,
            prev_bev=prev_bev,
            shift=shift,
            image_shape=image_shape,
            use_prev_bev=use_prev_bev,
        )

        return bev_embed

    def forward_trt(
        self,
        mlvl_feats,
        bev_queries,
        object_query_embed,
        bev_h,
        bev_w,
        can_bus,
        lidar2img,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        reg_branches=None,
        cls_branches=None,
        prev_bev=None,
        image_shape=None,
        use_prev_bev=None,
    ):

        bev_embed = self.get_bev_features_trt(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            can_bus,
            lidar2img,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            image_shape=image_shape,
            use_prev_bev=use_prev_bev,
        )
        # bev_embed shape: bs, bev_h*bev_w, embed_dims

        query_pos, query = torch.split(
            object_query_embed.unsqueeze(1), self.embed_dims, dim=2
        )
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        bev_embed = bev_embed.view(-1, 1, self.embed_dims)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points.view(1, 900, 3),
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
        )

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out


@TRANSFORMER.register_module()
class PerceptionTransformerTRTP2(PerceptionTransformerTRTP):
    def __init__(self, *args, **kwargs):
        super(PerceptionTransformerTRTP2, self).__init__(*args, **kwargs)
        self.rotate = rotate_trt2
