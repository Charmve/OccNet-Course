# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence, build_positional_encoding
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16


@TRANSFORMER.register_module()
class HybridPerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 position=None,  # positional embedding of query point
                 encoder_embed_dims=[256, 128, 64 ,32, 16],
                 feature_map_z=[1, 2, 4, 8, 16],
                 pos_dims=[128, 64, 32, 16, 8],
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 can_bus_in_dataset=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 decoder_on_bev=True,
                 bev_z=16,
                 **kwargs):
        super(HybridPerceptionTransformer, self).__init__(**kwargs)
        
        self.encoder_block_num = len(encoder_embed_dims)
        self.voxel_encoder_num = self.encoder_block_num -1  # the number of voxel encoders
        self.feature_map_z = feature_map_z
        self.encoder_embed_dims = encoder_embed_dims
        self.pos_dims = pos_dims
        self.encoders = nn.ModuleList()
        self.positional_encodings = nn.ModuleList()

        # define the first bev encoder
        self.define_bev_encoder(encoder.bev, position.bev)
        self.define_voxel_encoder(encoder.voxel, position.voxel)

        # 3D detection decoder
        if decoder is not None:  
            self.decoder = build_transformer_layer_sequence(decoder)
        else:
            self.decoder = None

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.can_bus_in_dataset = can_bus_in_dataset
        self.use_cams_embeds = use_cams_embeds
        self.decoder_on_bev = decoder_on_bev
        self.bev_z = bev_z
        self.two_stage_num_proposals = two_stage_num_proposals 
        
        self.init_layers()
        self.rotate_center = rotate_center

    def define_bev_encoder(self, model_config, pos_config):
        self.encoders.append(build_transformer_layer_sequence(model_config))
        self.positional_encodings.append(build_positional_encoding(pos_config))
    
    def define_voxel_encoder(self, model_config, pos_config):
        for i in range(1, self.encoder_block_num):
            # set model config 
            model_config.transformerlayers.attn_cfgs[0].embed_dims = self.encoder_embed_dims[i]
            model_config.transformerlayers.attn_cfgs[1].deformable_attention.embed_dims = self.encoder_embed_dims[i]
            model_config.transformerlayers.attn_cfgs[1].embed_dims = self.encoder_embed_dims[i]
            model_config.transformerlayers.ffn_cfgs.embed_dims = self.encoder_embed_dims[i]
            model_config.transformerlayers.ffn_cfgs.feedforward_channels = self.encoder_embed_dims[i]*2
            model_config.transformerlayers.feedforward_channels = self.encoder_embed_dims[i]*2
            self.encoders.append(build_transformer_layer_sequence(model_config))

            # set pos_config
            pos_config.num_feats = self.pos_dims[i]
            pos_config.z_num_embed = self.feature_map_z[i]
            self.positional_encodings.append(build_positional_encoding(pos_config))

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        if self.use_cams_embeds:
            self.cams_embeds = nn.Parameter(
                torch.Tensor(self.num_cams, self.embed_dims))
        if self.use_can_bus:
            self.can_bus_mlp = nn.Sequential(
                nn.Linear(18, self.embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims // 2, self.embed_dims),
                nn.ReLU(inplace=True),
            )
            if self.can_bus_norm:
                self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
        
        if self.decoder is not None and self.decoder_on_bev:
            voxel2bev = []
            last_feature = self.feature_map_z[-1]*self.encoder_embed_dims[-1]
            mid_num = last_feature
            voxel2bev.append(nn.Linear(last_feature, mid_num))
            voxel2bev.append(nn.LayerNorm(mid_num))
            voxel2bev.append(nn.ReLU(inplace=True))
            voxel2bev.append(nn.Linear(mid_num, self.embed_dims))
            voxel2bev.append(nn.LayerNorm(self.embed_dims))
            voxel2bev.append(nn.ReLU(inplace=True))
            self.voxel2bev = nn.Sequential(*voxel2bev)

        if self.decoder is not None:
            self.reference_points = nn.Linear(self.embed_dims, 3)

        # mid-stage bev->voxe->voxel-> voxel
        self.bev_voxel_transfers = nn.ModuleList()
        for i in range(self.encoder_block_num-1):
            fc1 = self.encoder_embed_dims[i]*self.feature_map_z[i]
            fc2 = self.encoder_embed_dims[i+1]*self.feature_map_z[i+1]
            block = nn.Sequential(
                nn.Linear(fc1, fc2),
                nn.ReLU(inplace=True),
                nn.LayerNorm(fc2),
            )
            self.bev_voxel_transfers.append(block)

        # the transition between the different voxel encoder, use MLP for simplicity
        self.image_feature_maps = nn.ModuleList()
        for i in range(self.encoder_block_num):
            block = nn.Sequential(
                nn.Linear(self.embed_dims, self.encoder_embed_dims[i]),
                nn.ReLU(inplace=True),
                )
            self.image_feature_maps.append(block)
            
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        if self.use_cams_embeds:
            normal_(self.cams_embeds)
        if self.use_can_bus:
            xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
        if self.decoder is not None and self.decoder_on_bev:
            xavier_init(self.voxel2bev, distribution='uniform', bias=0.)
        if self.decoder is not None:
            xavier_init(self.reference_points, distribution='uniform', bias=0.)

        for block in self.bev_voxel_transfers:
            xavier_init(block, distribution='uniform', bias=0.)
        
        for block in self.image_feature_maps:
            xavier_init(block, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_voxel_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_z,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)  # (num_query, bs, embed_dims)
 
        # obtain rotation angle and shift with ego motion
        if self.can_bus_in_dataset:
            delta_x = np.array([each['can_bus'][0]
                            for each in kwargs['img_metas']])
            delta_y = np.array([each['can_bus'][1]
                            for each in kwargs['img_metas']])
            ego_angle = np.array(
                [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
            grid_length_y = grid_length[0]
            grid_length_x = grid_length[1]
            translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
            translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
            bev_angle = ego_angle - translation_angle
            shift_y = translation_length * \
                np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
            shift_x = translation_length * \
                np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        else:
            shift_y=np.array([0]*bs)
            shift_x=np.array([0]*bs)
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0)  # (2, bs) -> (bs, 2)
        # add can bus signals
        if self.use_can_bus:
            can_bus = bev_queries.new_tensor(
                [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
            can_bus = self.can_bus_mlp(can_bus)[None, :, :]  
            bev_queries = bev_queries + can_bus * self.use_can_bus  # (query_num, bs, embed_dims)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_queries.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten_original = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        block_features = []
        for block_index in range(self.encoder_block_num):
            # encoder： BEV -> Voxeli -> Voxelj -> Voxelk
            block_bev_z = self.feature_map_z[block_index]
            # block_embed_dims = self.encoder_embed_dims[block_index]
            if block_bev_z == 1:
                bev_mask = torch.zeros((bs, bev_h, bev_w),
                            device=bev_queries.device).to(bev_queries.dtype)
            else:
                bev_mask = torch.zeros((bs, block_bev_z, bev_h, bev_w),
                            device=bev_queries.device).to(bev_queries.dtype)
            pos = self.positional_encodings[block_index](bev_mask).to(bev_queries.dtype)  # (bs, embed_dims, h, w)
            pos = pos.flatten(2).permute(2, 0, 1)  # (query_num, bs, embed_dims)
            
            feat_flatten = self.image_feature_maps[block_index](feat_flatten_original)
            
            if prev_bev is not None:  # (bs, num_query, embed_dims)
                stage_prev_bev = prev_bev[block_index]
                if block_bev_z == 1:  # 2D BEV
                    if stage_prev_bev.shape[1] == bev_h * bev_w:
                        stage_prev_bev = stage_prev_bev.permute(1, 0, 2)  # (num_query, bs, embed_dims)
                    if self.rotate_prev_bev:
                        for i in range(bs):
                            # num_prev_bev = prev_bev.size(1)
                            rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                            tmp_prev_bev = stage_prev_bev[:, i].reshape(
                                bev_h, bev_w, -1).permute(2, 0, 1)  # (embed_dims, bev_h, bev_w)
                            tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                                center=self.rotate_center)  # TODO: for 3D voxel
                            tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                                bev_h * bev_w, 1, -1)
                            stage_prev_bev[:, i] = tmp_prev_bev[:, 0]
                        
                else:  # 3D Voxel
                    if stage_prev_bev.shape[1] == block_bev_z* bev_h * bev_w:
                        stage_prev_bev = stage_prev_bev.permute(1, 0, 2)  # (num_query, bs, embed_dims)
                    if self.rotate_prev_bev:  # revise for 3D feature map
                        for i in range(bs):
                            rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                            tmp_prev_bev = stage_prev_bev[:, i].reshape(block_bev_z, bev_h, bev_w, -1).permute(3, 0, 1, 2)  # (embed_dims, bev_z, bev_h, bev_w)
                            tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                            tmp_prev_bev = tmp_prev_bev.permute(1, 2, 3, 0).reshape(block_bev_z * bev_h * bev_w, 1, -1)
                            stage_prev_bev[:, i] = tmp_prev_bev[:, 0] 
            else:
                stage_prev_bev = None

            output = self.encoders[block_index](
                bev_queries,
                feat_flatten,
                feat_flatten,
                bev_z=block_bev_z,
                bev_h=bev_h,
                bev_w=bev_w,
                bev_pos=pos,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_bev=stage_prev_bev,
                shift=shift,
                **kwargs
            )
            
            block_features.append(output)
            
            if block_index < self.encoder_block_num-1:  # bev-> voxel or voxel_i -> voxel_j
                bev_queries = output.view(block_bev_z, bev_h, bev_w, bs, self.encoder_embed_dims[block_index])
                bev_queries = bev_queries.permute(1, 2, 3, 0, 4)
                bev_queries = bev_queries.flatten(3)  # (bev_h, bev_w, bs, embed_dims1*z1)
                bev_queries = self.bev_voxel_transfers[block_index](bev_queries)  # (bev_h, bev_w, bs, embed_dims2*z2)
                bev_queries = bev_queries.view(bev_h, bev_w, bs, self.feature_map_z[block_index+1], self.encoder_embed_dims[block_index+1])
                bev_queries = bev_queries.permute(3, 0, 1, 2, 4)
                bev_queries = bev_queries.reshape(-1, bs, self.encoder_embed_dims[block_index+1])  # (num_query, bs, embed_dims)
                
        return block_features  # is a list 

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_z,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        block_features = self.get_voxel_features(
            mlvl_feats,
            bev_queries,
            bev_z,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # voxel_embed shape: (bs, num_query, embed_dims)

        voxel_embed = block_features[-1]
        
        bs = mlvl_feats[0].size(0)
        # object_query_embed  (num_box_query, embed_dims*2)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)  # (bs, num_box_query, embed_dims)
        query = query.unsqueeze(0).expand(bs, -1, -1)  # (bs, num_box_query, embed_dims)
        reference_points = self.reference_points(query_pos)  # (bs, num_box_query, 3)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        voxel_embed = voxel_embed.permute(1, 0, 2)  # (num_query, bs, embed_dims)

        if self.decoder_on_bev:
            bev_embed = voxel_embed.view(self.feature_map_z[-1], bev_h, bev_w, bs, self.encoder_embed_dims[-1])
            bev_embed = bev_embed.permute(1, 2, 3, 0, 4)
            bev_embed = bev_embed.flatten(3)
            bev_embed = self.voxel2bev(bev_embed)
            bev_embed = bev_embed.view(-1, bs, self.embed_dims)
            
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
                **kwargs)
        else:
            inter_states, inter_references = self.decoder(
                query=query,
                key=None,
                value=voxel_embed,
                query_pos=query_pos,
                reference_points=reference_points,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                spatial_shapes=torch.tensor([[bev_z, bev_h, bev_w]], device=query.device),
                level_start_index=torch.tensor([0], device=query.device),
                **kwargs)

        inter_references_out = inter_references

        return block_features, inter_states, init_reference_out, inter_references_out



