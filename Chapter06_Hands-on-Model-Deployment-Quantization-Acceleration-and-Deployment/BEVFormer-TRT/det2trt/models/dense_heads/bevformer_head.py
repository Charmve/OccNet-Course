import copy
import torch
import torch.nn as nn
from mmdet.models import HEADS
from third_party.bev_mmdet3d import BEVFormerHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox import build_bbox_coder
from ..utils import LINEAR_LAYERS


class ReLUAddZeros(nn.Module):
    def __init__(self):
        super(ReLUAddZeros, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + torch.zeros_like(x))


@HEADS.register_module()
class BEVFormerHeadTRT(BEVFormerHead):
    """Head of Detr3D.
        Args:
            with_box_refine (bool): Whether to refine the reference points
                in the decoder. Defaults to False.
            as_two_stage (bool) : Whether to generate the proposal from
                the outputs of encoder.
            transformer (obj:`ConfigDict`): ConfigDict is used for building
                the Encoder and Decoder.
            bev_h, bev_w (int): spatial shape of BEV queries.
        """

    def __init__(
        self,
        *args,
        with_box_refine=False,
        as_two_stage=False,
        transformer=None,
        bbox_coder=None,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=30,
        bev_w=30,
        linear_cfg=None,
        **kwargs
    ):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        if linear_cfg is None:
            linear_cfg = dict(type="Linear")
        self.linear = LINEAR_LAYERS.get(linear_cfg["type"])

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage
        if "code_size" in kwargs:
            self.code_size = kwargs["code_size"]
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        super(BEVFormerHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False), requires_grad=False
        )

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(self.linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(self.linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(self.linear(self.embed_dims, self.embed_dims))
            # TODO: Waiting bug of TensorRT-8.5.1.7 fixed
            reg_branch.append(
                nn.ReLU(inplace=True) if self.linear == nn.Linear else ReLUAddZeros()
            )
        reg_branch.append(self.linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if self.as_two_stage
            else self.transformer.decoder.num_layers
        )

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

    def forward_trt(
        self, mlvl_feats, prev_bev, can_bus, lidar2img, image_shape, use_prev_bev
    ):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros(
            (bs, self.bev_h, self.bev_w), device=bev_queries.device
        ).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        outputs = self.transformer.forward_trt(
            mlvl_feats,
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches
            if self.with_box_refine
            else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None,
            can_bus=can_bus,
            lidar2img=lidar2img,
            prev_bev=prev_bev,
            image_shape=image_shape,
            use_prev_bev=use_prev_bev,
        )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            )

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        return bev_embed, outputs_classes, outputs_coords

    def get_bboxes_trt(self, outputs_classes, outputs_coords, img_metas, rescale=False):
        dic = {"all_cls_scores": outputs_classes, "all_bbox_preds": outputs_coords}
        return self.get_bboxes(dic, img_metas, rescale=rescale)


@HEADS.register_module()
class BEVFormerHeadTRTP(BEVFormerHeadTRT):
    """Head of Detr3D.
        Args:
            with_box_refine (bool): Whether to refine the reference points
                in the decoder. Defaults to False.
            as_two_stage (bool) : Whether to generate the proposal from
                the outputs of encoder.
            transformer (obj:`ConfigDict`): ConfigDict is used for building
                the Encoder and Decoder.
            bev_h, bev_w (int): spatial shape of BEV queries.
        """

    def __init__(self, *args, **kwargs):
        super(BEVFormerHeadTRTP, self).__init__(*args, **kwargs)

    def forward_trt(
        self, mlvl_feats, prev_bev, can_bus, lidar2img, image_shape, use_prev_bev
    ):
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight
        bev_queries = self.bev_embedding.weight

        bev_mask = torch.zeros(
            (1, self.bev_h, self.bev_w), device=bev_queries.device
        ).to(dtype)
        bev_pos = self.positional_encoding(bev_mask)

        outputs = self.transformer.forward_trt(
            mlvl_feats,
            bev_queries,  # [200*200, 256]
            object_query_embeds,  # [900, 512]
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,  # [1, 256, 200, 200]
            reg_branches=self.reg_branches  # self.reg_branches
            if self.with_box_refine
            else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None,  # None
            can_bus=can_bus,
            lidar2img=lidar2img,
            prev_bev=prev_bev,
            image_shape=image_shape,
            use_prev_bev=use_prev_bev,
        )

        bev_embed, hs, init_reference, inter_references = outputs
        init_reference = init_reference.view(1, self.num_query, 3)
        inter_references = inter_references.view(-1, 1, self.num_query, 3)
        hs = hs.view(-1, 1, self.num_query, self.embed_dims)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(int(hs.shape[0])):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            outputs_coord = self.reg_branches[lvl](hs[lvl])

            assert reference.shape[-1] == 3
            outputs_coord[..., 0:2] += reference[..., 0:2]
            outputs_coord[..., 0:2] = outputs_coord[..., 0:2].sigmoid()
            outputs_coord[..., 4:5] += reference[..., 2:3]
            outputs_coord[..., 4:5] = outputs_coord[..., 4:5].sigmoid()
            outputs_coord[..., 0:1] = (
                outputs_coord[..., 0:1] * (self.pc_range[3] - self.pc_range[0])
                + self.pc_range[0]
            )
            outputs_coord[..., 1:2] = (
                outputs_coord[..., 1:2] * (self.pc_range[4] - self.pc_range[1])
                + self.pc_range[1]
            )
            outputs_coord[..., 4:5] = (
                outputs_coord[..., 4:5] * (self.pc_range[5] - self.pc_range[2])
                + self.pc_range[2]
            )

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        return bev_embed, outputs_classes, outputs_coords
