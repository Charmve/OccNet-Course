import torch
from mmdet.models import DETECTORS
from third_party.bev_mmdet3d import BEVFormer
from third_party.bev_mmdet3d.core.bbox import bbox3d2result


@DETECTORS.register_module()
class BEVFormerTRT(BEVFormer):
    def __init__(self, *args, **kwargs):
        super(BEVFormerTRT, self).__init__(*args, **kwargs)

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W)
                )
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def forward_trt(self, image, prev_bev, use_prev_bev, can_bus, lidar2img):
        image_shape = image.shape[-2:]
        img_feats = self.extract_feat(img=image)
        assert len(img_feats[0]) == 1
        bev_embed, outputs_classes, outputs_coords = self.pts_bbox_head.forward_trt(
            img_feats, prev_bev, can_bus, lidar2img, image_shape, use_prev_bev
        )
        return bev_embed, outputs_classes, outputs_coords

    def post_process(self, outputs_classes, outputs_coords, img_metas):
        dic = {"all_cls_scores": outputs_classes, "all_bbox_preds": outputs_coords}
        result_list = self.pts_bbox_head.get_bboxes(dic, img_metas, rescale=True)

        return [
            {
                "pts_bbox": bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in result_list
            }
        ]
