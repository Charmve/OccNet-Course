from mmdet.models import DETECTORS
from ..utils.register import TRT_FUNCTIONS

from third_party.bev_mmdet3d import BEVDet
from third_party.bev_mmdet3d.core.bbox import bbox3d2result


@DETECTORS.register_module()
class BEVDetTRT(BEVDet):
    def __init__(self, *args, **kwargs):
        super(BEVDetTRT, self).__init__(*args, **kwargs)
        self.bev_pool_v2 = TRT_FUNCTIONS.get("bev_pool_v2_2")

    def get_bev_pool_input(
        self, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda
    ):
        coor = self.img_view_transformer.get_lidar_coor(
            sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda
        )
        (
            ranks_bev,
            ranks_depth,
            ranks_feat,
            order,
            coor,
        ) = self.img_view_transformer.voxel_pooling_prepare_v2(coor)
        return ranks_bev, ranks_depth, ranks_feat, order, coor

    def forward_trt(
        self,
        image,
        ranks_bev,
        ranks_depth,
        ranks_feat,
        interval_starts,
        interval_lengths,
    ):
        ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = (
            ranks_bev.int(),
            ranks_depth.int(),
            ranks_feat.int(),
            interval_starts.long(),
            interval_lengths.long(),
        )

        x = self.img_backbone(image.flatten(0, 1))
        x = self.img_neck(x)
        x = self.img_view_transformer.depth_net(x)
        depth = x[:, : self.img_view_transformer.D].softmax(dim=1)
        tran_feat = x[
            :,
            self.img_view_transformer.D : (
                self.img_view_transformer.D + self.img_view_transformer.out_channels
            ),
        ]
        tran_feat = tran_feat.permute(0, 2, 3, 1)

        depth = depth.contiguous()
        tran_feat = tran_feat.contiguous()

        x = self.bev_pool_v2(
            depth,
            tran_feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        bev_feat = self.bev_encoder(x)
        outs = self.pts_bbox_head.forward_trt(bev_feat)
        reg, height, dim, rot, vel, heatmap = (
            outs["reg"],
            outs["height"],
            outs["dim"],
            outs["rot"],
            outs["vel"],
            outs["heatmap"],
        )

        return reg, height, dim, rot, vel, heatmap

    def post_process(self, reg, height, dim, rot, vel, heatmap, img_metas):
        out_dict = [
            [dict(reg=reg, height=height, dim=dim, rot=rot, vel=vel, heatmap=heatmap)]
        ]
        result_list = self.pts_bbox_head.get_bboxes(out_dict, img_metas, rescale=True)

        return [
            {
                "pts_bbox": bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in result_list
            }
        ]
