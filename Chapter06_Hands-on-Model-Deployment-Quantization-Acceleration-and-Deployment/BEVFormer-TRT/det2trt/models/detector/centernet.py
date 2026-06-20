from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.centernet import CenterNet
from mmdet.core import bbox2result


@DETECTORS.register_module()
class CenterNetTRT(CenterNet):
    def __init__(self, *args, **kwargs):
        super(CenterNetTRT, self).__init__(*args, **kwargs)

    def forward_trt(self, image):  # 被trt优化
        out = self.forward_dummy(image)

        center_heatmap_preds = out[0]
        wh_preds = out[1]
        offset_preds = out[2]

        return center_heatmap_preds, wh_preds, offset_preds

    def post_process(self, center_heatmap_preds, wh_preds, offset_preds, img_metas):

        if type(center_heatmap_preds) != type([]):
            center_heatmap_preds = [center_heatmap_preds]
            wh_preds = [wh_preds]
            offset_preds = [offset_preds]

        result_list = self.bbox_head.get_bboxes(
            center_heatmap_preds, wh_preds, offset_preds, img_metas, rescale=True
        )

        return [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in result_list
        ]
