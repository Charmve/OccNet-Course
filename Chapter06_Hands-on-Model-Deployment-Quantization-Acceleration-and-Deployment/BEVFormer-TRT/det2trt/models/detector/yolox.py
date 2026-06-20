from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.yolox import YOLOX
from mmdet.core import bbox2result


@DETECTORS.register_module()
class YOLOXTRT(YOLOX):
    r"""Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_

    Note: Considering the trade-off between training speed and accuracy,
    multi-scale training is temporarily kept. More elegant implementation
    will be adopted in the future.

    Args:
        backbone (nn.Module): The backbone module.
        neck (nn.Module): The neck module.
        bbox_head (nn.Module): The bbox head module.
        train_cfg (obj:`ConfigDict`, optional): The training config
            of YOLOX. Default: None.
        test_cfg (obj:`ConfigDict`, optional): The testing config
            of YOLOX. Default: None.
        pretrained (str, optional): model pretrained path.
            Default: None.
        input_size (tuple): The model default input image size. The shape
            order should be (height, width). Default: (640, 640).
        size_multiplier (int): Image size multiplication factor.
            Default: 32.
        random_size_range (tuple): The multi-scale random range during
            multi-scale training. The real training image size will
            be multiplied by size_multiplier. Default: (15, 25).
        random_size_interval (int): The iter interval of change
            image size. Default: 10.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self, *args, **kwargs):
        super(YOLOXTRT, self).__init__(*args, **kwargs)

    def forward_trt(self, image):
        out = self.forward_dummy(image)
        cls_score_1, cls_score_2, cls_score_3 = out[0]
        bbox_pred_1, bbox_pred_2, bbox_pred_3 = out[1]
        objectness_1, objectness_2, objectness_3 = out[2]
        return (
            cls_score_1,
            cls_score_2,
            cls_score_3,
            bbox_pred_1,
            bbox_pred_2,
            bbox_pred_3,
            objectness_1,
            objectness_2,
            objectness_3,
        )

    def post_process(
        self,
        cls_score_1,
        cls_score_2,
        cls_score_3,
        bbox_pred_1,
        bbox_pred_2,
        bbox_pred_3,
        objectness_1,
        objectness_2,
        objectness_3,
        img_metas,
    ):
        cls_scores = [cls_score_1, cls_score_2, cls_score_3]
        bbox_preds = [bbox_pred_1, bbox_pred_2, bbox_pred_3]
        objectnesses = [objectness_1, objectness_2, objectness_3]
        result_list = self.bbox_head.get_bboxes(
            cls_scores, bbox_preds, objectnesses, img_metas, rescale=True
        )
        return [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in result_list
        ]
