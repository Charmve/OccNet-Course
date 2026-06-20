from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.yolox_head import YOLOXHead

from pytorch_quantization import nn as quant_nn


@HEADS.register_module()
class YOLOXHeadQ(YOLOXHead):
    """YOLOXHead head used in `YOLOX <https://arxiv.org/abs/2107.08430>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Default: 256
        stacked_convs (int): Number of stacking convs of the head.
            Default: 2.
        strides (tuple): Downsample factor of each feature map.
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_obj (dict): Config of objectness loss.
        loss_l1 (dict): Config of L1 loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, *args, **kwargs):
        super(YOLOXHeadQ, self).__init__(*args, **kwargs)

    def _build_predictor(self):
        """Initialize predictor layers of a single level head."""
        conv_cls = quant_nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        conv_reg = quant_nn.Conv2d(self.feat_channels, 4, 1)
        conv_obj = quant_nn.Conv2d(self.feat_channels, 1, 1)
        return conv_cls, conv_reg, conv_obj
