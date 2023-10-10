# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

import torch
import torch.nn as nn
from mmdet.models import HEADS
from mmcv.cnn import xavier_init
import math

@HEADS.register_module()
class SegmentationHead(nn.Module):
    """
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    Taken from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/models/LMSCNet.py#L7
    """

    def __init__(self, inplanes=64, planes=64, nbr_classes=16, dilations_conv_list=[1, 2, 3],
                 num_occ_fcs=2,
                 flow_head=False,
                 flow_gt_dimension=2):
        super().__init__()

        # First convolution
        self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn1 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.conv2 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.relu = nn.ReLU()

        # cov3dL input: N, C_{in}, D_{in}, H_{in}, W_{in}
        self.conv_classes = nn.Conv3d(
            planes, nbr_classes, kernel_size=3, padding=1, stride=1
        )  
        
        self.flow_head = flow_head
        if self.flow_head:  # TODO
            self.conv_flow = nn.Conv3d(
            planes, planes, kernel_size=3, padding=1, stride=1
        )  
            flow_branch = []
            for _ in range(num_occ_fcs):
                flow_branch.append(nn.Linear(planes, planes))
                flow_branch.append(nn.LayerNorm(planes))
                flow_branch.append(nn.ReLU(inplace=True))
            flow_branch.append(nn.Linear(planes, flow_gt_dimension))
            self.flow_branches = nn.Sequential(*flow_branch)

    def forward(self, x_in):
        # Convolution to go from inplanes to planes features...
        x_in = self.relu(self.conv0(x_in))
        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        occ = self.conv_classes(x_in)
        
        if self.flow_head is not None:
            flow = self.conv_flow(x_in)  # (bs, c, d, h, w)
            flow = flow.permute(0, 2, 3, 4, 1)
            flow = self.flow_branches(flow)
        else:
            flow = None
        return occ, flow
