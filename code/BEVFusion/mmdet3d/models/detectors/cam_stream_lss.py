"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from torchvision.models.resnet import resnet18
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from torchvision.utils import save_image
from mmdet3d.models.fusion_layers import apply_3d_transformation
import torch.nn.functional as F

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, x2.shape[2:],  mode='bilinear', align_corners=True)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0) # 求和 [1,2,3,4,5] -> [1,3,6,10,15]
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool) # [551698, 1]
        kept[:-1] = (ranks[1:] != ranks[:-1]) # 判断ketp前后两个index是否相等

        x, geom_feats = x[kept], geom_feats[kept] # 取特征
        x = torch.cat((x[:1], x[1:] - x[:-1])) # 错位相减

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class CamEncode(nn.Module):
    def __init__(self, D, C, inputC):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C
        self.depthnet = nn.Conv2d(inputC, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        return depth, new_x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x, depth


class LiftSplatShoot(nn.Module):
    def __init__(self, lss=False, final_dim=(900, 1600), camera_depth_range=[4.0, 45.0, 1.0], pc_range=[-50, -50, -5, 50, 50, 3], downsample=4, grid=3, inputC=256, camC=64):
        """
        Args:
            lss (bool): using default downsampled r18 BEV encoder in LSS.
            final_dim: actual RGB image size for actual BEV coordinates, default (900, 1600)
            downsample (int): the downsampling rate of the input camera feature spatial dimension (default (224, 400)) to final_dim (900, 1600), default 4. 
            camera_depth_range, img_depth_loss_weight, img_depth_loss_method: for depth supervision wich is not mentioned in paper.
            pc_range: point cloud range.
            inputC: input camera feature channel dimension (default 256).
            grid: stride for splat, see https://github.com/nv-tlabs/lift-splat-shoot.

        """
        super(LiftSplatShoot, self).__init__()
        self.pc_range = pc_range
        self.grid_conf = {
            'xbound': [pc_range[0], pc_range[3], grid],
            'ybound': [pc_range[1], pc_range[4], grid],
            'zbound': [pc_range[2], pc_range[5], grid],
            'dbound': camera_depth_range,
        }
        self.final_dim = final_dim 
        self.grid = grid

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'], )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = downsample
        self.fH, self.fW = self.final_dim[0] // self.downsample, self.final_dim[1] // self.downsample
        self.camC = camC
        self.inputC = inputC
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.inputC)
        

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
        z = self.grid_conf['zbound']
        cz = int(self.camC * ((z[1] - z[0]) // z[2]))
        self.lss = lss
        self.bevencode = nn.Sequential(
            nn.Conv2d(cz, cz, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cz),
            nn.ReLU(inplace=True),
            nn.Conv2d(cz, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, inputC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inputC),
            nn.ReLU(inplace=True)
        )
        if self.lss:
          self.bevencode = nn.Sequential(
            nn.Conv2d(cz, camC, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(camC),
            BevEncode(inC=camC, outC=inputC)

        )

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.final_dim # [450, 800]
        fH, fW = self.fH, self.fW # [56, 100]
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW) # [41, 56, 100], d方向均匀划分
        D, _, _ = ds.shape # 41
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW) # [41, 56, 100], w方向均匀划分
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW) # [41, 56, 100], h方向均匀划分
        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1) # [41, 56, 100, 3]
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, post_rots=None, post_trans=None,extra_rots=None,extra_trans=None):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape # [1, 6, 3]
        # ADD
        # undo post-transformation
        # B x N x D x H x W x 3
        if post_rots is not None or post_trans is not None:
            if post_trans is not None:
                points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
            if post_rots is not None:
                points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        else:
            # 图像视锥空间按照B和N复制
            # [41, 56, 100, 3] -> [1, 6, 41, 56, 100, 3] -> [1, 6, 41, 56, 100, 3, 1]
            points = self.frustum.repeat(B, N, 1, 1, 1, 1).unsqueeze(-1)  # B x N x D x H x W x 3 x 1

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5) # [1, 6, 41, 56, 100, 3, 1], [u,v] -> [x,y,z], [ud, vd, d] -> [x, y, z]
        points = rots.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1) # [1, 6, 3, 3] -> [1, 6, 1, 1, 1, 3, 3] -> [1, 6, 41, 56, 100, 3]
        points += trans.view(B, N, 1, 1, 1, 3) # [1, 6, 3, 3] -> [1, 6, 1, 1, 1, 3] -> [1, 6, 41, 56, 100, 3]

        if extra_rots is not None or extra_trans is not None:
            if extra_rots is not None:
                points = extra_rots.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
            if extra_trans is not None:
                points += extra_trans.view(B, N, 1, 1, 1, 3)
        return points # [1, 6, 41, 56, 100, 3]

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, H, W = x.shape # [1, 6, 256, 56, 100]

        x = x.view(B * N, C, H, W) # [1, 6, 256, 56, 100] -> [6, 256, 56, 100]
        """
        CamEncode(
            (depthnet): Conv2d(256, 105, kernel_size=(1, 1), stride=(1, 1))
            )
        105 = 64 + 41
        """
        x, depth = self.camencode(x) # x: [6, 64, 41, 56, 100], [6, 41, 56, 100], depth with softmax
        x = x.view(B, N, self.camC, self.D, H, W) # x: [1, 6, 64, 41, 56, 100]
        x = x.permute(0, 1, 3, 4, 5, 2) # x: [1, 6, 64, 41, 56, 100] -> [1, 6, 41, 56, 100, 64]
        depth = depth.view(B, N, self.D, H, W) # [6, 41, 56, 100] -> [1, 6, 41, 56, 100]
        return x, depth # [1, 6, 41, 56, 100, 64], [1, 6, 41, 56, 100]

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape # 1, 6, 41, 56, 100, 64
        Nprime = B * N * D * H * W # 1x6x41x56x100 = 1,377,600
        batch_size = x.shape[0] # 1

        # flatten x feature展平
        x = x.reshape(Nprime, C) # [1, 6, 41, 56, 100, 64] -> [1377600, 64]

        # flatten indices 坐标展平
        # self.bx: [-49.75, -49.75, -4.75]
        # self.dx: [0.5, 0.5, 0.5]
        # self.nx: [200, 200, 16]
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long() # [1, 6, 41, 56, 100, 3], 坐标网格化
        geom_feats = geom_feats.view(Nprime, 3) # [1377600, 3]
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)]) # [1377600, 1]
        batch_ix = batch_ix.to(geom_feats.device)
        geom_feats = torch.cat((geom_feats, batch_ix), 1) # 把batch id附在geom_feats后面 [1377600, 4]
        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2]) # [1377600,], True or False
        x = x[kept] # [551698, 64]
        geom_feats = geom_feats[kept] # [551698, 4]
        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3] # 排序值, rank相等的点在同一个batch，并且在在同一个格子里面, [551698, ]
        sorts = ranks.argsort() # [551698, ] 
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts] # x: [551698, 64], geom_feats: [551698, 4], ranks: [551698, ]
        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks) # 同一个格子的feature 相加, x: [52800, 64], geom_feats: [52800, 64]

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device) # final: [1, 64, 16, 200, 200]
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x # final: [1, 64, 16, 200, 200]
        return final

    def get_voxels(self, x, rots=None, trans=None, post_rots=None, post_trans=None,extra_rots=None,extra_trans=None):
        """
            x: img_feats_view, [1, 6, 256, 56, 100]
            rots: 旋转矩阵, img->lidar, [1, 6, 3, 3]
            trans: 平移矩阵，img->lidar, [1, 6, 3]
        """
        geom = self.get_geometry(rots, trans, post_rots, post_trans,extra_rots,extra_trans) # [1, 6, 41, 56, 100, 3]
        x, depth = self.get_cam_feats(x) # [1, 6, 41, 56, 100, 64], [1, 6, 41, 56, 100]
        x = self.voxel_pooling(geom, x) # [1, 64, 16, 200, 200]
        return x, depth # x: [1, 64, 16, 200, 200], depth: [1, 6, 41, 56, 100]

    def s2c(self, x):
        B, C, H, W, L = x.shape # 1, 64, 16, 200, 200
        bev = torch.reshape(x, (B, C*H, W, L)) # [1, 64, 16, 200, 200] -> [1, 64*16, 200, 200]
        bev = bev.permute((0,1,3,2)) # [1, 1024, 200, 200] -> [1, 1024, 200, 200]
        return bev

    def forward(self, x, rots, trans, lidar2img_rt=None, img_metas=None, post_rots=None, post_trans=None, extra_rots=None,extra_trans=None):
        """
            x: img_feats_view, [1, 6, 256, 56, 100]
            rots: 旋转矩阵, img->lidar, [1, 6, 3, 3]
            trans: 平移矩阵，img->lidar, [1, 6, 3]
            post_rots, post_trans, extra_rots, extra_trans: None
        """
        x, depth = self.get_voxels(x, rots, trans, post_rots, post_trans,extra_rots,extra_trans) # [B, C, H, W, L]
        """
            x: [1, 64, 16, 200, 200]
            depth: [1, 6, 41, 56, 100]
        """
        bev = self.s2c(x) # [1, 1024, 200, 200]
        """
        Sequential(
        (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU(inplace=True)
        (9): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (10): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (11): ReLU(inplace=True)
        )
        """
        x = self.bevencode(bev) # [1, 256, 200, 200]
        return x, depth # [1, 256, 200, 200], [1, 6, 41, 56, 100]

