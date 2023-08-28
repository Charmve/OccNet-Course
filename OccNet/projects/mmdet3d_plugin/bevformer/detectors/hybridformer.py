# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

from tkinter.messagebox import NO
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from torch.nn import functional as F


@DETECTORS.register_module()
class HybridFormer(MVXTwoStageDetector):
    """HybridFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 keep_bev_history=False,
                 use_occ_gts=False,
                 only_occ=False,
                 only_det=True,
                 dataset_type='nuscenes',
                 can_bus_in_dataset=True,
                 ):

        super(HybridFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.dataset_type=dataset_type
        self.can_bus_in_dataset = can_bus_in_dataset

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        self.keep_bev_history = keep_bev_history
        self.use_occ_gts = use_occ_gts
        self.only_occ = only_occ  # only output occupancy, do not consider the detection result
        self.only_det = only_det  # only output detection result


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)  # (bs, num_cam, 3, H, W)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                # img.squeeze_()  # this is wrong for num_cam = 1
                img.squeeze_(dim=0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)  
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)  # for num_outs=4, [(bs, _dim_, hi, wi)]

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points.
           return: List[List(Tensor)]
        """

        pc_range = pts[0][0].new_tensor(self.pts_bbox_head.point_cloud_range)
        voxel_size = pts[0][0].new_tensor(self.pts_bbox_head.occupancy_size)
        x_dim = int((pc_range[3] - pc_range[0]) / voxel_size[0])
        y_dim = int((pc_range[4] - pc_range[1]) / voxel_size[1])
        z_dim = int((pc_range[5] - pc_range[2]) / voxel_size[2])
        coors = []
        for pts_sample in pts:
            coors_sample = []
            for p in pts_sample:
                coors_sample.append(self.voxelize([p], pc_range, voxel_size, (x_dim, y_dim, z_dim)))
            coors.append(coors_sample)
        return coors

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None, points=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        if points is not None:
            pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
            return (img_feats, pts_feats)
        
        return img_feats


    def forward_pts_train(self,
                          img_feats,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          occ_gts,
                          flow_gts,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev)

        # the pts_bbox_head is VoxelFormerOccupancyHead
        if self.dataset_type == 'semantic_kitti':
            loss_inputs = [occ_gts, outs]
            losses = self.pts_bbox_head.loss_semantic_kitti(*loss_inputs, img_metas=img_metas)
        else:
            if self.only_det:
                loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
                losses = self.pts_bbox_head.loss_only_detection(*loss_inputs, img_metas=img_metas)
            elif self.only_occ:
                loss_inputs = [gt_bboxes_3d, gt_labels_3d, pts_feats, occ_gts, flow_gts, outs]
                losses = self.pts_bbox_head.loss_only_occupancy(*loss_inputs, img_metas=img_metas)
            else:  # joint detection and occupancy
                loss_inputs = [gt_bboxes_3d, gt_labels_3d, pts_feats, occ_gts, flow_gts, outs]
                losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev
    
    def obtain_all_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()
        prev_bevs = []
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
                prev_bevs.append(prev_bev)
            self.train()
            return prev_bevs 

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      prev_gt_bboxes_3d=None,
                      prev_gt_labels=None,
                      occ_gts=None,
                      flow_gts=None,
                      **kwargs
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        if self.keep_bev_history:
            prev_bev = self.obtain_all_history_bev(prev_img, prev_img_metas)  # list(Tensor), Tensor: (bs, num_query, embed_dims)
        else:
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
            if self.use_occ_gts:
                gt_bboxes_3d = [_gt_bboxes_3d[-1] for _gt_bboxes_3d in gt_bboxes_3d] 
                gt_labels_3d = [_gt_labels_3d[-1] for _gt_labels_3d in gt_labels_3d]
                occ_gts = [[p[-1]] for p in occ_gts]  # occ_gts[bs][queue_index]
                if flow_gts is not None:
                    flow_gts = [[p[-1]] for p in flow_gts]

        img_metas = [each[len_queue-1] for each in img_metas]
        feats = self.extract_feat(img=img, img_metas=img_metas, points=points)
        if points is not None:
            img_feats, pts_feats = feats
        else:
            img_feats = feats
            pts_feats = None
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, pts_feats,
                                            gt_bboxes_3d,
                                            gt_labels_3d, occ_gts, flow_gts,
                                            img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, points=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        if self.can_bus_in_dataset:
            tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
            tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
            if self.prev_frame_info['prev_bev'] is not None:
                img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
                img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
            else:
                img_metas[0][0]['can_bus'][-1] = 0
                img_metas[0][0]['can_bus'][:3] = 0
        new_prev_bev, bbox_results, occ_results = self.simple_test(
            img_metas[0], img[0], points, prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        if self.can_bus_in_dataset:
            self.prev_frame_info['prev_pos'] = tmp_pos
            self.prev_frame_info['prev_angle'] = tmp_angle
            self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results, occ_results

    def simple_test_pts(self, x, pts_feats, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        occupancy_preds = outs.get('occupancy_preds', None)
        flow_preds = outs.get('flow_preds', None)
        occ_results = {}
        occ_results['occupancy_preds'] = occupancy_preds
        occ_results['flow_preds'] = flow_preds

        if self.only_occ:
            bbox_results = None
        else:
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas, rescale=rescale)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
        return outs['bev_embed'], bbox_results, occ_results
       
    def simple_test(self, img_metas, img=None, points=None, prev_bev=None, rescale=False,
                    occ_threshold=0.25):
        """Test function without augmentaiton."""
        feats = self.extract_feat(img=img, img_metas=img_metas, points=points)
        if points is not None:
            img_feats, pts_feats = feats
        else:
            img_feats = feats
            pts_feats = None

        bbox_list = [dict() for i in range(len(img_metas))]
        result = self.simple_test_pts(
            img_feats, pts_feats, img_metas, prev_bev, rescale=rescale)
        
        new_prev_bev, bbox_pts, occ_results = result
        if occ_results['occupancy_preds'] is not None:
            occ_results = self.pts_bbox_head.get_occupancy_prediction(occ_results, occ_threshold)
        
        if bbox_pts is None:
            bbox_list = None
        else:
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list, occ_results
        

    