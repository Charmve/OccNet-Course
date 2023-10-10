import copy
import os
from pickle import TRUE
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
from os import path as osp, stat
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmcv.parallel import DataContainer as DC
import random
from torch.nn import functional as F
from mmcv.runner import force_fp32
from collections import defaultdict
import math
from tqdm import tqdm
import cv2
import shutil
from .kitti_metrics import KittiSSCMetrics
import pickle

@DATASETS.register_module()
class CustomSemanticKittiDataset(NuScenesDataset):
    r"""SemanticKitti Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """
    def __init__(self, queue_length=4, bev_size=(256, 256), overlap_test=False,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        
        self.point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
        self.occupancy_size = [0.2, 0.2, 0.2]
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.occupancy_classes = 19
        self.voxel_num = self.occ_xdim*self.occ_ydim*self.occ_zdim

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.
        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos
        
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = []

        # temporal aug
        prev_indexs_list = list(range(index-self.queue_length, index))
        random.shuffle(prev_indexs_list)
        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
        ##

        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        frame_idx = input_dict['frame_idx']
        scene_token = input_dict['scene_token']
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        
        data_queue.insert(0, example)
        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict['frame_idx'] < frame_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                frame_idx = input_dict['frame_idx']
            data_queue.insert(0, copy.deepcopy(example))
        return self.union2one(data_queue)

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if i == 0:
                metas_map[i]['prev_bev'] = False
                # prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                # prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                # metas_map[i]['can_bus'][:3] = 0
                # metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                # tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                # tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                # metas_map[i]['can_bus'][:3] -= prev_pos  # the delta position of adjacent timestamps
                # metas_map[i]['can_bus'][-1] -= prev_angle  # the delta orientation of adjacent timestamps
                # prev_pos = copy.deepcopy(tmp_pos)
                # prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        # if 'occ_gts' in queue[-1]:
        #     occ_gt_list = [each['occ_gts'].data for each in queue] 
        #     queue[-1]['occ_gts'] = DC(occ_gt_list, cpu_only=False)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=None,
            pts_filename=None,
            sweeps=None,
            ego2global_translation=None,
            ego2global_rotation=None,
            prev_idx=None,
            next_idx=None,
            scene_token=info['scene_token'],
            can_bus=None,
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )
        
        if 'occ_gt_path' in info:
            input_dict['occ_gt_path'] = info['occ_gt_path']
       
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2cam_rt = cam_info['lidar2cam']
                lidar2img_rt = viewpad @ lidar2cam_rt  # point in lidar to pixel in image
                lidar2img_rts.append(lidar2img_rt)  

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt)   # point in lidar to point in cam

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            # annos = self.get_ann_info(index)
            annos=None
            input_dict['ann_info'] = annos
            
        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def evaluate_occ_iou(self, occupancy_results, flow_results, show_dir=None, 
                         save_interval=50, occ_threshold=0.25, runner=None):
        """ save the gt_occupancy_sparse and evaluate the iou metrics"""
        assert len(occupancy_results) == len(self)
        thre_str = 'thre_'+'{:.2f}'.format(occ_threshold)
        
        # set the metrics
        n_classes = 20
        self.eval_metrics = KittiSSCMetrics(n_classes)
        
        # loop the dataset
        for index in tqdm(range(len(occupancy_results))):
            info = self.data_infos[index]
            scene_name = info['scene_name']
            frame_idx = info['frame_idx']
                
            # load occ_gt
            gt_occupancy = np.load(info['occ_gt_path'])  # ( 256, 256, 32)
            
            relabel_prediction = True
            if relabel_prediction:  # to (256, 256, 32)
                occ_pred_sparse = occupancy_results[index].cpu().numpy()
                pred_occupancy = self.get_voxel_prediction(occ_pred_sparse)
                pred_occupancy = pred_occupancy.reshape(32, 256, 256)
                pred_occupancy = pred_occupancy.transpose(2, 1, 0)  # value in 0, 1...,19
                pred_occupancy = pred_occupancy+1
                pred_occupancy[pred_occupancy == n_classes] = 0
                
            # output result
            if show_dir and index % save_interval == 0:
                save_dir = os.path.join(show_dir, thre_str, scene_name)
                os.makedirs(save_dir, exist_ok=True)
                occ_save_dir = os.path.join(save_dir, 'occupancy')
                os.makedirs(occ_save_dir, exist_ok=True)
                image_save_dir = os.path.join(save_dir, 'images')
                os.makedirs(image_save_dir, exist_ok=True)
            
                out_dict = {"y_pred": pred_occupancy.astype(np.uint16)}
                out_dict["target"] = gt_occupancy.astype(np.uint16)
                out_dict["fov_mask_1"] = info['cams']['image_2']['fov_mask_1']
                out_dict["cam_k"] = info['cams']['image_2']['cam_intrinsic']
                out_dict["T_velo_2_cam"] = info['cams']['image_2']['lidar2cam']
                
                filepath = os.path.join(occ_save_dir, "{:06d}.pkl".format(frame_idx))
                with open(filepath, "wb") as handle:
                    pickle.dump(out_dict, handle)
                    print("wrote to", filepath)
                
                image_save_path = osp.join(image_save_dir, '{:05d}.png'.format(frame_idx))
                shutil.copyfile(info['cams']['image_2']['data_path'], image_save_path)
            
            # using ssc metrics within a batch
            y_pred = np.expand_dims(pred_occupancy, axis=0) 
            y_true = np.expand_dims(gt_occupancy, axis=0)  
            self.eval_metrics.add_batch(y_pred, y_true)

        # dataset metrics  19类
        class_names = [
                        "empty",
                        "car",
                        "bicycle",
                        "motorcycle",
                        "truck",
                        "other-vehicle",
                        "person",
                        "bicyclist",
                        "motorcyclist",
                        "road",
                        "parking",
                        "sidewalk",
                        "other-ground",
                        "building",
                        "fence",
                        "vegetation",
                        "trunk",
                        "terrain",
                        "pole",
                        "traffic-sign",
                    ]

        eval_result= self.eval_metrics.get_stats()
       
        print(f'======out evaluation metrics: {thre_str}=========')
        for i, class_name in enumerate(class_names):
            print("miou/{}: {:.2f}".format(class_name, eval_result["iou_ssc"][i]))
        print("miou: {:.2f}".format(eval_result["iou_ssc_mean"]))
        print("iou: {:.2f}".format(eval_result["iou"]))
        print("Precision: {:.4f}".format(eval_result["precision"]))
        print("Recall: {:.4f}".format(eval_result["recall"]))
        
        if runner is not None:
            for i, class_name in enumerate(class_names):
                runner.log_buffer.output[class_name] =  eval_result["iou_ssc"][i]
            runner.log_buffer.output['miou'] =  eval_result["iou_ssc_mean"]
            runner.log_buffer.output['iou'] =  eval_result["iou"]
            runner.log_buffer.output['precision'] =  eval_result["precision"]
            runner.log_buffer.output['recall'] =  eval_result["recall"]
            
            runner.log_buffer.ready = True
            

    def get_voxel_prediction(self, occupancy):
        occ_index = occupancy[:, 0]
        occ_class = occupancy[:, 1]
        pred_occupancy = np.ones(self.voxel_num, dtype=np.int64)*self.occupancy_classes
        pred_occupancy[occ_index] = occ_class  # (num_voxels)
        return pred_occupancy