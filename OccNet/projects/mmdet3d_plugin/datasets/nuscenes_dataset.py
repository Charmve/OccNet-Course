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
from .nuscnes_eval import Axis, NuScenesEval_custom
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random
from torch.nn import functional as F
from tqdm import tqdm
import shutil
from .occupancy_metrics import SSCMetrics

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, queue_length=4, bev_size=(200, 200), 
                 overlap_test=False, 
                 occ_type='normal',
                 use_occ_gts=True,
                 train_with_partial_data=False,
                 split_divisor=1,
                 *args, **kwargs):
        self.use_occ_gts = use_occ_gts
        self.occ_type = occ_type
        self.point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
        if self.occ_type == 'normal':
            self.occupancy_size = [0.5, 0.5, 0.5]
        elif self.occ_type == 'fine':
            self.occupancy_size = [0.25, 0.25, 0.25]
        elif self.occ_type == 'coarse':
            self.occupancy_size = [1.0, 1.0, 1.0]
        
        # define occupancy class name: 10 foreground + 6 backgrounds
        self.occupancy_class_names = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 
                            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation']
            
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.occupancy_classes = 16
        self.voxel_num = self.occ_xdim*self.occ_ydim*self.occ_zdim
        
        self.train_with_partial_data = train_with_partial_data
        if self.train_with_partial_data:
            self.valid_scenes = set(np.load(os.path.join('data/nuscenes_train_splits', f'split_{split_divisor}.npy')))
        
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.
        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        if self.train_with_partial_data:
            data_infos = [info for info in data_infos if info['scene_name'] in self.valid_scenes]
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
        if self.filter_empty_gt and \
                (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None
        data_queue.insert(0, example)
        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict['frame_idx'] < frame_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                if self.filter_empty_gt and \
                        (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                    return None
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
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos  # the delta position of adjacent timestamps
                metas_map[i]['can_bus'][-1] -= prev_angle  # the delta orientation of adjacent timestamps
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        if self.use_occ_gts:
            queue[-1]['gt_bboxes_3d'] = DC([each['gt_bboxes_3d'].data for each in queue], cpu_only=True)
            queue[-1]['gt_labels_3d'] = DC([each['gt_labels_3d'].data for each in queue])
            if 'occ_gts' in queue[-1]:
                occ_gt_list = [each['occ_gts'].data for each in queue] 
                queue[-1]['occ_gts'] = DC(occ_gt_list, cpu_only=False)
            if 'flow_gts' in queue[-1]:
                flow_gt_list = [each['flow_gts'].data for each in queue] 
                queue[-1]['flow_gts'] = DC(flow_gt_list, cpu_only=False)
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
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if 'occ_gt_path' in info:
            input_dict['occ_gt_path'] = info['occ_gt_path']
        if 'flow_gt_path' in info:
            input_dict['flow_gt_path'] = info['flow_gt_path']

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

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

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root,
                             verbose=True)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def evaluate_occ_iou(self, occupancy_results, flow_results, show_dir=None, 
                         save_interval=1, occ_threshold=0.25, runner=None):
        """ save the gt_occupancy_sparse and evaluate the iou metrics"""
        assert len(occupancy_results) == len(self)
        thre_str = 'thre_'+'{:.2f}'.format(occ_threshold)

        if show_dir: 
            scene_names = [info['scene_name'] for info in self.data_infos]
            scene_names = np.unique(scene_names)
            save_scene_names = scene_names[:5]
            print('save_scene_names:', save_scene_names)
        
        # set the metrics
        self.eval_metrics = SSCMetrics(self.occupancy_classes+1, occ_type=self.occ_type)
        # loop the dataset
        for index in tqdm(range(len(occupancy_results))):
            info = self.data_infos[index]
            scene_name = info['scene_name']
            frame_idx = info['frame_idx']
            
            # load occ_gt
            occ_gt_sparse = np.load(info['occ_gt_path'])
            occ_index = occ_gt_sparse[:, 0]
            occ_class = occ_gt_sparse[:, 1] 
            gt_occupancy = np.ones(self.voxel_num, dtype=np.uint8)*self.occupancy_classes
            gt_occupancy[occ_index] = occ_class  # (num_voxels)

            # load occ_invalid
            if 'occ_invalid_path' in info:
                occ_invalid_index = np.load(info['occ_invalid_path'])
                visible_mask = np.ones(self.voxel_num, dtype=np.uint8)
                visible_mask[occ_invalid_index] = 0
            else:
                visible_mask = None
            #  load occ_pred
            occ_pred_sparse = occupancy_results[index].cpu().numpy()
            pred_occupancy = self.get_voxel_prediction(occ_pred_sparse)

            # load flow info
            flow_pred, flow_true = None, None
            if flow_results is not None:
                flow_pred_sparse = flow_results[index].cpu().numpy()
                flow_gt_sparse = np.load(info['flow_gt_path']) 
                flow_true, flow_pred = self.parse_flow_info(occ_gt_sparse, occ_pred_sparse, flow_gt_sparse, flow_pred_sparse)

            self.eval_metrics.add_batch(pred_occupancy, gt_occupancy, flow_pred=flow_pred, flow_true=flow_true, visible_mask=visible_mask)

            # save occ, flow, image
            if show_dir and index % save_interval == 0:
                save_result = False
                if scene_name in save_scene_names:
                    save_result = True
                if save_result:
                    occ_gt_save_dir = os.path.join(show_dir, thre_str, scene_name, 'occ_gts')
                    occ_pred_save_dir = os.path.join(show_dir, thre_str, scene_name, 'occ_preds')
                    image_save_dir = os.path.join(show_dir, thre_str, scene_name, 'images')
                    os.makedirs(occ_gt_save_dir, exist_ok=True)
                    os.makedirs(occ_pred_save_dir, exist_ok=True)
                    os.makedirs(image_save_dir, exist_ok=True)

                    np.save(osp.join(occ_gt_save_dir, '{:03d}_occ.npy'.format(frame_idx)), occ_gt_sparse)  
                    np.save(osp.join(occ_pred_save_dir, '{:03d}_occ.npy'.format(frame_idx)), occ_pred_sparse)  

                    if flow_results is not None:
                        np.save(osp.join(occ_gt_save_dir, '{:03d}_flow.npy'.format(frame_idx)), flow_gt_sparse)  
                        np.save(osp.join(occ_pred_save_dir, '{:03d}_flow.npy'.format(frame_idx)), flow_pred_sparse)
                    image_save_path = osp.join(image_save_dir, '{:03d}.png'.format(frame_idx))
                    if 'surround_image_path' in info:
                        shutil.copyfile(info['surround_image_path'], image_save_path)

        eval_resuslt = self.eval_metrics.get_stats()
        print(f'======out evaluation metrics: {thre_str}=========')
        
        for i, class_name in enumerate(self.occupancy_class_names):
            print("miou/{}: {:.2f}".format(class_name, eval_resuslt["iou_ssc"][i]))
        print("miou: {:.2f}".format(eval_resuslt["miou"]))
        print("iou: {:.2f}".format(eval_resuslt["iou"]))
        print("Precision: {:.4f}".format(eval_resuslt["precision"]))
        print("Recall: {:.4f}".format(eval_resuslt["recall"]))
        
        # flow_distance = eval_resuslt['flow_distance']
        # flow_states=['flow_distance_sta', 'flow_distance_mov', 'flow_distance_all']
        # for i in range(len(flow_states)):
        #     if flow_distance[i] is not None:
        #         print("{}: {:.4f}".format(flow_states[i], flow_distance[i]))
        
        if runner is not None:
            for i, class_name in enumerate(self.occupancy_class_names):
                runner.log_buffer.output[class_name] =  eval_resuslt["iou_ssc"][i]
            runner.log_buffer.output['miou'] =  eval_resuslt["miou"]
            runner.log_buffer.output['iou'] =  eval_resuslt["iou"]
            runner.log_buffer.output['precision'] =  eval_resuslt["precision"]
            runner.log_buffer.output['recall'] =  eval_resuslt["recall"]
            runner.log_buffer.ready = True
                
    
    def obtain_occ_pred_valid(self, occ_pred_sparse, invalid_occupancy):
        occ_index, occ_class = occ_pred_sparse[:, 0], occ_pred_sparse[:, 1]
        occ_pred_full = np.ones(self.voxel_num, dtype=np.uint8)*self.occupancy_classes
        occ_pred_full[occ_index] = occ_class
        valid_mask = (occ_pred_full != self.occupancy_classes) & (invalid_occupancy != 255)
        occ_pred_valid_index = np.where(valid_mask)[0]
        occ_pred_valid_label = occ_pred_full[valid_mask]
        occ_pred_valid = np.stack([occ_pred_valid_index, occ_pred_valid_label], axis=-1)
        return occ_pred_valid
    
    def obtain_flow_pred_valid(self, occ_pred_sparse, flow_pred_sparse, invalid_occupancy):
        occ_index, occ_class = occ_pred_sparse[:, 0], occ_pred_sparse[:, 1]
        occ_pred_full = np.ones(self.voxel_num, dtype=np.uint8)*self.occupancy_classes
        occ_pred_full[occ_index] = occ_class
        flow_pred_full = np.zeros((self.voxel_num, 2), dtype=np.float32)
        flow_pred_full[occ_index] = flow_pred_sparse
        valid_mask = (occ_pred_full != self.occupancy_classes) & (invalid_occupancy != 255)
        flow_pred_valid = flow_pred_full[valid_mask]
        return flow_pred_valid

    def get_voxel_prediction(self, occupancy):
        occ_index = occupancy[:, 0]
        occ_class = occupancy[:, 1]
        pred_occupancy = np.ones(self.voxel_num, dtype=np.uint8)*self.occupancy_classes
        pred_occupancy[occ_index] = occ_class  # (num_voxels)
        return pred_occupancy
    
    def parse_flow_info(self, occ_gt, occ_pred, flow_gt, flow_pred):
        """
        transform sparse data into consecutive data
        """
        occ_gt_index = occ_gt[:, 0]
        flow_gt_full = np.zeros((self.voxel_num, 2), dtype=np.float32)
        flow_gt_full[occ_gt_index] = flow_gt

        occ_pred_index = occ_pred[:, 0]
        flow_pred_full = np.zeros((self.voxel_num, 2), dtype=np.float32)
        flow_pred_full[occ_pred_index] = flow_pred

        flow_gt_full = np.expand_dims(flow_gt_full, axis=0)  # (1, 640000, 2)
        flow_pred_full = np.expand_dims(flow_pred_full, axis=0)   # (1, 640000, 2)
        return flow_gt_full, flow_pred_full