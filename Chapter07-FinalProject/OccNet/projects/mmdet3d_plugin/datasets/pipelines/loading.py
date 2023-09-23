import mmcv
import torch
import numpy as np

from mmdet3d.core.points import BasePoints
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadPointsFromMultiSweepsWithPadding(object):
    """Load points from multiple sweeps. WILL PAD POINTS DIM TO LOAD DIM
    This is usually used for nuScenes dataset to utilize previous sweeps.
    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.
        Args:
            pts_filename (str): Filename of point clouds data.
        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.
        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.
        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.
        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.
        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.
                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']
        if points.tensor.size(-1) < self.load_dim:
            padding = points.tensor.new_zeros((points.tensor.size(0), self.load_dim-points.tensor.size(1)))
            points.tensor = torch.cat((points.tensor, padding), dim=-1)
            points.points_dim = self.load_dim
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class LoadOccupancyGT(object):
    """load occupancy GT data
       gt_type: index_class, store the occ index and occ class in one file with shape (n, 2)
    """
    def __init__(self, gt_type='index_class', data_type='nuscenes', 
                 relabel=False, occupancy_classes=16):
        self.gt_type = gt_type
        self.data_type = data_type
        self.relabel = relabel
        self.occupancy_classes = occupancy_classes

    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        occ_gts = np.load(occ_gt_path)  # (n, 2)
        if self.data_type == 'semantic_kitti' and self.relabel:
            """
            for semantic kitti: the label is in the order (x, y, z) 0:empty, 255: invalid
            while in our model, the voxe ls in the oder (z, y, x)
            works for the focal loss 
            """
            occ_gts = occ_gts.transpose(2, 1, 0).astype(np.int32)
            occ_gts = occ_gts -1 
            occ_gts[occ_gts==-1] = self.occupancy_classes  # 19: means background
            occ_gts[occ_gts==254] = 255

        results['occ_gts'] = occ_gts
        return results

@PIPELINES.register_module()
class LoadFlowGT(object):
    """load occupancy flow GT data
       flow_type: 2D, only x and y direction flows are given
    """
    def __init__(self, flow_type='2D'):
        self.flow_type = flow_type

    def __call__(self, results):
        flow_gt_path = results['flow_gt_path']
        flow_gts = np.load(flow_gt_path)  # (n, 2)
        results['flow_gts'] = flow_gts
        return results
