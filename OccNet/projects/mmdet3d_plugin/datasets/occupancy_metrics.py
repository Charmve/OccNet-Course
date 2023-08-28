import numpy as np

class SSCMetrics:
    def __init__(self, n_classes=17, eval_far=False, eval_near=False,
                 near_distance=10, far_distance=30, occ_type='normal'):
        """
        non-empty class: 0-15
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 
        'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
        'driveable_surface', 'other_flat', 'sidewalk',
        'terrain', 'manmade,', 'vegetation'
        16: empty
        """
        self.n_classes = n_classes
        self.empty_label = n_classes
        print('class num:', self.n_classes)
        self.point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
        if occ_type == 'normal':
            self.occupancy_size = [0.5, 0.5, 0.5]
        elif occ_type == 'fine':
            self.occupancy_size = [0.25, 0.25, 0.25]
        elif occ_type == 'coarse':
            self.occupancy_size = [1.0, 1.0, 1.0]
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])

        self.eval_far = eval_far
        self.eval_near = eval_near
        self.far_distance = far_distance 
        self.near_distance = near_distance
        self.hist = np.zeros((self.n_classes, self.n_classes))

        if eval_far or self.eval_near:
            self.obtain_masked_distanced_voxel()

    def hist_info(self, n_cl, pred, gt):
        """
        Confusion Matrix: n*n
        row: reference
        col: prediction
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def add_batch(self, y_pred, y_true, flow_pred=None, flow_true=None, visible_mask=None):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        if visible_mask is not None:
            y_pred = y_pred[visible_mask==1]
            y_true = y_true[visible_mask==1]
        
        batch_hist, _, _ = self.hist_info(self.n_classes, y_pred, y_true)
        self.hist = self.hist + batch_hist

    def get_stats(self):
        miou = np.diag(self.hist) / (self.hist.sum(1) + self.hist.sum(0) - np.diag(self.hist)+ 1e-6)*100.0
        completion_tp = np.sum(self.hist[:-1, :-1])
        completion_fp = np.sum(self.hist[-1, :-1])
        completion_fn = np.sum(self.hist[:-1, -1])

        if completion_tp != 0:
            precision = completion_tp / (completion_tp + completion_fp)
            recall = completion_tp / (completion_tp + completion_fn)
            iou = completion_tp / (completion_tp + completion_fp + completion_fn)*100.0
        else:
            precision, recall, iou = 0, 0, 0

        iou_ssc = miou[:self.n_classes-1]  # exclude the empty voxel

        return {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "iou_ssc": iou_ssc,  # class IOU
            "miou": np.mean(iou_ssc),
        }
    
    def obtain_masked_distanced_voxel(self):
        index_x  = np.arange(self.occ_xdim)
        index_y  = np.arange(self.occ_ydim)
        index_z  = np.arange(self.occ_zdim)
        z, y, x = np.meshgrid(index_z, index_y, index_x, indexing='ij')
        index_xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        points_x = (index_xyz[:, 0] + 0.5) / self.occ_xdim * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0]
        points_y = (index_xyz[:, 1] + 0.5) / self.occ_ydim * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1]
        points_z = (index_xyz[:, 2] + 0.5) / self.occ_zdim * (self.point_cloud_range[5] - self.point_cloud_range[2]) + self.point_cloud_range[2]
        points = np.concatenate([points_x.reshape(-1, 1), points_y.reshape(-1, 1), points_z.reshape(-1, 1)], axis=-1)

        points_distance = np.linalg.norm(points[:, :2], axis=-1)  
        self.far_voxel_mask = points_distance > self.far_distance
        self.near_voxel_mask = points_distance < self.near_distance