import torch
import unittest
from .base_test_case import BaseTestCase


depth_shape = [6, 160, 32, 88]
feat_shape = [6, 32, 88, 128]
ranks_depth_shape = [699899]
ranks_feat_shape = [699899]
ranks_bev_shape = [699899]
interval_starts_shape = [29351]
interval_lengths_shape = [29351]
output_shape = [1, 200, 200, 128]


def get_lidar_coor(sensor2ego, cam2imgs, post_rots, post_trans, bda):
    B, N, _, _ = sensor2ego.shape

    H_in, W_in = 512, 1408
    H_feat, W_feat = H_in // 16, W_in // 16
    d = (
        torch.arange(1.0, 161.0, 1.0, dtype=torch.float)
        .view(-1, 1, 1)
        .expand(-1, H_feat, W_feat)
    )
    x = (
        torch.linspace(0, W_in - 1, W_feat, dtype=torch.float)
        .view(1, 1, W_feat)
        .expand(d.shape[0], H_feat, W_feat)
    )
    y = (
        torch.linspace(0, H_in - 1, H_feat, dtype=torch.float)
        .view(1, H_feat, 1)
        .expand(d.shape[0], H_feat, W_feat)
    )
    frustum = torch.stack((x, y, d), -1)

    # post-transformation
    # B x N x D x H x W x 3
    points = frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
    points = (
        torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
    )

    # cam_to_ego
    points = torch.cat(
        (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5
    )
    combine = sensor2ego[:, :, :3, :3].matmul(torch.inverse(cam2imgs))
    points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
    points += sensor2ego[:, :, :3, 3].view(B, N, 1, 1, 1, 3)
    points = bda.view(B, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
    return points


def bev_pool_prepare():
    sensor2keyegos = torch.tensor(
        [
            [
                [
                    [8.2076e-01, -3.4143e-04, 5.7128e-01, 1.5239e00],
                    [-5.7127e-01, 3.2195e-03, 8.2075e-01, 4.9463e-01],
                    [-2.1195e-03, -9.9999e-01, 2.4474e-03, 1.5093e00],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                ],
                [
                    [6.2286e-03, -5.5937e-03, 9.9996e-01, 1.7707e00],
                    [-9.9998e-01, -8.5861e-04, 6.2239e-03, 1.6166e-02],
                    [8.2376e-04, -9.9998e-01, -5.5989e-03, 1.5124e00],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                ],
                [
                    [-8.3232e-01, 4.2987e-05, 5.5429e-01, 1.6938e00],
                    [-5.5422e-01, 1.6328e-02, -8.3221e-01, -4.9285e-01],
                    [-9.0865e-03, -9.9987e-01, -1.3567e-02, 1.4986e00],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                ],
                [
                    [9.4656e-01, 8.7274e-03, -3.2241e-01, 1.4300e00],
                    [3.2251e-01, -1.4109e-02, 9.4646e-01, 4.8685e-01],
                    [3.7113e-03, -9.9986e-01, -1.6170e-02, 1.5984e00],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                ],
                [
                    [-2.6211e-04, -1.6699e-02, -9.9986e-01, 3.3118e-01],
                    [9.9999e-01, -4.0898e-03, -1.9385e-04, 1.7097e-03],
                    [-4.0860e-03, -9.9985e-01, 1.6700e-02, 1.5849e00],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                ],
                [
                    [-9.3536e-01, 1.5928e-02, -3.5334e-01, 1.2277e00],
                    [3.5353e-01, 1.1332e-02, -9.3536e-01, -4.8050e-01],
                    [-1.0894e-02, -9.9981e-01, -1.6230e-02, 1.5665e00],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                ],
            ]
        ],
        device="cuda",
    )
    intrins = torch.tensor(
        [
            [
                [
                    [1.2726e03, 0.0000e00, 8.2662e02],
                    [0.0000e00, 1.2726e03, 4.7975e02],
                    [0.0000e00, 0.0000e00, 1.0000e00],
                ],
                [
                    [1.2664e03, 0.0000e00, 8.1627e02],
                    [0.0000e00, 1.2664e03, 4.9151e02],
                    [0.0000e00, 0.0000e00, 1.0000e00],
                ],
                [
                    [1.2608e03, 0.0000e00, 8.0797e02],
                    [0.0000e00, 1.2608e03, 4.9533e02],
                    [0.0000e00, 0.0000e00, 1.0000e00],
                ],
                [
                    [1.2567e03, 0.0000e00, 7.9211e02],
                    [0.0000e00, 1.2567e03, 4.9278e02],
                    [0.0000e00, 0.0000e00, 1.0000e00],
                ],
                [
                    [8.0922e02, 0.0000e00, 8.2922e02],
                    [0.0000e00, 8.0922e02, 4.8178e02],
                    [0.0000e00, 0.0000e00, 1.0000e00],
                ],
                [
                    [1.2595e03, 0.0000e00, 8.0725e02],
                    [0.0000e00, 1.2595e03, 5.0120e02],
                    [0.0000e00, 0.0000e00, 1.0000e00],
                ],
            ]
        ],
        device="cuda",
    )
    post_rots = torch.tensor(
        [
            [
                [
                    [0.4400, 0.0000, 0.0000],
                    [0.0000, 0.4400, 0.0000],
                    [0.0000, 0.0000, 1.0000],
                ],
                [
                    [0.4400, 0.0000, 0.0000],
                    [0.0000, 0.4400, 0.0000],
                    [0.0000, 0.0000, 1.0000],
                ],
                [
                    [0.4400, 0.0000, 0.0000],
                    [0.0000, 0.4400, 0.0000],
                    [0.0000, 0.0000, 1.0000],
                ],
                [
                    [0.4400, 0.0000, 0.0000],
                    [0.0000, 0.4400, 0.0000],
                    [0.0000, 0.0000, 1.0000],
                ],
                [
                    [0.4400, 0.0000, 0.0000],
                    [0.0000, 0.4400, 0.0000],
                    [0.0000, 0.0000, 1.0000],
                ],
                [
                    [0.4400, 0.0000, 0.0000],
                    [0.0000, 0.4400, 0.0000],
                    [0.0000, 0.0000, 1.0000],
                ],
            ]
        ],
        device="cuda",
    )
    post_trans = torch.tensor(
        [
            [
                [0.0, -140.0, 0.0],
                [0.0, -140.0, 0.0],
                [0.0, -140.0, 0.0],
                [0.0, -140.0, 0.0],
                [0.0, -140.0, 0.0],
                [0.0, -140.0, 0.0],
            ]
        ],
        device="cuda",
    )
    bda = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device="cuda"
    )
    coor = get_lidar_coor(sensor2keyegos, intrins, post_rots, post_trans, bda)
    B, N, D, H, W, _ = coor.shape
    grid_lower_bound = torch.tensor([-100.0, -100.0, -5.0000])
    grid_interval = torch.tensor([1.0, 1.0, 8.0000])
    grid_size = [200, 200, 1]
    num_points = B * N * D * H * W
    # record the index of selected points for acceleration purpose
    ranks_depth = torch.arange(0, num_points, dtype=torch.int, device=coor.device)
    ranks_feat = torch.arange(0, num_points // D, dtype=torch.int, device=coor.device)
    ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
    ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
    # convert coordinate into the voxel space
    coor = (coor - grid_lower_bound.to(coor)) / grid_interval.to(coor)
    coor = coor.long().view(num_points, 3)
    batch_idx = (
        torch.arange(0, B)
        .reshape(B, 1)
        .expand(B, num_points // B)
        .reshape(num_points, 1)
        .to(coor)
    )
    coor = torch.cat((coor, batch_idx), 1)

    # filter out points that are outside box
    kept = (
        (coor[:, 0] >= 0)
        & (coor[:, 0] < grid_size[0])
        & (coor[:, 1] >= 0)
        & (coor[:, 1] < grid_size[1])
        & (coor[:, 2] >= 0)
        & (coor[:, 2] < grid_size[2])
    )

    coor, ranks_depth, ranks_feat = coor[kept], ranks_depth[kept], ranks_feat[kept]
    # get tensors from the same voxel next to each other
    ranks_bev = coor[:, 3] * grid_size[0] * grid_size[1]
    ranks_bev += coor[:, 2] * grid_size[0] * grid_size[1]
    ranks_bev += coor[:, 1] * grid_size[0] + coor[:, 0]
    order = ranks_bev.argsort()
    ranks_bev, ranks_depth, ranks_feat = (
        ranks_bev[order],
        ranks_depth[order],
        ranks_feat[order],
    )

    kept = torch.ones(ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
    kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
    interval_starts = torch.where(kept)[0].int()
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]

    ranks_bev = ranks_bev.float().contiguous()
    ranks_depth = ranks_depth.float().contiguous()
    ranks_feat = ranks_feat.float().contiguous()
    interval_starts = interval_starts.float().contiguous()
    interval_lengths = interval_lengths.float().contiguous()
    return ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths


class BEVPoolV2TestCase(BaseTestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func_name = args[0]

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(0)
        print("\n" + "#" * 20 + f" Running {cls.__name__} " + "#" * 20)
        (
            ranks_bev,
            ranks_depth,
            ranks_feat,
            interval_starts,
            interval_lengths,
        ) = bev_pool_prepare()
        cls.input_data = dict(
            depth=torch.randn(*depth_shape, device="cuda"),
            feat=torch.randn(*feat_shape, device="cuda"),
            ranks_depth=ranks_depth,
            ranks_feat=ranks_feat,
            ranks_bev=ranks_bev,
            interval_starts=interval_starts,
            interval_lengths=interval_lengths,
        )

    @classmethod
    def tearDownClass(cls):
        del cls.input_data
        torch.cuda.empty_cache()

    def createInputs(self):
        if self.fp16:
            inputs_pth = self.getInputs()
            self.inputs_pth_fp16 = {
                key: val.half() if key in ["depth", "feat"] else val
                for key, val in inputs_pth.items()
            }
            self.inputs_np_fp16 = {
                key: val.cpu().numpy() for key, val in self.inputs_pth_fp16.items()
            }
        elif self.int8:
            self.inputs_pth_int8 = self.getInputs()
            self.inputs_np_int8 = {
                key: val.cpu().numpy() for key, val in self.inputs_pth_int8.items()
            }
        else:
            self.inputs_pth_fp32 = self.getInputs()
            self.inputs_np_fp32 = {
                key: val.cpu().numpy() for key, val in self.inputs_pth_fp32.items()
            }

    def setUp(self):
        params = [{"out_height": 200, "out_width": 200}]
        BaseTestCase.__init__(
            self,
            "bev_pool_v2",
            input_shapes={
                "depth": depth_shape,
                "feat": feat_shape,
                "ranks_depth": ranks_depth_shape,
                "ranks_feat": ranks_feat_shape,
                "ranks_bev": ranks_bev_shape,
                "interval_starts": interval_starts_shape,
                "interval_lengths": interval_lengths_shape,
            },
            output_shapes={"output": output_shape},
            params=params,
            device="cuda",
            func_name=self.func_name,
        )
        self.buildEngine(opset_version=13)

    def tearDown(self):
        del self

    def getInputs(self):
        inputs = {}
        for key in self.input_shapes:
            inputs[key] = self.__class__.input_data[key].to(torch.device(self.device))
        return inputs

    def fp32_case(self, delta=None):
        delta = self.delta if delta is None else delta
        for dic in self.models:
            output_pth = self.torchForward(dic["model_pth_fp32"], fp16=False)
            output_trt, t = self.engineForward(dic["engine_fp32"], fp16=False)
            cost = self.getCost(output_trt, output_pth)
            self.assertLessEqual(cost, delta)

    def fp16_case(self, delta=None):
        delta = self.delta if delta is None else delta
        for dic in self.models:
            output_pth = self.torchForward(dic["model_pth_fp16"], fp16=True)
            output_trt, t = self.engineForward(dic["engine_fp16"], fp16=True)
            cost = self.getCost(output_trt, output_pth)
            self.assertLessEqual(cost, delta)

    def int8_case(self, delta=None):
        delta = self.delta if delta is None else delta
        for dic in self.models:
            output_pth = self.torchForward(dic["model_pth_int8"], int8=True)
            output_trt, t = self.engineForward(dic["engine_int8"], int8=True)
            cost = self.getCost(output_trt, output_pth)
            self.assertLessEqual(cost, delta)

    def test_fp32(self):
        self.fp32_case(1e-5)

    def test_fp16(self):
        self.fp16_case(3e-3)

    def test_int8(self):
        self.int8_case(0.3)


class BEVPoolV2TestCase2(BaseTestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func_name = args[0]

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(0)
        print("\n" + "#" * 20 + f" Running {cls.__name__} " + "#" * 20)
        (
            ranks_bev,
            ranks_depth,
            ranks_feat,
            interval_starts,
            interval_lengths,
        ) = bev_pool_prepare()
        cls.input_data = dict(
            depth=torch.randn(*depth_shape, device="cuda"),
            feat=torch.randn(*feat_shape, device="cuda"),
            ranks_depth=ranks_depth,
            ranks_feat=ranks_feat,
            ranks_bev=ranks_bev,
            interval_starts=interval_starts,
            interval_lengths=interval_lengths,
        )

    @classmethod
    def tearDownClass(cls):
        del cls.input_data
        torch.cuda.empty_cache()

    def createInputs(self):
        if self.fp16:
            inputs_pth = self.getInputs()
            self.inputs_pth_fp16 = {
                key: val.half() if key in ["depth", "feat"] else val
                for key, val in inputs_pth.items()
            }
            self.inputs_np_fp16 = {
                key: val.cpu().numpy() for key, val in self.inputs_pth_fp16.items()
            }
        elif self.int8:
            self.inputs_pth_int8 = self.getInputs()
            self.inputs_np_int8 = {
                key: val.cpu().numpy() for key, val in self.inputs_pth_int8.items()
            }
        else:
            self.inputs_pth_fp32 = self.getInputs()
            self.inputs_np_fp32 = {
                key: val.cpu().numpy() for key, val in self.inputs_pth_fp32.items()
            }

    def setUp(self):
        params = [{"out_height": 200, "out_width": 200}]
        BaseTestCase.__init__(
            self,
            "bev_pool_v2_2",
            input_shapes={
                "depth": depth_shape,
                "feat": feat_shape,
                "ranks_depth": ranks_depth_shape,
                "ranks_feat": ranks_feat_shape,
                "ranks_bev": ranks_bev_shape,
                "interval_starts": interval_starts_shape,
                "interval_lengths": interval_lengths_shape,
            },
            output_shapes={"output": output_shape},
            params=params,
            device="cuda",
            func_name=self.func_name,
        )
        self.buildEngine(opset_version=13)

    def tearDown(self):
        del self

    def getInputs(self):
        inputs = {}
        for key in self.input_shapes:
            inputs[key] = self.__class__.input_data[key].to(torch.device(self.device))
        return inputs

    def fp32_case(self, delta=None):
        delta = self.delta if delta is None else delta
        for dic in self.models:
            output_pth = self.torchForward(dic["model_pth_fp32"], fp16=False)
            output_trt, t = self.engineForward(dic["engine_fp32"], fp16=False)
            cost = self.getCost(output_trt, output_pth)
            self.assertLessEqual(cost, delta)

    def fp16_case(self, delta=None):
        delta = self.delta if delta is None else delta
        for dic in self.models:
            output_pth = self.torchForward(dic["model_pth_fp16"], fp16=True)
            output_trt, t = self.engineForward(dic["engine_fp16"], fp16=True)
            cost = self.getCost(output_trt, output_pth)
            self.assertLessEqual(cost, delta)

    def int8_case(self, delta=None):
        delta = self.delta if delta is None else delta
        for dic in self.models:
            output_pth = self.torchForward(dic["model_pth_int8"], int8=True)
            output_trt, t = self.engineForward(dic["engine_int8"], int8=True)
            cost = self.getCost(output_trt, output_pth)
            self.assertLessEqual(cost, delta)

    def test_fp32(self):
        self.fp32_case(1e-5)

    def test_fp16(self):
        self.fp16_case(3e-3)

    def test_int8(self):
        self.int8_case(0.3)
