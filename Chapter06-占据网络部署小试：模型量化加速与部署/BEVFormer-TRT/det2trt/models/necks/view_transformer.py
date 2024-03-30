import torch
from mmdet.models import NECKS

from third_party.bev_mmdet3d import LSSViewTransformer


@NECKS.register_module()
class LSSViewTransformerTRT(LSSViewTransformer):
    r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
        `paper <https://arxiv.org/abs/2211.17111>`

    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
        sid (bool): Whether to use Spacing Increasing Discretization (SID)
            depth distribution as `STS: Surround-view Temporal Stereo for
            Multi-view 3D Detection`.
        collapse_z (bool): Whether to collapse in z direction.
    """

    def __init__(self, *args, **kwargs):
        super(LSSViewTransformerTRT, self).__init__(*args, **kwargs)
