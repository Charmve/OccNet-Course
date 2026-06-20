import torch
import torch.nn as nn
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK
from mmcv.utils import deprecated_api_warning
from mmcv.cnn import Linear, build_activation_layer
from ..utils import LINEAR_LAYERS


class ReLUAddZeros(nn.Module):
    def __init__(self):
        super(ReLUAddZeros, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + torch.zeros_like(x))


@FEEDFORWARD_NETWORK.register_module()
class FFNTRT(FFN):
    @deprecated_api_warning(
        {"dropout": "ffn_drop", "add_residual": "add_identity"}, cls_name="FFNQ"
    )
    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        linear_cfg=None,
        **kwargs,
    ):
        super(FFN, self).__init__(init_cfg)
        assert num_fcs >= 2, "num_fcs should be no less " f"than 2. got {num_fcs}."
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        if linear_cfg is None:
            linear_cfg = dict(type="Linear")
        linear = LINEAR_LAYERS.get(linear_cfg["type"])

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    linear(in_channels, feedforward_channels),
                    # TODO: Waiting bug of TensorRT-8.5.1.7 fixed
                    self.activate if linear == nn.Linear else ReLUAddZeros(),
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
        )
        self.add_identity = add_identity
