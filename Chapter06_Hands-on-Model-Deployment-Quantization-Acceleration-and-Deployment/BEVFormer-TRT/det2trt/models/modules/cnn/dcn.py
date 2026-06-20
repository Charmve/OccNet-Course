import torch
import torch.nn as nn
import pytorch_quantization.nn as quant_nn
from mmcv.ops.modulated_deform_conv import (
    ModulatedDeformConv2dPack,
    ModulatedDeformConv2d,
)
from mmcv.cnn.bricks.registry import CONV_LAYERS
from det2trt.models.functions import (
    modulated_deformable_conv2d,
    modulated_deformable_conv2d2,
)


@CONV_LAYERS.register_module("DCNv2Q")
class ModulatedDeformConv2dPackQ(ModulatedDeformConv2dPack):
    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConv2dPack, self).__init__(*args, **kwargs)
        self.conv_offset = quant_nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.init_weights()


@CONV_LAYERS.register_module("DCNv2P")
class ModulatedDeformConv2dPackPlugin(ModulatedDeformConv2d):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConv2dPackPlugin, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.init_weights()

    def init_weights(self):
        super(ModulatedDeformConv2dPackPlugin, self).init_weights()
        if hasattr(self, "conv_offset"):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deformable_conv2d(
            input=x,
            offset=offset,
            mask=mask,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            deform_groups=self.deform_groups,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, ModulatedDeformConvPack
            # loads previous benchmark models.
            if (
                prefix + "conv_offset.weight" not in state_dict
                and prefix[:-1] + "_offset.weight" in state_dict
            ):
                state_dict[prefix + "conv_offset.weight"] = state_dict.pop(
                    prefix[:-1] + "_offset.weight"
                )
            if (
                prefix + "conv_offset.bias" not in state_dict
                and prefix[:-1] + "_offset.bias" in state_dict
            ):
                state_dict[prefix + "conv_offset.bias"] = state_dict.pop(
                    prefix[:-1] + "_offset.bias"
                )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


@CONV_LAYERS.register_module("DCNv2P2")
class ModulatedDeformConv2dPackPlugin2(ModulatedDeformConv2dPackPlugin):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deformable_conv2d2(
            input=x,
            offset=offset,
            mask=mask,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            deform_groups=self.deform_groups,
        )


@CONV_LAYERS.register_module("DCNv2PQ")
class ModulatedDeformConv2dPackPluginQ(ModulatedDeformConv2dPackPlugin):
    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConv2dPackPlugin, self).__init__(*args, **kwargs)
        self.conv_offset = quant_nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.init_weights()


@CONV_LAYERS.register_module("DCNv2PQ2")
class ModulatedDeformConv2dPackPluginQ2(ModulatedDeformConv2dPackPlugin2):
    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConv2dPackPlugin2, self).__init__(*args, **kwargs)
        self.conv_offset = quant_nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.init_weights()
