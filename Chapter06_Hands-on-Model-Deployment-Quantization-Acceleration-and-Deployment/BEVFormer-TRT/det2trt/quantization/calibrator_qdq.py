import torch
import mmcv
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor


def init_quant_desc(calibrator, per_channel_quantization=False):
    assert calibrator in ["max", "histogram"]
    input_kw = {"calib_method": calibrator}
    weight_kw = {"calib_method": calibrator}
    if calibrator == "max":
        if not per_channel_quantization:
            input_kw["axis"] = None
            weight_kw["axis"] = None

    quant_desc_input = QuantDescriptor(**input_kw)
    quant_desc_weight = QuantDescriptor(**weight_kw)

    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
    quant_nn.QuantConvTranspose2d.set_default_quant_desc_weight(quant_desc_weight)
    quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)


def calibrator_qdq(
    model,
    calibrator,
    loader,
    per_channel_quantization=False,
    data_length=500,
    samples_per_gpu=16,
    **kwargs
):
    init_quant_desc(
        calibrator=calibrator, per_channel_quantization=per_channel_quantization
    )

    batches = (data_length + samples_per_gpu - 1) // samples_per_gpu

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        print("Calibrating...")
        prog_bar = mmcv.ProgressBar(batches * samples_per_gpu)
        for i, data in enumerate(loader):
            if i >= batches:
                break
            model(**data, **kwargs)
            for _ in range(samples_per_gpu):
                prog_bar.update()

        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(method="percentile", percentile=99.99)

        model.cuda()

    return model
