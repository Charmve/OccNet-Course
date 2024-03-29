
## Chapter06-占据网络部署小试

### Overview

0. 模型量化部署

![QAT、PTQ简介](https://developer.horizon.ai/api/v1/fileData/horizon_j5_open_explorer_v1_1_33_cn_doc/_images/qat_ptq_contrast.png)

- 训练后量化（PTQ） [地平线J5](https://developer.horizon.ai/api/v1/fileData/horizon_j5_open_explorer_v1_1_33_cn_doc/oe_mapper/source/ptq/ptq.html) ｜ [英伟达Orin]()
- 量化感知训练（QAT） [地平线J5](https://developer.horizon.ai/api/v1/fileData/horizon_j5_open_explorer_v1_1_33_cn_doc/plugin/source/index.html) ｜ [英伟达Orin]()

BEV部署：https://developer.horizon.cc/api/v1/fileData/horizon_j5_reference_package_release/index.html 2.BEV算法

1. BEVFormer->VoxFormer 部署

https://github.com/fundamentalvision/BEVFormer


2. [BEVFusion CUDA部署](./CUDA-BEVFusion)

[BEVFusion: 基于统一BEV表征的多任务多传感器融合](https://zhuanlan.zhihu.com/p/521821929)

### Pipeline overview
![pipeline](assets/pipeline.png)

### GetStart
```
$ cd OccNet-Course/Chapter06-占据网络部署小试：模型量化加速与部署/
```
- For each specific task please refer to the readme in the sub-folder.

#### 3D Sparse Convolution
A tiny inference engine for [3d sparse convolutional networks](https://github.com/tianweiy/CenterPoint/blob/master/det3d/models/backbones/scn.py) using int8/fp16.
- **Tiny Engine:** Tiny Lidar-Backbone inference engine independent of TensorRT.
- **Flexible:** Build execution graph from ONNX.
- **Easy To Use:** Simple interface and onnx export solution.
- **High Fidelity:** Low accuracy drop on nuScenes validation.
- **Low Memory:** 422MB@SCN FP16, 426MB@SCN INT8.
- **Compact:** Based on the CUDA kernels and independent of cutlass.

#### CUDA BEVFusion
CUDA & TensorRT solution for [BEVFusion](https://arxiv.org/abs/2205.13542) inference, including:
- **Camera Encoder**: ResNet50 and finetuned BEV pooling with TensorRT and onnx export solution.
- **Lidar Encoder**: Tiny Lidar-Backbone inference independent of TensorRT and onnx export solution.
- **Feature Fusion**: Camera & Lidar feature fuser with TensorRT and onnx export solution.
- **Pre/Postprocess**: Interval precomputing, lidar voxelization, feature decoder with CUDA kernels.
- **Easy To Use**: Preparation, inference, evaluation all in one to reproduce torch Impl accuracy.
- **PTQ**: Quantization solutions for [mmdet3d/spconv](https://github.com/mit-han-lab/bevfusion/tree/main/mmdet3d/ops/spconv), Easy to understand.

#### CUDA BEVFormer

cd workspace/OccNet-Course/Chapter06-占据网络部署小试：模型量化加速与部署/
git clone https://github.com/DerryHub/BEVFormer_tensorrt BEVFormer_TRT

#### libs

##### cuOSD(CUDA On-Screen Display Library)
Draw all elements using a single CUDA kernel.
- **Line:** Plotting lines by interpolation(Nearest or Linear).
- **RotateBox:** Supports drawn with different border colors and fill colors.
- **Circle:** Supports drawn with different border colors and fill colors.
- **Rectangle:** Supports drawn with different border colors and fill colors.
- **Text:** Supports [stb_truetype](https://github.com/nothings/stb/blob/master/stb_truetype.h) and [pango-cairo](https://pango.gnome.org/) backends, allowing fonts to be read via TTF or using font-family.
- **Arrow:** Combination of arrows by 3 lines.
- **Point:** Plotting points by interpolation(Nearest or Linear).
- **Clock:** Time plotting based on text support

##### cuPCL(CUDA Point Cloud Library)
Provide several GPU accelerated Point Cloud operations with high accuracy and high performance at the same time: cuICP, cuFilter, cuSegmentation, cuOctree, cuCluster, cuNDT, Voxelization(incoming).
- **cuICP:** CUDA accelerated iterative corresponding point vertex cloud(point-to-point) registration implementation.
- **cuFilter:** Support CUDA accelerated features: PassThrough and VoxelGrid.
- **cuSegmentation:** Support CUDA accelerated features: RandomSampleConsensus with a plane model.
- **cuOctree:** Support CUDA accelerated features: Approximate Nearest Search and Radius Search.
- **cuCluster:** Support CUDA accelerated features: Cluster based on the distance among points.
- **cuNDT:** CUDA accelerated 3D Normal Distribution Transform registration implementation for point cloud data.

##### YUVToRGB(CUDA Conversion)
YUV to RGB conversion. Combine Resize/Padding/Conversion/Normalization into a single kernel function.
- **Most of the time, it can be bit-aligned with OpenCV.**
    - It will give an exact result when the scaling factor is a rational number.
    - Better performance is usually achieved when the stride can divide by 4.
- Supported Input Format:
    - **NV12BlockLinear**
    - **NV12PitchLinear**
    - **YUV422Packed_YUYV**
- Supported Interpolation methods:
    - **Nearest**
    - **Bilinear**
- Supported Output Data Type:
    - **Uint8**
    - **Float32**
    - **Float16**
- Supported Output Layout:
    - **CHW_RGB/BGR**
    - **HWC_RGB/BGR**
    - **CHW16/32/4/RGB/BGR for DLA input**
- Supported Features:
    - **Resize**
    - **Padding**
    - **Conversion**
    - **Normalization**

### Thanks
This project makes use of a number of awesome open source libraries, including:

- [stb_image](https://github.com/nothings/stb) for PNG and JPEG support
- [pybind11](https://github.com/pybind/pybind11) for seamless C++ / Python interop
- and others! See the dependencies folder.

Many thanks to the authors of these brilliant projects!


### 扩充材料

- [模型部署入门教程（五）：ONNX 模型的修改与调试](https://zhuanlan.zhihu.com/p/516920606)
- [25FPS！全网首发 | 英伟达开放BEVFusion部署源代码，边缘端实时运行](https://mp.weixin.qq.com/s/6BWohe2FxRN8E-yyp_32fg)