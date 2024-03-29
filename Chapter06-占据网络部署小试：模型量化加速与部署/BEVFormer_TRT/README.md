# Deployment of BEV 3D Detection on TensorRT

This repository is a deployment project of BEV 3D Detection (including [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [BEVDet](https://github.com/HuangJunJie2017/BEVDet)) on [TensorRT](https://developer.nvidia.com/tensorrt), supporting **FP32/FP16/INT8** inference. Meanwhile, in order to improve the inference speed of BEVFormer on TensorRT, this project implements some TensorRT Ops that support [**nv_half**](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__ARITHMETIC.html#group__CUDA__MATH____HALF__ARITHMETIC),  [**nv_half2**](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__ARITHMETIC.html#group__CUDA__MATH____HALF2__ARITHMETIC) and **INT8**. With the accuracy almost unaffected, the inference speed of the **BEVFormer base** can be increased by more than **four times**, the engine size can be reduced by more than **90%**, and the GPU memory usage can be saved by more than **80%**. In addition, the project also supports common 2D object detection models in [MMDetection](https://github.com/open-mmlab/mmdetection), which support **INT8 Quantization** and **TensorRT Deployment** with a small number of code changes.

## Benchmarks

### BEVFormer


#### BEVFormer PyTorch

|                            Model                             |   Data   | Batch Size |          NDS/mAP          | FPS  | Size (MB) | Memory (MB) |  Device  |
| :----------------------------------------------------------: | :------: | :--------: | :-----------------------: | :--: | :-------: | :---------: | :------: |
| BEVFormer tiny<br />[download](https://drive.google.com/file/d/1VJ_WgA9UZZ-DJr5kUFYAFfIC0UF1oCeQ/view?usp=share_link) | NuScenes |     1      | NDS: 0.354<br/>mAP: 0.252 | 15.9 |    383    |    2167     | RTX 3090 |
| BEVFormer small<br />[download](https://drive.google.com/file/d/1n5_Ca2bqkCY3Q19M5U7futKUT1pEyQZ7/view?usp=sharing) | NuScenes |     1      | NDS: 0.478<br/>mAP: 0.370 | 5.1  |    680    |    3147     | RTX 3090 |
| BEVFormer base<br />[download](https://drive.google.com/file/d/1UMN35jPHeJbVK-8P4HhxIZl341KS1bDU/view?usp=share_link) | NuScenes |     1      | NDS: 0.517<br/>mAP: 0.416 | 2.4  |    265    |    5435     | RTX 3090 |

#### BEVFormer TensorRT with MMDeploy Plugins (Only Support FP32)

|         Model         |   Data   | Batch Size | Float/Int |     Quantization Method     |          NDS/mAP          |     FPS      |  Size (MB)  |  Memory (MB)  |  Device  |
| :-------------------: | :------: | :--------: | :-------: | :-------------------------: | :-----------------------: | :----------: | :---------: | :-----------: | :------: |
|    BEVFormer tiny     | NuScenes |     1      |   FP32    |              -              | NDS: 0.354<br/>mAP: 0.252 |  37.9 (x1)   |  136 (x1)   |   2159 (x1)   | RTX 3090 |
|    BEVFormer tiny     | NuScenes |     1      |   FP16    |              -              | NDS: 0.354<br/>mAP: 0.252 | 69.2 (x1.83) | 74 (x0.54)  | 1729 (x0.80)  | RTX 3090 |
|    BEVFormer tiny     | NuScenes |     1      | FP32/INT8 | PTQ entropy<br />per-tensor | NDS: 0.353<br/>mAP: 0.249 | 65.1 (x1.72) | 58 (x0.43)  | 1737 (x0.80)  | RTX 3090 |
|    BEVFormer tiny     | NuScenes |     1      | FP16/INT8 | PTQ entropy<br />per-tensor | NDS: 0.353<br/>mAP: 0.249 | 70.7 (x1.87) | 54 (x0.40)  | 1665 (x0.77)  | RTX 3090 |
|    BEVFormer small    | NuScenes |     1      |   FP32    |              -              | NDS: 0.478<br/>mAP: 0.370 |   6.6 (x1)   |  245 (x1)   |   4663 (x1)   | RTX 3090 |
|    BEVFormer small    | NuScenes |     1      |   FP16    |              -              | NDS: 0.478<br/>mAP: 0.370 | 12.8 (x1.94) | 126 (x0.51) | 3719 (x0.80)  | RTX 3090 |
|    BEVFormer small    | NuScenes |     1      | FP32/INT8 | PTQ entropy<br />per-tensor | NDS: 0.476<br/>mAP: 0.367 | 8.7 (x1.32)  | 158 (x0.64) | 4079 (x0.87)  | RTX 3090 |
|    BEVFormer small    | NuScenes |     1      | FP16/INT8 | PTQ entropy<br />per-tensor | NDS: 0.477<br/>mAP: 0.368 | 13.3 (x2.02) | 106 (x0.43) | 3441 (x0.74)  | RTX 3090 |
| BEVFormer base **\*** | NuScenes |     1      |   FP32    |              -              | NDS: 0.517<br/>mAP: 0.416 |   1.5 (x1)   |  1689 (x1)  |  13893 (x1)   | RTX 3090 |
|    BEVFormer base     | NuScenes |     1      |   FP16    |              -              | NDS: 0.517<br/>mAP: 0.416 | 1.8 (x1.20)  | 849 (x0.50) | 11865 (x0.85) | RTX 3090 |
| BEVFormer base **\*** | NuScenes |     1      | FP32/INT8 | PTQ entropy<br />per-tensor | NDS: 0.516<br/>mAP: 0.414 | 1.8 (x1.20)  | 426 (x0.25) | 12429 (x0.89) | RTX 3090 |
| BEVFormer base **\*** | NuScenes |     1      | FP16/INT8 | PTQ entropy<br />per-tensor | NDS: 0.515<br/>mAP: 0.414 | 2.2 (x1.47)  | 244 (x0.14) | 11011 (x0.79) | RTX 3090 |

**\*** `Out of Memory` when onnx2trt with TensorRT-8.5.1.7, but they convert successfully with TensorRT-8.4.3.1. So the version of these engines is TensorRT-8.4.3.1.

#### BEVFormer TensorRT with Custom Plugins (Support nv_half, nv_half2 and int8)

**FP16 Plugins with nv_half**

|      Model      |   Data   | Batch Size | Float/Int |     Quantization Method     |          NDS/mAP          |  FPS/Improve  |  Size (MB)  | Memory (MB)  |  Device  |
| :-------------: | :------: | :--------: | :-------: | :-------------------------: | :-----------------------: | :-----------: | :---------: | :----------: | :------: |
| BEVFormer tiny  | NuScenes |     1      |   FP32    |              -              | NDS: 0.354<br/>mAP: 0.252 | 40.0 (x1.06)  | 135 (x0.99) | 1693 (x0.78) | RTX 3090 |
| BEVFormer tiny  | NuScenes |     1      |   FP16    |              -              | NDS: 0.355<br/>mAP: 0.252 | 81.2 (x2.14)  | 70 (x0.51)  | 1203 (x0.56) | RTX 3090 |
| BEVFormer tiny  | NuScenes |     1      | FP32/INT8 | PTQ entropy<br />per-tensor | NDS: 0.351<br/>mAP: 0.249 | 90.1 (x2.38)  | 58 (x0.43)  | 1105 (x0.51) | RTX 3090 |
| BEVFormer tiny  | NuScenes |     1      | FP16/INT8 | PTQ entropy<br />per-tensor | NDS: 0.351<br/>mAP: 0.249 | 107.4 (x2.83) | 52 (x0.38)  | 1095 (x0.51) | RTX 3090 |
| BEVFormer small | NuScenes |     1      |   FP32    |              -              | NDS: 0.478<br/>mAP: 0.37  |  7.4 (x1.12)  | 250 (x1.02) | 2585 (x0.55) | RTX 3090 |
| BEVFormer small | NuScenes |     1      |   FP16    |              -              | NDS: 0.479<br/>mAP: 0.37  | 15.8 (x2.40)  | 127 (x0.52) | 1729 (x0.37) | RTX 3090 |
| BEVFormer small | NuScenes |     1      | FP32/INT8 | PTQ entropy<br />per-tensor | NDS: 0.477<br/>mAP: 0.367 | 17.9 (x2.71)  | 166 (x0.68) | 1637 (x0.35) | RTX 3090 |
| BEVFormer small | NuScenes |     1      | FP16/INT8 | PTQ entropy<br />per-tensor | NDS: 0.476<br/>mAP: 0.366 | 20.4 (x3.10)  | 108 (x0.44) | 1467 (x0.31) | RTX 3090 |
| BEVFormer base  | NuScenes |     1      |   FP32    |              -              | NDS: 0.517<br/>mAP: 0.416 |  3.0 (x2.00)  | 292 (x0.17) | 5715 (x0.41) | RTX 3090 |
| BEVFormer base  | NuScenes |     1      |   FP16    |              -              | NDS: 0.517<br/>mAP: 0.416 |  4.9 (x3.27)  | 148 (x0.09) | 3417 (x0.25) | RTX 3090 |
| BEVFormer base  | NuScenes |     1      | FP32/INT8 | PTQ entropy<br />per-tensor | NDS: 0.515<br/>mAP: 0.414 |  6.9 (x4.60)  | 202 (x0.12) | 3307 (x0.24) | RTX 3090 |
| BEVFormer base  | NuScenes |     1      | FP16/INT8 | PTQ entropy<br />per-tensor | NDS: 0.514<br/>mAP: 0.413 |  8.0 (x5.33)  | 131 (x0.08) | 2429 (x0.17) | RTX 3090 |

**FP16 Plugins with nv_half2**

|      Model      |   Data   | Batch Size | Float/Int |     Quantization Method     |          NDS/mAP          |      FPS      |  Size (MB)  | Memory (MB)  |  Device  |
| :-------------: | :------: | :--------: | :-------: | :-------------------------: | :-----------------------: | :-----------: | :---------: | :----------: | :------: |
| BEVFormer tiny  | NuScenes |     1      |   FP16    |              -              | NDS: 0.355<br/>mAP: 0.251 | 84.2 (x2.22)  | 72 (x0.53)  | 1205 (x0.56) | RTX 3090 |
| BEVFormer tiny  | NuScenes |     1      | FP16/INT8 | PTQ entropy<br />per-tensor | NDS: 0.354<br/>mAP: 0.250 | 108.3 (x2.86) | 52 (x0.38)  | 1093 (x0.51) | RTX 3090 |
| BEVFormer small | NuScenes |     1      |   FP16    |              -              | NDS: 0.479<br/>mAP: 0.371 | 18.6 (x2.82)  | 124 (x0.51) | 1725 (x0.37) | RTX 3090 |
| BEVFormer small | NuScenes |     1      | FP16/INT8 | PTQ entropy<br />per-tensor | NDS: 0.477<br/>mAP: 0.368 | 22.9 (x3.47)  | 110 (x0.45) | 1487 (x0.32) | RTX 3090 |
| BEVFormer base  | NuScenes |     1      |   FP16    |              -              | NDS: 0.517<br/>mAP: 0.416 |  6.6 (x4.40)  | 146 (x0.09) | 3415 (x0.25) | RTX 3090 |
| BEVFormer base  | NuScenes |     1      | FP16/INT8 | PTQ entropy<br />per-tensor | NDS: 0.516<br/>mAP: 0.415 |  8.6 (x5.73)  | 159 (x0.09) | 2479 (x0.18) | RTX 3090 |

### BEVDet

#### BEVDet PyTorch

|      Model      |   Data   | Batch Size |         NDS/mAP          | FPS  | Size (MB) | Memory (MB) |   Device   |
| :-------------: | :------: | :--------: | :----------------------: | :--: | :-------: | :---------: | :--------: |
| BEVDet R50 CBGS | NuScenes |     1      | NDS: 0.38<br/>mAP: 0.298 | 29.0 |    170    |    1858     | RTX 2080Ti |

#### BEVDet TensorRT

**with Custom Plugin bev_pool_v2 (Support nv_half, nv_half2 and int8), modified from [Official BEVDet](https://github.com/HuangJunJie2017/BEVDet)**

|      Model      |   Data   | Batch Size | Float/Int |     Quantization Method     |          NDS/mAP          |  FPS  | Size (MB) | Memory (MB) |   Device   |
| :-------------: | :------: | :--------: | :-------: | :-------------------------: | :-----------------------: | :---: | :-------: | :---------: | :--------: |
| BEVDet R50 CBGS | NuScenes |     1      |   FP32    |              -              | NDS: 0.38<br/>mAP: 0.298  | 44.6  |    245    |    1032     | RTX 2080Ti |
| BEVDet R50 CBGS | NuScenes |     1      |   FP16    |              -              | NDS: 0.38<br/>mAP: 0.298  | 135.1 |    86     |     790     | RTX 2080Ti |
| BEVDet R50 CBGS | NuScenes |     1      | FP32/INT8 | PTQ entropy<br />per-tensor | NDS: 0.355<br/>mAP: 0.274 | 234.7 |    44     |     706     | RTX 2080Ti |
| BEVDet R50 CBGS | NuScenes |     1      | FP16/INT8 | PTQ entropy<br />per-tensor | NDS: 0.357<br/>mAP: 0.277 | 236.4 |    44     |     706     | RTX 2080Ti |

### 2D Detection Models

This project also supports common 2D object detection models in MMDetection with little modification. The following are deployment examples of YOLOx and CenterNet.

#### YOLOx

|                            Model                             | Data | Framework | Batch Size | Float/Int |     Quantization Method     |    mAP     |       FPS       |  Size (MB)  | Memory (MB)  |  Device  |
| :----------------------------------------------------------: | :--: | :-------: | :--------: | :-------: | :-------------------------: | :--------: | :-------------: | :---------: | :----------: | :------: |
| YOLOx<br />[download](https://drive.google.com/file/d/10_mRoiLfK1JEIVq2uqtkBLHES9QgoFgm/view?usp=share_link) | COCO |  PyTorch  |     32     |   FP32    |              -              | mAP: 0.506 |      63.1       |     379     |     7617     | RTX 3090 |
|                            YOLOx                             | COCO | TensorRT  |     32     |   FP32    |              -              | mAP: 0.506 |    71.3 (x1)    |  546 (x1)   |  9943 (x1)   | RTX 3090 |
|                            YOLOx                             | COCO | TensorRT  |     32     |   FP16    |              -              | mAP: 0.506 | 296.8   (x4.16) | 192 (x0.35) | 4567 (x0.46) | RTX 3090 |
|                            YOLOx                             | COCO | TensorRT  |     32     | FP32/INT8 | PTQ entropy<br />per-tensor | mAP: 0.488 |  556.4 (x7.80)  | 99 (x0.18)  | 5225 (x0.53) | RTX 3090 |
|                            YOLOx                             | COCO | TensorRT  |     32     | FP16/INT8 | PTQ entropy<br />per-tensor | mAP: 0.479 |  550.6 (x7.72)  | 99 (x0.18)  | 5119 (x0.51) | RTX 3090 |

#### CenterNet

|                            Model                             | Data | Framework | Batch Size | Float/Int |     Quantization Method     |    mAP     |      FPS       | Size (MB)  | Memory (MB)  |  Device  |
| :----------------------------------------------------------: | :--: | :-------: | :--------: | :-------: | :-------------------------: | :--------: | :------------: | :--------: | :----------: | :------: |
| CenterNet<br />[download](https://drive.google.com/file/d/1uZVIpDQEWIgY5-1IivQte-9Kue6k71A9/view?usp=share_link) | COCO |  PyTorch  |     32     |   FP32    |              -              | mAP: 0.299 |     337.4      |     56     |     5171     | RTX 3090 |
|                          CenterNet                           | COCO | TensorRT  |     32     |   FP32    |              -              | mAP: 0.299 |   475.6 (x1)   |  58 (x1)   |  8241 (x1)   | RTX 3090 |
|                          CenterNet                           | COCO | TensorRT  |     32     |   FP16    |              -              | mAP: 0.297 | 1247.1 (x2.62) | 29 (x0.50) | 5183 (x0.63) | RTX 3090 |
|                          CenterNet                           | COCO | TensorRT  |     32     | FP32/INT8 | PTQ entropy<br />per-tensor | mAP: 0.27  | 1534.0 (x3.22) | 20 (x0.34) | 6549 (x0.79) | RTX 3090 |
|                          CenterNet                           | COCO | TensorRT  |     32     | FP16/INT8 | PTQ entropy<br />per-tensor | mAP: 0.285 | 1889.0 (x3.97) | 17 (x0.29) | 6453 (x0.78) | RTX 3090 |

## Clone

```shell
git clone git@github.com:DerryHub/BEVFormer_tensorrt.git
cd BEVFormer_tensorrt
PROJECT_DIR=$(pwd)
```

## Data Preparation

### MS COCO (For 2D Detection)

Download the [COCO 2017](https://cocodataset.org/#download) datasets to `/path/to/coco` and unzip them.

```shell
cd ${PROJECT_DIR}/data
ln -s /path/to/coco coco
```

### NuScenes and CAN bus (For BEVFormer)

Download nuScenes V1.0 full dataset data and CAN bus expansion data [HERE](https://www.nuscenes.org/download) as `/path/to/nuscenes` and `/path/to/can_bus`.

Prepare nuscenes data like [BEVFormer](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/prepare_dataset.md).

```shell
cd ${PROJECT_DIR}/data
ln -s /path/to/nuscenes nuscenes
ln -s /path/to/can_bus can_bus

cd ${PROJECT_DIR}
sh samples/bevformer/create_data.sh
```

### Tree

```shell
${PROJECT_DIR}/data/.
├── can_bus
│   ├── scene-0001_meta.json
│   ├── scene-0001_ms_imu.json
│   ├── scene-0001_pose.json
│   └── ...
├── coco
│   ├── annotations
│   ├── test2017
│   ├── train2017
│   └── val2017
└── nuscenes
    ├── maps
    ├── samples
    ├── sweeps
    └── v1.0-trainval
```

## Install

### With Docker

```shell
cd ${PROJECT_DIR}
docker build -t trt85 -f docker/Dockerfile .
docker run -it --gpus all -v ${PROJECT_DIR}:/workspace/BEVFormer_tensorrt/ \
-v /path/to/can_bus:/workspace/BEVFormer_tensorrt/data/can_bus \
-v /path/to/coco:/workspace/BEVFormer_tensorrt/data/coco \
-v /path/to/nuscenes:/workspace/BEVFormer_tensorrt/data/nuscenes \
--shm-size 8G trt85 /bin/bash

# in container
cd /workspace/BEVFormer_tensorrt/TensorRT/build
cmake .. -DCMAKE_TENSORRT_PATH=/usr
make -j$(nproc)
make install
cd /workspace/BEVFormer_tensorrt/third_party/bev_mmdet3d
python setup.py build develop --user
```

**NOTE:** You can download the **Docker Image** [HERE](https://pan.baidu.com/s/1dPR6kvgpUoKow51870KNug?pwd=6xkq).

### From Source

#### CUDA/cuDNN/TensorRT

Download and install the `CUDA-11.6/cuDNN-8.6.0/TensorRT-8.5.1.7` following [NVIDIA](https://www.nvidia.com/en-us/).

#### PyTorch

Install PyTorch and TorchVision following the [official instructions](https://pytorch.org/get-started/locally/).

```shell
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

#### MMCV-full

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.5.0
pip install -r requirements/optional.txt
MMCV_WITH_OPS=1 pip install -e .
```

#### MMDetection

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.25.1
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

#### MMDeploy

```shell
git clone git@github.com:open-mmlab/mmdeploy.git
cd mmdeploy
git checkout v0.10.0

git clone git@github.com:NVIDIA/cub.git third_party/cub
cd third_party/cub
git checkout c3cceac115

# go back to third_party directory and git clone pybind11
cd ..
git clone git@github.com:pybind/pybind11.git pybind11
cd pybind11
git checkout 70a58c5
```

##### Build TensorRT Plugins of MMDeploy

**Make sure cmake version >= 3.14.0 and gcc version >= 7.**

```shell
export MMDEPLOY_DIR=/the/root/path/of/MMDeploy
export TENSORRT_DIR=/the/path/of/tensorrt
export CUDNN_DIR=/the/path/of/cuda

export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH

cd ${MMDEPLOY_DIR}
mkdir -p build
cd build
cmake -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_TARGET_BACKENDS=trt -DTENSORRT_DIR=${TENSORRT_DIR} -DCUDNN_DIR=${CUDNN_DIR} ..
make -j$(nproc) 
make install
```

##### Install MMDeploy

```shell
cd ${MMDEPLOY_DIR}
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

#### Install this Project

```shell
cd ${PROJECT_DIR}
pip install -r requirements.txt
```

##### Build and Install Custom TensorRT Plugins

**NOTE: CUDA>=11.4, SM version>=7.5**

```shell
cd ${PROJECT_DIR}/TensorRT/build
cmake .. -DCMAKE_TENSORRT_PATH=/path/to/TensorRT
make -j$(nproc)
make install
```

**Run Unit Test of  Custom TensorRT Plugins**

```shell
cd ${PROJECT_DIR}
sh samples/test_trt_ops.sh
```

##### Build and Install Part of Ops in MMDetection3D

```shell
cd ${PROJECT_DIR}/third_party/bev_mmdet3d
python setup.py build develop
```

#### Prepare the Checkpoints

Download above PyTorch checkpoints to `${PROJECT_DIR}/checkpoints/pytorch/`. The ONNX files and TensorRT engines will be saved in `${PROJECT_DIR}/checkpoints/onnx/` and `${PROJECT_DIR}/checkpoints/tensorrt/`.

## Custom TensorRT Plugins

Support Common TensorRT Ops in BEVFormer:

* Grid Sampler
* Multi-scale Deformable Attention
* Modulated Deformable Conv2d
* Rotate
* Inverse
* BEV Pool V2
* Flash Multi-Head Attention

Each operation is implemented as 2 versions: **FP32/FP16 (nv_half)/INT8** and **FP32/FP16 (nv_half2)/INT8**.

For specific speed comparison, see [**Custom TensorRT Plugins**](./TensorRT/).

## Run

The following tutorial uses `BEVFormer base` as an example.

* Evaluate with PyTorch

```shell
cd ${PROJECT_DIR}
# defult gpu_id is 0
sh samples/bevformer/base/pth_evaluate.sh -d ${gpu_id}
```

* Evaluate with TensorRT and MMDeploy Plugins

```shell
# convert .pth to .onnx
sh samples/bevformer/base/pth2onnx.sh -d ${gpu_id}
# convert .onnx to TensorRT engine (FP32)
sh samples/bevformer/base/onnx2trt.sh -d ${gpu_id}
# convert .onnx to TensorRT engine (FP16)
sh samples/bevformer/base/onnx2trt_fp16.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP32)
sh samples/bevformer/base/trt_evaluate.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP16)
sh samples/bevformer/base/trt_evaluate_fp16.sh -d ${gpu_id}

# Quantization
# calibration and convert .onnx to TensorRT engine (FP32/INT8)
sh samples/bevformer/base/onnx2trt_int8.sh -d ${gpu_id}
# calibration and convert .onnx to TensorRT engine (FP16/INT8)
sh samples/bevformer/base/onnx2trt_int8_fp16.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP32/INT8)
sh samples/bevformer/base/trt_evaluate_int8.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP16/INT8)
sh samples/bevformer/base/trt_evaluate_int8_fp16.sh -d ${gpu_id}

# quantization aware train
# defult gpu_ids is 0,1,2,3,4,5,6,7
sh samples/bevformer/base/quant_aware_train.sh -d ${gpu_ids}
# then following the post training quantization process
```

* Evaluate with TensorRT and Custom Plugins

```shell
# nv_half
# convert .pth to .onnx
sh samples/bevformer/plugin/base/pth2onnx.sh -d ${gpu_id}
# convert .onnx to TensorRT engine (FP32)
sh samples/bevformer/plugin/base/onnx2trt.sh -d ${gpu_id}
# convert .onnx to TensorRT engine (FP16-nv_half)
sh samples/bevformer/plugin/base/onnx2trt_fp16.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP32)
sh samples/bevformer/plugin/base/trt_evaluate.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP16-nv_half)
sh samples/bevformer/plugin/base/trt_evaluate_fp16.sh -d ${gpu_id}

# nv_half2
# convert .pth to .onnx
sh samples/bevformer/plugin/base/pth2onnx_2.sh -d ${gpu_id}
# convert .onnx to TensorRT engine (FP16-nv_half2)
sh samples/bevformer/plugin/base/onnx2trt_fp16_2.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP16-nv_half2)
sh samples/bevformer/plugin/base/trt_evaluate_fp16_2.sh -d ${gpu_id}

# Quantization
# nv_half
# calibration and convert .onnx to TensorRT engine (FP32/INT8)
sh samples/bevformer/plugin/base/onnx2trt_int8.sh -d ${gpu_id}
# calibration and convert .onnx to TensorRT engine (FP16-nv_half/INT8)
sh samples/bevformer/plugin/base/onnx2trt_int8_fp16.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP32/INT8)
sh samples/bevformer/plugin/base/trt_evaluate_int8.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP16-nv_half/INT8)
sh samples/bevformer/plugin/base/trt_evaluate_int8_fp16.sh -d ${gpu_id}

# nv_half2
# calibration and convert .onnx to TensorRT engine (FP16-nv_half2/INT8)
sh samples/bevformer/plugin/base/onnx2trt_int8_fp16_2.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP16-nv_half2/INT8)
sh samples/bevformer/plugin/base/trt_evaluate_int8_fp16_2.sh -d ${gpu_id}
```

## Acknowledgement

This project is mainly based on these excellent open source projects:

* [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
* [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
* [PyTorch](https://github.com/pytorch/pytorch)
* [MMCV](https://github.com/open-mmlab/mmcv)
* [MMDetection](https://github.com/open-mmlab/mmdetection)
* [MMDeploy](https://github.com/open-mmlab/mmdeploy)

