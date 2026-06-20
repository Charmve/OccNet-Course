# Installation instructions
**a. Create a conda virtual environment and activate it.**
```shell
conda create -n occ python=3.7 -y
conda activate occ
```

**1. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

Pytorch and cuda with higher version is also supported.
```shell
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
```


**2. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.1
```

**3. Install mmdet and mmseg.**
```shell
pip install mmdet==2.19.0
pip install mmsegmentation==0.20.0
```

**4. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.18.1 # Other versions may not be compatible.
python setup.py install
```

**5. Install timm.**
```shell
pip install timm
```


**6. Clone OccNet.**
```
git clone https://github.com/OpenDriveLab/OccNet.git
```

**7. Prepare pretrained model**
```shell
cd OccNet
mkdir ckpts
cd ckpts 
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

**8. Install [InternImage](https://github.com/OpenGVLab/InternImage) (Optional).**

OccNet support InternImage backbone with much better performance than ImageNet backbone.
- Install dcnv3
```
cd OccNet
cd projects/mmdet3d_plugin/bevformer/backbones/ops_dcnv3
sh make.sh
python test.py
```
- Prepare pretrained model
```
cd OccNet/ckpts
wget https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_s_fpn_3x_coco.pth
```