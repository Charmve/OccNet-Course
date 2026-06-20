#!/bin/bash

gpu_id=0
while getopts "d:" opt
do
  case $opt in
    d)
      gpu_id=$OPTARG
      ;;
    ?)
      echo "There is unrecognized parameter."
      exit 1
      ;;
  esac
done

echo "Running on the GPU: $gpu_id"

CUDA_VISIBLE_DEVICES=$gpu_id python tools/pth2onnx.py \
configs/centernet/centernet_resnet18_dcnv2_140e_coco_trt.py \
checkpoints/pytorch/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth \
--opset_version 13 \
--cuda
