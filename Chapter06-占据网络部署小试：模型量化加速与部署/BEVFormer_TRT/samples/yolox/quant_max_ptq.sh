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

CUDA_VISIBLE_DEVICES=$gpu_id python tools/2d/post_training_quant.py \
configs/yolox/yolox_x_8x8_300e_coco_trt_q.py \
checkpoints/pytorch/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth \
--calibrator max
