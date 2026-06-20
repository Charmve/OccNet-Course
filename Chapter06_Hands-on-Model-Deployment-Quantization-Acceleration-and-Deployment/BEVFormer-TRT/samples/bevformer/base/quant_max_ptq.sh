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

CUDA_VISIBLE_DEVICES=$gpu_id python tools/bevformer/post_training_quant.py \
configs/bevformer/bevformer_base_trt_q.py \
checkpoints/pytorch/bevformer_r101_dcn_24ep.pth \
--calibrator max
