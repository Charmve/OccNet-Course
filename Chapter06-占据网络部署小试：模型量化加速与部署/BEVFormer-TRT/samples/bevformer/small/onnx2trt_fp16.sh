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

CUDA_VISIBLE_DEVICES=$gpu_id python tools/bevformer/onnx2trt.py \
configs/bevformer/bevformer_small_trt.py \
checkpoints/onnx/bevformer_small_epoch_24.onnx \
--fp16
