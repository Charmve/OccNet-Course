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

CUDA_VISIBLE_DEVICES=$gpu_id python tools/bevformer/evaluate_trt.py \
configs/bevformer/plugin/bevformer_base_trt_p2.py \
checkpoints/tensorrt/bevformer_r101_dcn_24ep_cp2_entropy_int8_fp16.trt
