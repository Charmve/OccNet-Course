#!/bin/bash

gpu_id=0,1,2,3,4,5,6,7
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

array=(${gpu_id//,/ })
count=${#array[@]}

echo "Running on $count GPU: $gpu_id"

CUDA_VISIBLE_DEVICES=$gpu_id \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m torch.distributed.run --nproc_per_node=$count \
tools/bevformer/train.py configs/bevformer/bevformer_tiny_trt_q.py \
--launcher pytorch
