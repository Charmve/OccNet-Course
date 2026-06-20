#!/bin/bash

gpu_id=0
config=None
while getopts "d:c:" opt
do
  case $opt in
    d)
      gpu_id=$OPTARG
      ;;
    c)
      config=$OPTARG
      ;;
    ?)
      echo "There is unrecognized parameter."
      exit 1
      ;;
  esac
done

echo "Running on the GPU: $gpu_id"

CUDA_VISIBLE_DEVICES=$gpu_id python tools/flops_params.py $config --cuda
