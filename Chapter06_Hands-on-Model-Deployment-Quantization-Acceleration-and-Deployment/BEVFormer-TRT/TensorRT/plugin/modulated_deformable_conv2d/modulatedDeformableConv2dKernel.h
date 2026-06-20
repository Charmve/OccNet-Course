//
// Created by Derry Lin on 2022/11/10.
//

#ifndef TENSORRT_OPS_MODULATEDDEFORMABLECONV2DKERNEL_H
#define TENSORRT_OPS_MODULATEDDEFORMABLECONV2DKERNEL_H

#include "cuda_int8.h"
#include <cublas_v2.h>

template <typename T>
void ModulatedDeformConvForwardCUDAKernel(
    const T *input, const T *weight, const T *bias, const T *offset,
    const T *mask, T *output, void *workspace, int batch, int channels,
    int height, int width, int channels_out, int kernel_w, int kernel_h,
    int stride_w, int stride_h, int pad_w, int pad_h, int dilation_w,
    int dilation_h, int group, int deformable_group, int im2col_step,
    cublasHandle_t cublas_handle, cudaStream_t stream);

template <typename T>
void ModulatedDeformConvForwardCUDAKernel_int8(
    const int8_4 *input, const float &scale_i, const int8_4 *weight,
    const float &scale_w, const T *bias, const int8_t *offset,
    const float &scale_off, const int8_t *mask, const float &scale_mask,
    int8_t *output, const float &scale_o, void *workspace, int batch,
    int channels, int height, int width, int channels_out, int kernel_w,
    int kernel_h, int stride_w, int stride_h, int pad_w, int pad_h,
    int dilation_w, int dilation_h, int group, int deformable_group,
    int im2col_step, cublasHandle_t cublas_handle, cudaStream_t stream);

#endif // TENSORRT_OPS_MODULATEDDEFORMABLECONV2DKERNEL_H
