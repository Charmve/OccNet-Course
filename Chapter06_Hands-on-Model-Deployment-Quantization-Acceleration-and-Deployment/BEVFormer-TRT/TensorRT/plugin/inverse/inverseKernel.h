//
// Created by Derry Lin on 2023/5/11.
//

#ifndef TENSORRT_OPS_INVERSEKERNEL_H
#define TENSORRT_OPS_INVERSEKERNEL_H

#include <cublas_v2.h>
#include <cuda_runtime.h>

template <typename T>
void inverse(T *output, T *input, void *workspace, T **inpPtrArr, T **outPtrArr,
             const int &matrix_n, const int &matrix_dim,
             cublasHandle_t cublas_handle, cudaStream_t stream);

#endif // TENSORRT_OPS_INVERSEKERNEL_H
