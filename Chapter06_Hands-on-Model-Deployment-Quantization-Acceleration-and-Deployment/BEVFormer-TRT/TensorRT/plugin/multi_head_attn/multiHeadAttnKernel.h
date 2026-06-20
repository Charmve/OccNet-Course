//
// Created by Derry Lin on 2023/6/16.
//

#ifndef TENSORRT_OPS_MULTIHEADATTNKERNEL_H
#define TENSORRT_OPS_MULTIHEADATTNKERNEL_H

#include "cuda_int8.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <typename T>
int qkv(T *workspace, const T *query, const T *key, const T *value, T *output,
        const int &batch, const int &q_len, const int &kv_len,
        const int &embed_dim, cublasHandle_t cublasHandle, cudaStream_t stream,
        cublasGemmAlgo_t algo_1 = CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        cublasGemmAlgo_t algo_2 = CUBLAS_GEMM_DEFAULT_TENSOR_OP);
int qkv_int8(int8_4 *workspace, const int8_4 *query, const float &scale_q,
             const int8_4 *key, const float &scale_k, const int8_4 *value,
             const float &scale_v, int8_4 *output, const float &scale_o,
             const int &batch, const int &q_len, const int &kv_len,
             const int &embed_dim, cublasHandle_t cublasHandle,
             cudaStream_t stream,
             cublasGemmAlgo_t algo_1 = CUBLAS_GEMM_DEFAULT_TENSOR_OP,
             cublasGemmAlgo_t algo_2 = CUBLAS_GEMM_DEFAULT_TENSOR_OP);

#endif // TENSORRT_OPS_MULTIHEADATTNKERNEL_H
