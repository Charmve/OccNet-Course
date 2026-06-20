//
// Created by Derry Lin on 2023/6/21.
//

#ifndef TENSORRT_OPS_MULTIHEADFLASHATTNKERNEL_H
#define TENSORRT_OPS_MULTIHEADFLASHATTNKERNEL_H

#include "cuda_int8.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <typename T>
int qkv_flash(const T *query, const T *key, const T *value, T *output,
              const int &batch, const int &q_len, const int &kv_len,
              const int &embed_dim, cudaStream_t stream);
int qkv_flash_int8(const int8_4 *query, const float &scale_q, const int8_4 *key,
                   const float &scale_k, const int8_4 *value,
                   const float &scale_v, int8_4 *output, const float &scale_o,
                   const int &batch, const int &q_len, const int &kv_len,
                   const int &embed_dim, cudaStream_t stream);

#endif // TENSORRT_OPS_MULTIHEADFLASHATTNKERNEL_H
