//
// Created by Derry Lin on 2023/5/11.
//

#ifndef TENSORRT_OPS_BEVPOOLKERNEL_H
#define TENSORRT_OPS_BEVPOOLKERNEL_H

#include "cuda_int8.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// CUDA function declarations
template <typename T>
void bev_pool_v2(int c, int n_intervals, int num_points, const T *depth,
                 const T *feat, const int *ranks_depth, const int *ranks_feat,
                 const int *ranks_bev, const int *interval_starts,
                 const int *interval_lengths, T *out, cudaStream_t stream);

void bev_pool_v2_h2(int c, int n_intervals, int num_points, const __half *depth,
                    const __half2 *feat, const int *ranks_depth,
                    const int *ranks_feat, const int *ranks_bev,
                    const int *interval_starts, const int *interval_lengths,
                    __half2 *out, cudaStream_t stream);

void bev_pool_v2_int8(int c, int n_intervals, int num_points,
                      const int8_t *depth, const float &scale_d,
                      const int8_4 *feat, const float &scale_f,
                      const int *ranks_depth, const int *ranks_feat,
                      const int *ranks_bev, const int *interval_starts,
                      const int *interval_lengths, int8_4 *out,
                      const float &scale_o, cudaStream_t stream);

#endif // TENSORRT_OPS_BEVPOOLKERNEL_H
