#include "bevPoolKernel.h"
#include "cuda_helper.h"
#include "cuda_int8.h"
/*
  Function: pillar pooling
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    depth            : input depth, FloatTensor[b,n,d,h,w]
    feat             : input feat, FloatTensor[b,n,h,w,c]
    ranks_depth      : input index of depth, IntTensor[n_points]
    ranks_feat       : input index of feat, IntTensor[n_points]
    ranks_bev        : output index, IntTensor[n_points]
    interval_lengths : starting position for pooled point,
  IntTensor[n_intervals] interval_starts  : how many points in each pooled
  point, IntTensor[n_intervals] out              : output features,
  FloatTensor[b, z, h, w, c]
*/
template <typename T>
__global__ void bev_pool_v2_kernel(
    int c, int n_intervals, const T *__restrict__ depth,
    const T *__restrict__ feat, const int *__restrict__ ranks_depth,
    const int *__restrict__ ranks_feat, const int *__restrict__ ranks_bev,
    const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths, T *__restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals)
    return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  T psum = 0;
  const T *cur_depth;
  const T *cur_feat;
  for (int i = 0; i < interval_length; i++) {
    cur_depth = depth + ranks_depth[interval_start + i];
    cur_feat = feat + ranks_feat[interval_start + i] * c + cur_c;
    psum += *cur_feat * *cur_depth;
  }

  const int *cur_rank = ranks_bev + interval_start;
  T *cur_out = out + *cur_rank * c + cur_c;
  *cur_out = psum;
}

template <>
__global__ void bev_pool_v2_kernel(
    int c, int n_intervals, const __half *__restrict__ depth,
    const __half *__restrict__ feat, const int *__restrict__ ranks_depth,
    const int *__restrict__ ranks_feat, const int *__restrict__ ranks_bev,
    const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths, __half *__restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals)
    return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  __half psum = 0;
  const __half *cur_depth;
  const __half *cur_feat;
  for (int i = 0; i < interval_length; i++) {
    cur_depth = depth + ranks_depth[interval_start + i];
    cur_feat = feat + ranks_feat[interval_start + i] * c + cur_c;
    psum = __hfma(*cur_feat, *cur_depth, psum);
  }

  const int *cur_rank = ranks_bev + interval_start;
  __half *cur_out = out + *cur_rank * c + cur_c;
  *cur_out = psum;
}

__global__ void bev_pool_v2_kernel_h2(
    int c, int n_intervals, const __half *__restrict__ depth,
    const __half2 *__restrict__ feat, const int *__restrict__ ranks_depth,
    const int *__restrict__ ranks_feat, const int *__restrict__ ranks_bev,
    const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths, __half2 *__restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals)
    return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  __half2 psum = __float2half2_rn(0);
  const __half *cur_depth;
  const __half2 *cur_feat;
  for (int i = 0; i < interval_length; i++) {
    cur_depth = depth + ranks_depth[interval_start + i];
    cur_feat = feat + ranks_feat[interval_start + i] * c + cur_c;
    psum = __hfma2(*cur_feat, __half2half2(*cur_depth), psum);
  }

  const int *cur_rank = ranks_bev + interval_start;
  __half2 *cur_out = out + *cur_rank * c + cur_c;
  *cur_out = psum;
}

template <typename T> __forceinline__ __device__ T sign_05(T x) {
  if (x > 0) {
    return 0.5f;
  }
  return -0.5f;
}

template <typename T> __forceinline__ __device__ int8_t T2int8(T a) {
  a = a > 127 ? 127 : a;
  a = a < -128 ? -128 : a;
  return int8_t(a + sign_05<T>(a));
}

__global__ void bev_pool_v2_kernel_int8(
    int c, int n_intervals, const int8_t *__restrict__ depth,
    const int8_4 *__restrict__ feat, const int *__restrict__ ranks_depth,
    const int *__restrict__ ranks_feat, const int *__restrict__ ranks_bev,
    const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths, int8_4 *__restrict__ out,
    const float scale_io) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals)
    return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  int32_4 psum = 0;
  const int8_t *cur_depth;
  const int8_4 *cur_feat;
  for (int i = 0; i < interval_length; i++) {
    cur_depth = depth + ranks_depth[interval_start + i];
    cur_feat = feat + ranks_feat[interval_start + i] * c + cur_c;
    psum.x += (cur_feat->x) * (*cur_depth);
    psum.y += (cur_feat->y) * (*cur_depth);
    psum.z += (cur_feat->z) * (*cur_depth);
    psum.w += (cur_feat->w) * (*cur_depth);
  }
  int8_4 output;
  output.x = T2int8<float>(psum.x * scale_io);
  output.y = T2int8<float>(psum.y * scale_io);
  output.z = T2int8<float>(psum.z * scale_io);
  output.w = T2int8<float>(psum.w * scale_io);

  const int *cur_rank = ranks_bev + interval_start;
  int8_4 *cur_out = out + *cur_rank * c + cur_c;
  *cur_out = output;
}

template <typename T>
void bev_pool_v2(int c, int n_intervals, int num_points, const T *depth,
                 const T *feat, const int *ranks_depth, const int *ranks_feat,
                 const int *ranks_bev, const int *interval_starts,
                 const int *interval_lengths, T *out, cudaStream_t stream) {
  cudaMemset((T *)out, 0, num_points * sizeof(T));
  bev_pool_v2_kernel<<<GET_BLOCKS(n_intervals * c), THREADS_PER_BLOCK, 0,
                       stream>>>(c, n_intervals, depth, feat, ranks_depth,
                                 ranks_feat, ranks_bev, interval_starts,
                                 interval_lengths, out);
  cudaCheckError();
}

void bev_pool_v2_h2(int c, int n_intervals, int num_points, const __half *depth,
                    const __half2 *feat, const int *ranks_depth,
                    const int *ranks_feat, const int *ranks_bev,
                    const int *interval_starts, const int *interval_lengths,
                    __half2 *out, cudaStream_t stream) {
  cudaMemset((__half *)out, 0, num_points * sizeof(__half));
  bev_pool_v2_kernel_h2<<<GET_BLOCKS(n_intervals * c / 2), THREADS_PER_BLOCK, 0,
                          stream>>>(c / 2, n_intervals, depth, feat,
                                    ranks_depth, ranks_feat, ranks_bev,
                                    interval_starts, interval_lengths, out);
  cudaCheckError();
}

void bev_pool_v2_int8(int c, int n_intervals, int num_points,
                      const int8_t *depth, const float &scale_d,
                      const int8_4 *feat, const float &scale_f,
                      const int *ranks_depth, const int *ranks_feat,
                      const int *ranks_bev, const int *interval_starts,
                      const int *interval_lengths, int8_4 *out,
                      const float &scale_o, cudaStream_t stream) {
  cudaMemset((int8_t *)out, 0, num_points * sizeof(int8_t));
  bev_pool_v2_kernel_int8<<<GET_BLOCKS(n_intervals * c / 4), THREADS_PER_BLOCK,
                            0, stream>>>(
      c / 4, n_intervals, depth, feat, ranks_depth, ranks_feat, ranks_bev,
      interval_starts, interval_lengths, out, scale_d * scale_f / scale_o);
  cudaCheckError();
}

template void bev_pool_v2(int c, int n_intervals, int num_points,
                          const float *depth, const float *feat,
                          const int *ranks_depth, const int *ranks_feat,
                          const int *ranks_bev, const int *interval_starts,
                          const int *interval_lengths, float *out,
                          cudaStream_t stream);

template void bev_pool_v2(int c, int n_intervals, int num_points,
                          const __half *depth, const __half *feat,
                          const int *ranks_depth, const int *ranks_feat,
                          const int *ranks_bev, const int *interval_starts,
                          const int *interval_lengths, __half *out,
                          cudaStream_t stream);
