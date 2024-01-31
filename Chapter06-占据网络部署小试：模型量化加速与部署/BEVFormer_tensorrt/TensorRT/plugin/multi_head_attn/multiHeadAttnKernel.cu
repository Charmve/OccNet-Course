//
// Created by Derry Lin on 2023/6/16.
//

#include "cuda_helper.h"
#include "multiHeadAttnKernel.h"
#include <cmath>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda_fp16.h>

constexpr int kWarpSize = 32;

template <typename T> __forceinline__ __device__ T exp(const T &val) {
  return cuda::std::exp(val);
}

template <> __forceinline__ __device__ __half exp(const __half &val) {
  return hexp(val);
}

template <> __forceinline__ __device__ __half2 exp(const __half2 &val) {
  return h2exp(val);
}

template <typename T> __forceinline__ __device__ T max(const T &a, const T &b) {
  return max(a, b);
}

#if __CUDA_ARCH__ >= 800
template <>
__forceinline__ __device__ __half max(const __half &a, const __half &b) {
  return __hmax(a, b);
}
template <>
__forceinline__ __device__ __half2 max(const __half2 &a, const __half2 &b) {
  return __hmax2(a, b);
}
#else
template <>
__forceinline__ __device__ __half max(const __half &a, const __half &b) {
  return __hgt(a, b) ? a : b;
}
template <>
__forceinline__ __device__ __half2 max(const __half2 &a, const __half2 &b) {
  return __hfma2(__hgt2(a, b), a, __hmul2(__hle2(a, b), b));
}
#endif

template <typename T, int num_batch_per_thr, int num_elements_per_thr>
__global__ void softmaxWarpKernel(T *__restrict__ inp_out, const int batch,
                                  const int num_elements) {
  const int batch_i_ =
      (blockDim.y * blockIdx.x + threadIdx.y) * num_batch_per_thr;
  const int col_i = threadIdx.x;
  const int stride = gridDim.x * blockDim.y;
  T buf[num_batch_per_thr][num_elements_per_thr];
  T thr_max[num_batch_per_thr];
  T sum[num_batch_per_thr];

  for (int batch_i = batch_i_; batch_i < batch; batch_i += stride) {
#pragma unroll
    for (int b_i = 0; b_i < num_batch_per_thr; b_i++) {
      thr_max[b_i] = -cuda::std::numeric_limits<T>::infinity();
#pragma unroll
      for (int c_i = 0; c_i < num_elements_per_thr; c_i++) {
        if ((col_i + c_i * kWarpSize < num_elements) &&
            (batch_i + b_i < batch)) {
          buf[b_i][c_i] =
              inp_out[num_elements * (batch_i + b_i) + col_i + c_i * kWarpSize];
          thr_max[b_i] = max(thr_max[b_i], buf[b_i][c_i]);
        } else {
          buf[b_i][c_i] = -cuda::std::numeric_limits<T>::infinity();
        }
      }
    }

#pragma unroll
    for (int b_i = 0; b_i < num_batch_per_thr; b_i++) {
#pragma unroll
      for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
        thr_max[b_i] =
            max(thr_max[b_i], __shfl_xor_sync(0xffffffff, thr_max[b_i], mask));
      }

      sum[b_i] = 0;
#pragma unroll
      for (int c_i = 0; c_i < num_elements_per_thr; c_i++) {
        buf[b_i][c_i] = exp(buf[b_i][c_i] - thr_max[b_i]);
        sum[b_i] += buf[b_i][c_i];
      }

#pragma unroll
      for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
        sum[b_i] += __shfl_xor_sync(0xffffffff, sum[b_i], mask);
      }
    }

#pragma unroll
    for (int b_i = 0; b_i < num_batch_per_thr; b_i++) {
      int index = num_elements * (batch_i + b_i);
#pragma unroll
      for (int c_i = 0; c_i < num_elements_per_thr; c_i++) {
        if ((col_i + c_i * kWarpSize < num_elements) &&
            (batch_i + b_i < batch)) {
          inp_out[index + col_i + c_i * kWarpSize] = buf[b_i][c_i] / sum[b_i];
        }
      }
    }
  }
}

template <int num_batch_per_thr, int num_elements_per_thr>
__global__ void softmaxWarpKernel(__half2 *__restrict__ inp_out,
                                  const int batch, const int num_elements) {
  const int batch_i_ =
      (blockDim.y * blockIdx.x + threadIdx.y) * num_batch_per_thr;
  const int col_i = threadIdx.x;
  const int stride = gridDim.x * blockDim.y;
  __half2 buf[num_batch_per_thr][num_elements_per_thr];
  __half2 thr_max[num_batch_per_thr];
  __half2 sum[num_batch_per_thr];
  const __half2 neg_min =
      __half2half2(-cuda::std::numeric_limits<__half>::infinity());

  for (int batch_i = batch_i_; batch_i < batch; batch_i += stride) {
#pragma unroll
    for (int b_i = 0; b_i < num_batch_per_thr; b_i++) {
      thr_max[b_i] = neg_min;
#pragma unroll
      for (int c_i = 0; c_i < num_elements_per_thr; c_i++) {
        if ((col_i + c_i * kWarpSize < num_elements) &&
            (batch_i + b_i < batch)) {
          buf[b_i][c_i] =
              inp_out[num_elements * (batch_i + b_i) + col_i + c_i * kWarpSize];
          thr_max[b_i] = max(thr_max[b_i], buf[b_i][c_i]);
        } else {
          buf[b_i][c_i] = neg_min;
        }
      }
    }

#pragma unroll
    for (int b_i = 0; b_i < num_batch_per_thr; b_i++) {
#pragma unroll
      for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
        thr_max[b_i] =
            max(thr_max[b_i], __shfl_xor_sync(0xffffffff, thr_max[b_i], mask));
      }
      thr_max[b_i] = max(thr_max[b_i], __lowhigh2highlow(thr_max[b_i]));

      sum[b_i] = __float2half2_rn(0.f);
#pragma unroll
      for (int c_i = 0; c_i < num_elements_per_thr; c_i++) {
        buf[b_i][c_i] = exp(__hsub2(buf[b_i][c_i], thr_max[b_i]));
        sum[b_i] = __hadd2(sum[b_i], buf[b_i][c_i]);
      }

#pragma unroll
      for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
        sum[b_i] =
            __hadd2(sum[b_i], __shfl_xor_sync(0xffffffff, sum[b_i], mask));
      }
      sum[b_i] = __hadd2(sum[b_i], __lowhigh2highlow(sum[b_i]));
    }

#pragma unroll
    for (int b_i = 0; b_i < num_batch_per_thr; b_i++) {
      int index = num_elements * (batch_i + b_i);
#pragma unroll
      for (int c_i = 0; c_i < num_elements_per_thr; c_i++) {
        if ((col_i + c_i * kWarpSize < num_elements) &&
            (batch_i + b_i < batch)) {
          inp_out[index + col_i + c_i * kWarpSize] =
              __h2div(buf[b_i][c_i], sum[b_i]);
        }
      }
    }
  }
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

template <> __forceinline__ __device__ int8_t T2int8(__half a) {
  short temp = __half2short_rn(a);
  temp = temp > static_cast<short>(127) ? static_cast<short>(127) : temp;
  temp = temp < static_cast<short>(-128) ? static_cast<short>(-128) : temp;
  return static_cast<int8_t>(temp);
}

template <int num_batch_per_thr, int num_elements_per_thr>
__global__ void softmaxWarpInt8Kernel(int32_4 *__restrict__ input,
                                      const float scale_i,
                                      int8_4 *__restrict__ output,
                                      const int batch, const int num_elements,
                                      float *__restrict__ sum_each_line) {
  const int batch_i_ =
      (blockDim.y * blockIdx.x + threadIdx.y) * num_batch_per_thr;
  const int col_i = threadIdx.x;
  const int stride = gridDim.x * blockDim.y;
  __half2 buf[num_batch_per_thr][num_elements_per_thr * 2];
  __half2 thr_max[num_batch_per_thr];
  __half2 sum[num_batch_per_thr];
  const __half2 c127 = __float2half2_rn(-127.f);
  const __half2 c254 = __float2half2_rn(254.f);
  const __half2 neg_min =
      __half2half2(-cuda::std::numeric_limits<__half>::infinity());

  for (int batch_i = batch_i_; batch_i < batch; batch_i += stride) {
#pragma unroll
    for (int b_i = 0; b_i < num_batch_per_thr; b_i++) {
      thr_max[b_i] = neg_min;
#pragma unroll
      for (int c_i = 0; c_i < num_elements_per_thr; c_i++) {
        if ((col_i + c_i * kWarpSize < num_elements) &&
            (batch_i + b_i < batch)) {
          int32_4 inp =
              input[num_elements * (batch_i + b_i) + col_i + c_i * kWarpSize];
          buf[b_i][2 * c_i] =
              __floats2half2_rn(inp.x * scale_i, inp.y * scale_i);
          buf[b_i][2 * c_i + 1] =
              __floats2half2_rn(inp.z * scale_i, inp.w * scale_i);
          thr_max[b_i] = max(thr_max[b_i], buf[b_i][2 * c_i]);
          thr_max[b_i] = max(thr_max[b_i], buf[b_i][2 * c_i + 1]);
        } else {
          buf[b_i][2 * c_i] = neg_min;
          buf[b_i][2 * c_i + 1] = neg_min;
        }
      }
    }

#pragma unroll
    for (int b_i = 0; b_i < num_batch_per_thr; b_i++) {
#pragma unroll
      for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
        thr_max[b_i] =
            max(thr_max[b_i], __shfl_xor_sync(0xffffffff, thr_max[b_i], mask));
      }
      thr_max[b_i] = max(thr_max[b_i], __lowhigh2highlow(thr_max[b_i]));

      sum[b_i] = __float2half2_rn(0.f);
#pragma unroll
      for (int c_i = 0; c_i < num_elements_per_thr; c_i++) {
        buf[b_i][2 * c_i] = exp(__hsub2(buf[b_i][2 * c_i], thr_max[b_i]));
        buf[b_i][2 * c_i + 1] =
            exp(__hsub2(buf[b_i][2 * c_i + 1], thr_max[b_i]));
        sum[b_i] = __hadd2(sum[b_i], buf[b_i][2 * c_i]);
        sum[b_i] = __hadd2(sum[b_i], buf[b_i][2 * c_i + 1]);
      }

#pragma unroll
      for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
        sum[b_i] =
            __hadd2(sum[b_i], __shfl_xor_sync(0xffffffff, sum[b_i], mask));
      }
      sum[b_i] = __hadd2(sum[b_i], __lowhigh2highlow(sum[b_i]));
    }

#pragma unroll
    for (int b_i = 0; b_i < num_batch_per_thr; b_i++) {
      int index = num_elements * (batch_i + b_i);
#pragma unroll
      for (int c_i = 0; c_i < num_elements_per_thr; c_i++) {
        if ((col_i + c_i * kWarpSize < num_elements) &&
            (batch_i + b_i < batch)) {
          const __half2 out_1 = __hfma2(buf[b_i][2 * c_i], c254, c127);
          const __half2 out_2 = __hfma2(buf[b_i][2 * c_i + 1], c254, c127);
          int8_4 out;
          out.x = T2int8(__low2half(out_1));
          out.y = T2int8(__high2half(out_1));
          out.z = T2int8(__low2half(out_2));
          out.w = T2int8(__high2half(out_2));
          output[index + col_i + c_i * kWarpSize] = out;
        }
      }
      if (col_i == 0) {
        sum_each_line[batch_i + b_i] = __low2float(sum[b_i]);
      }
    }
  }
}

__global__ void int32cpy2int8(int32_4 *__restrict__ input,
                              int8_4 *__restrict__ output, const float scale_o,
                              const int n, float *__restrict__ sum_each_line,
                              int embed_dim) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int32_4 inp = input[index];
    const int sum_index = index / embed_dim;
    int8_4 out;
    out.x = T2int8(inp.x * scale_o / sum_each_line[sum_index]);
    out.y = T2int8(inp.y * scale_o / sum_each_line[sum_index]);
    out.z = T2int8(inp.z * scale_o / sum_each_line[sum_index]);
    out.w = T2int8(inp.w * scale_o / sum_each_line[sum_index]);

    output[index] = out;
  }
}

template <typename T>
int qkv(T *workspace, const T *query, const T *key, const T *value, T *output,
        const int &batch, const int &q_len, const int &kv_len,
        const int &embed_dim, cublasHandle_t cublasHandle, cudaStream_t stream,
        cublasGemmAlgo_t algo_1, cublasGemmAlgo_t algo_2) {
  const T sqrt_d = 1.f / std::sqrt((float)embed_dim);
  const T alpha = 1.f;
  const T beta = 0.f;
  const int stride_q = q_len * embed_dim;
  const int stride_kv = kv_len * embed_dim;
  if (kv_len > 1024) {
    printf("Softmax failure %s:%d: '%s'\n", __FILE__, __LINE__,
           "softmax_elements should be lower then 1024.");
    exit(1);
  }
  static_assert(THREADS_PER_BLOCK % kWarpSize == 0, "");

  cublasStatus_t status;

  status = cublasGemmStridedBatchedWrap<T>(
      cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, kv_len, q_len, embed_dim, &sqrt_d,
      key, embed_dim, stride_kv, query, embed_dim, stride_q, &beta, workspace,
      kv_len, q_len * kv_len, batch, algo_1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    return 1;
  }
#define DEFINE_ONE_ELIF(col)                                                   \
  else if (kv_len <= col * kWarpSize) {                                        \
    softmaxWarpKernel<T, 1, col>                                               \
        <<<dim3(GET_BLOCKS(q_len *batch *kWarpSize)),                          \
           dim3(kWarpSize, THREADS_PER_BLOCK / kWarpSize), 0, stream>>>(       \
            workspace, q_len * batch, kv_len);                                 \
  }
  if (kv_len <= 0) {
    exit(1);
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(3)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(5)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(7)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(9)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(11)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(13)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(15)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(17)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(19)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(21)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(23)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(25)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(27)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(29)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(31)
  DEFINE_ONE_ELIF(32)
  else {
    exit(1);
  }

  cudaCheckError();
  status = cublasGemmStridedBatchedWrap<T>(
      cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, embed_dim, q_len, kv_len, &alpha,
      value, embed_dim, stride_kv, workspace, kv_len, q_len * kv_len, &beta,
      output, embed_dim, stride_q, batch, algo_2);

  if (status != CUBLAS_STATUS_SUCCESS) {
    return 1;
  }

  return 0;
}

template <>
int qkv(__half2 *workspace, const __half2 *query, const __half2 *key,
        const __half2 *value, __half2 *output, const int &batch,
        const int &q_len, const int &kv_len, const int &embed_dim,
        cublasHandle_t cublasHandle, cudaStream_t stream,
        cublasGemmAlgo_t algo_1, cublasGemmAlgo_t algo_2) {
  const __half sqrt_d = 1.f / std::sqrt((float)embed_dim);
  const __half alpha = 1.f;
  const __half beta = 0.f;
  const int stride_q = q_len * embed_dim;
  const int stride_kv = kv_len * embed_dim;
  if (kv_len % 2 != 0) {
    exit(1);
  }
  if (kv_len > 1024) {
    printf("Softmax failure %s:%d: '%s'\n", __FILE__, __LINE__,
           "softmax_elements should be lower then 1024.");
    exit(1);
  }
  static_assert(THREADS_PER_BLOCK % kWarpSize == 0, "");

  cublasStatus_t status;

  status = cublasGemmStridedBatchedWrap<__half>(
      cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, kv_len, q_len, embed_dim, &sqrt_d,
      (__half *)key, embed_dim, stride_kv, (__half *)query, embed_dim, stride_q,
      &beta, (__half *)workspace, kv_len, q_len * kv_len, batch, algo_1);
#define DEFINE_ONE_ELIF_H2(col)                                                \
  else if (kv_len <= col * kWarpSize) {                                        \
    softmaxWarpKernel<1, col / 2>                                              \
        <<<dim3(GET_BLOCKS(q_len *batch *kWarpSize)),                          \
           dim3(kWarpSize, THREADS_PER_BLOCK / kWarpSize), 0, stream>>>(       \
            workspace, q_len * batch, kv_len / 2);                             \
  }
  if (kv_len <= 0) {
    exit(1);
  }
  DEFINE_ONE_ELIF_H2(2)
  DEFINE_ONE_ELIF_H2(4)
  DEFINE_ONE_ELIF_H2(6)
  DEFINE_ONE_ELIF_H2(8)
  DEFINE_ONE_ELIF_H2(10)
  DEFINE_ONE_ELIF_H2(12)
  DEFINE_ONE_ELIF_H2(14)
  DEFINE_ONE_ELIF_H2(16)
  DEFINE_ONE_ELIF_H2(18)
  DEFINE_ONE_ELIF_H2(20)
  DEFINE_ONE_ELIF_H2(22)
  DEFINE_ONE_ELIF_H2(24)
  DEFINE_ONE_ELIF_H2(26)
  DEFINE_ONE_ELIF_H2(28)
  DEFINE_ONE_ELIF_H2(30)
  DEFINE_ONE_ELIF_H2(32)
  else {
    exit(1);
  }

  cudaCheckError();
  status = cublasGemmStridedBatchedWrap<__half>(
      cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, embed_dim, q_len, kv_len, &alpha,
      (__half *)value, embed_dim, stride_kv, (__half *)workspace, kv_len,
      q_len * kv_len, &beta, (__half *)output, embed_dim, stride_q, batch,
      algo_2);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return 1;
  }

  return 0;
}

int qkv_int8(int8_4 *workspace, const int8_4 *query, const float &scale_q,
             const int8_4 *key, const float &scale_k, const int8_4 *value,
             const float &scale_v, int8_4 *output, const float &scale_o,
             const int &batch, const int &q_len, const int &kv_len,
             const int &embed_dim, cublasHandle_t cublasHandle,
             cudaStream_t stream, cublasGemmAlgo_t algo_1,
             cublasGemmAlgo_t algo_2) {
  const float scale_qk = scale_q * scale_k / std::sqrt((float)embed_dim);
  const int32_t alpha = 1;
  const int32_t beta = 0;
  const int stride_q = q_len * embed_dim;
  const int stride_kv = kv_len * embed_dim;
  if (kv_len % 4 != 0) {
    exit(1);
  }
  if (kv_len > 1024) {
    printf("Softmax failure %s:%d: '%s'\n", __FILE__, __LINE__,
           "softmax_elements should be lower then 1024.");
    exit(1);
  }
  static_assert(THREADS_PER_BLOCK % kWarpSize == 0, "");

  cublasStatus_t status;

  status = cublasGemmStridedBatchedWrap_int8(
      cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, kv_len, q_len, embed_dim, &alpha,
      (int8_t *)key, embed_dim, stride_kv, (int8_t *)query, embed_dim, stride_q,
      &beta, (int32_t *)workspace, kv_len, q_len * kv_len, batch, algo_1);
  int8_4 *softmax_out_ws = (int8_4 *)(workspace + batch * q_len * kv_len);
  int8_t *c127Mat = (int8_t *)softmax_out_ws + batch * q_len * kv_len;
  float *sum_each_line = (float *)(c127Mat + batch * q_len * kv_len);
  cudaMemsetAsync(c127Mat, 127, batch * q_len * kv_len * sizeof(int8_t),
                  stream);

#define DEFINE_ONE_ELIF_INT8(col)                                              \
  else if (kv_len <= col * kWarpSize) {                                        \
    softmaxWarpInt8Kernel<1, col / 4>                                          \
        <<<dim3(GET_BLOCKS(q_len *batch *kWarpSize)),                          \
           dim3(kWarpSize, THREADS_PER_BLOCK / kWarpSize), 0, stream>>>(       \
            (int32_4 *)workspace, scale_qk, softmax_out_ws, q_len * batch,     \
            kv_len / 4, sum_each_line);                                        \
  }
  if (kv_len <= 0) {
    exit(1);
  }
  DEFINE_ONE_ELIF_INT8(4)
  DEFINE_ONE_ELIF_INT8(8)
  DEFINE_ONE_ELIF_INT8(12)
  DEFINE_ONE_ELIF_INT8(16)
  DEFINE_ONE_ELIF_INT8(20)
  DEFINE_ONE_ELIF_INT8(24)
  DEFINE_ONE_ELIF_INT8(28)
  DEFINE_ONE_ELIF_INT8(32)
  else {
    exit(1);
  }

  cudaCheckError();
  status = cublasGemmStridedBatchedWrap_int8(
      cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, embed_dim, q_len, kv_len, &alpha,
      (int8_t *)value, embed_dim, stride_kv, (int8_t *)c127Mat, kv_len,
      q_len * kv_len, &beta, (int32_t *)workspace, embed_dim, stride_q, batch,
      algo_2);
  status = cublasGemmStridedBatchedWrap_int8(
      cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, embed_dim, q_len, kv_len, &alpha,
      (int8_t *)value, embed_dim, stride_kv, (int8_t *)softmax_out_ws, kv_len,
      q_len * kv_len, &alpha, (int32_t *)workspace, embed_dim, stride_q, batch,
      algo_2);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return 1;
  }
  int32cpy2int8<<<GET_BLOCKS(batch * q_len * embed_dim / 4),
                  THREADS_PER_BLOCK>>>(
      (int32_4 *)workspace, output, scale_v / (254 * scale_o),
      batch * q_len * embed_dim / 4, sum_each_line, embed_dim);
  cudaCheckError();
}

template int qkv(float *workspace, const float *query, const float *key,
                 const float *value, float *output, const int &batch,
                 const int &q_len, const int &kv_len, const int &embed_dim,
                 cublasHandle_t cublasHandle, cudaStream_t stream,
                 cublasGemmAlgo_t algo_1, cublasGemmAlgo_t algo_2);
template int qkv(__half *workspace, const __half *query, const __half *key,
                 const __half *value, __half *output, const int &batch,
                 const int &q_len, const int &kv_len, const int &embed_dim,
                 cublasHandle_t cublasHandle, cudaStream_t stream,
                 cublasGemmAlgo_t algo_1, cublasGemmAlgo_t algo_2);
template int qkv(__half2 *workspace, const __half2 *query, const __half2 *key,
                 const __half2 *value, __half2 *output, const int &batch,
                 const int &q_len, const int &kv_len, const int &embed_dim,
                 cublasHandle_t cublasHandle, cudaStream_t stream,
                 cublasGemmAlgo_t algo_1, cublasGemmAlgo_t algo_2);
