//
// Created by Derry Lin on 2022/10/22.
//
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/pytorch/pytorch/blob/ec683299ebabf297a3504c76248d37be830e4342/aten/src/ATen/native/cuda/GridSampler.cuh
// and
// https://github.com/pytorch/pytorch/blob/ec683299ebabf297a3504c76248d37be830e4342/aten/src/ATen/native/cuda/GridSampler.cu

#include <cstdio>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "cuda_helper.h"
#include "gridSamplerKernel.h"
#include "helper.h"
#include <cstdio>

using helper::TensorDesc;

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

__forceinline__ __device__ int8_t half2int8(const __half &hval,
                                            const float &scale) {
  __half ret = __hdiv(hval, __float2half(scale));
  return T2int8<__half>(ret);
}

__forceinline__ __device__ void qmulf(const int8_4 &a, int8_4 &c,
                                      const float &b) {
  c.x = T2int8<float>(a.x * b);
  c.y = T2int8<float>(a.y * b);
  c.z = T2int8<float>(a.z * b);
  c.w = T2int8<float>(a.w * b);
}

__forceinline__ __device__ void dp4a(const int32_t *a, const int32_t *b,
                                     int32_t &c) {
#if __CUDA_ARCH__ >= 610
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "+r"(c) : "r"(*a), "r"(*b), "r"(c));
#else
  auto ap = (int8_4 *)a, bp = (int8_4 *)b;

  c += ap->x * bp->x;
  c += ap->y * bp->y;
  c += ap->z * bp->z;
  c += ap->w * bp->w;
#endif
}

// Unnormalizes a coordinate from the -1 to +1 scale to its pixel index value,
// where we view each pixel as an area between (idx - 0.5) and (idx + 0.5).
// if align_corners: -1 and +1 get sent to the centers of the corner pixels
//     -1 --> 0
//     +1 --> (size - 1)
//     scale_factor = (size - 1) / 2
// if not align_corners: -1 and +1 get sent to the image edges
//     -1 --> -0.5
//     +1 --> (size - 1) + 0.5 == size - 0.5
//     scale_factor = size / 2
template <typename scalar_t>
__forceinline__ __device__ scalar_t
grid_sampler_unnormalize(scalar_t coord, int size, bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-10, 10] to [0, size - 1]
    return ((coord + 10.f) / 2) * (static_cast<float>(size - 1) / 10.f);
  } else {
    // unnormalize coord from [-10, 10] to [-0.5, size - 0.5]
    return ((coord + 10.f) / 2) * (static_cast<float>(size) / 10.f) - 0.5;
  }
}

template <>
__forceinline__ __device__ __half grid_sampler_unnormalize(__half coord,
                                                           int size,
                                                           bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-10, 10] to [0, size - 1]
    return __hmul(
        __hfma(static_cast<__half>(0.5), coord, static_cast<__half>(5.)),
        static_cast<__half>((size - 1) * 0.1));
    //        return __hmul(__hfma(static_cast<__half>(0.5), coord,
    //        static_cast<__half>(0.5)), static_cast<__half>(size-1));
    //      return ((coord + 1.) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-10, 10] to [-0.5, size - 0.5]
    return __hfma(
        __hfma(static_cast<__half>(0.5), coord, static_cast<__half>(5.)),
        static_cast<__half>(size * 0.1), static_cast<__half>(-0.5));
    //        return __hfma(__hfma(coord, static_cast<__half>(size),
    //        static_cast<__half>(size)), static_cast<__half>(0.5),
    //        static_cast<__half>(-0.5));
    //      return ((coord + 1.) * size - 1) / 2;
  }
}

template <>
__forceinline__ __device__ __half2
grid_sampler_unnormalize(__half2 coord, int size, bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-10, 10] to [0, size - 1]
    return __hmul2(__hfma2(__float2half2_rn(0.5), coord, __float2half2_rn(5.)),
                   __float2half2_rn((size - 1) * 0.1));
    //        return __hmul2(__hfma2(__float2half2_rn(0.5), coord,
    //        __float2half2_rn(0.5)),
    //        __float2half2_rn(static_cast<float>(size-1)));
    //      return ((coord + 1.) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-10, 10] to [-0.5, size - 0.5]
    return __hfma2(__hfma2(__float2half2_rn(0.5), coord, __float2half2_rn(5.)),
                   __float2half2_rn(size * 0.1), __float2half2_rn(-0.5));
    //        return __hfma2(__hfma2(coord,
    //        __float2half2_rn(static_cast<float>(size)),
    //        __float2half2_rn(static_cast<float>(size))),
    //        __float2half2_rn(0.5), __float2half2_rn(-0.5));
    //      return ((coord + 1.) * size - 1) / 2;
  }
}

__forceinline__ __device__ __half2
grid_sampler_unnormalize_h2(__half2 coord, __half2 size, bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-10, 10] to [0, size - 1]
    return __hmul2(
        __hfma2(__float2half2_rn(0.5), coord, __float2half2_rn(5.)),
        __hfma2(size, __float2half2_rn(0.1), __float2half2_rn(-0.1)));
  } else {
    // unnormalize coord from [-10, 10] to [-0.5, size - 0.5]
    return __hfma2(__hfma2(__float2half2_rn(0.5), coord, __float2half2_rn(5.)),
                   __hmul2(size, __float2half2_rn(0.1)),
                   __float2half2_rn(-0.5));
  }
}

// Clips coordinates to between 0 and clip_limit - 1
template <typename scalar_t>
__forceinline__ __device__ scalar_t clip_coordinates(scalar_t in,
                                                     int clip_limit) {
  return ::min(static_cast<scalar_t>(clip_limit - 1),
               ::max(in, static_cast<scalar_t>(0)));
}

#if __CUDA_ARCH__ >= 800
template <>
__forceinline__ __device__ __half clip_coordinates(__half in, int clip_limit) {
  return __hmin(static_cast<__half>(clip_limit - 1),
                __hmax(in, static_cast<__half>(0)));
}

template <>
__forceinline__ __device__ __half2 clip_coordinates(__half2 in,
                                                    int clip_limit) {
  return __hmin2(__float2half2_rn(static_cast<float>(clip_limit - 1)),
                 __hmax2(in, __float2half2_rn(0.)));
}

__forceinline__ __device__ __half2 clip_coordinates_h2(__half2 in,
                                                       __half2 clip_limit) {
  return __hmin2(__hadd2(clip_limit, __float2half2_rn(-1.f)),
                 __hmax2(in, __float2half2_rn(0.)));
}
#else
static __forceinline__ __device__ __half hmin(const __half a, const __half b) {
  return __hgt(a, b) ? b : a;
}

static __forceinline__ __device__ __half hmax(const __half a, const __half b) {
  return __hgt(a, b) ? a : b;
}

template <>
__forceinline__ __device__ __half clip_coordinates(__half in, int clip_limit) {
  return hmin(static_cast<__half>(clip_limit - 1),
              hmax(in, static_cast<__half>(0)));
}

static __forceinline__ __device__ __half2 hmin2(const __half2 a,
                                                const __half2 b) {
  return __hmul2(__hge2(a, b), b) + __hmul2(__hlt2(a, b), a);
}

static __forceinline__ __device__ __half2 hmax2(const __half2 a,
                                                const __half2 b) {
  return __hmul2(__hge2(a, b), a) + __hmul2(__hlt2(a, b), b);
  //    return __hgt2(a, b) ? a : b;
}

template <>
__forceinline__ __device__ __half2 clip_coordinates(__half2 in,
                                                    int clip_limit) {
  return hmin2(__float2half2_rn(static_cast<float>(clip_limit - 1)),
               hmax2(in, __float2half2_rn(0.)));
}

__forceinline__ __device__ __half2 clip_coordinates_h2(__half2 in,
                                                       __half2 clip_limit) {
  return hmin2(__hadd2(clip_limit, __float2half2_rn(-1.f)),
               hmax2(in, __float2half2_rn(0.)));
}
#endif

// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
template <typename scalar_t>
__forceinline__ __device__ scalar_t reflect_coordinates(scalar_t in,
                                                        int twice_low,
                                                        int twice_high) {
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = ::fabs(in - min);
  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  scalar_t extra = ::fmod(in, span);
  int flips = static_cast<int>(::floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

static __forceinline__ __device__ __half hmod(const __half a, const __half b) {
  return __hsub(a, __hmul(htrunc(__hdiv(a, b)), b));
}

template <>
__forceinline__ __device__ __half reflect_coordinates(__half in, int twice_low,
                                                      int twice_high) {
  if (twice_low == twice_high) {
    return static_cast<__half>(0);
  }
  __half min = __hmul(static_cast<__half>(twice_low), static_cast<__half>(0.5));
  __half span = __hmul(static_cast<__half>(twice_high - twice_low),
                       static_cast<__half>(0.5));
  in = __habs(__hsub(in, min));

  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  __half extra = hmod(in, span);
  int flips = static_cast<int>(hfloor(__hdiv(in, span)));
  if (flips % 2 == 0) {
    return __hadd(extra, min);
  } else {
    return __hadd(__hsub(span, extra), min);
  }
}

static __forceinline__ __device__ __half2 hmod2(const __half2 a,
                                                const __half2 b) {
  return __hsub2(a, __hmul2(h2trunc(__h2div(a, b)), b));
}

template <>
__forceinline__ __device__ __half2 reflect_coordinates(__half2 in,
                                                       int twice_low,
                                                       int twice_high) {
  if (twice_low == twice_high) {
    return __float2half2_rn(0.);
  }
  __half2 min = __hmul2(__float2half2_rn(static_cast<float>(twice_low)),
                        __float2half2_rn(0.5));
  __half2 span =
      __hmul2(__float2half2_rn(static_cast<float>(twice_high - twice_low)),
              __float2half2_rn(0.5));
  in = __habs2(__hsub2(in, min));

  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  __half2 extra = hmod2(in, span);
  __half2 flips_h2 = h2floor(__h2div(in, span));
  int flips = static_cast<int>(__low2float(flips_h2));
  __half low, high;
  if (flips % 2 == 0) {
    low = __hadd(__low2half(extra), __low2half(min));
  } else {
    low = __hadd(__low2half(__hsub2(span, extra)), __low2half(min));
  }
  flips = static_cast<int>(__high2float(flips_h2));
  if (flips % 2 == 0) {
    high = __hadd(__high2half(extra), __high2half(min));
  } else {
    high = __hadd(__high2half(__hsub2(span, extra)), __high2half(min));
  }
  return __halves2half2(low, high);
}

__forceinline__ __device__ __half2 reflect_coordinates_h2(__half2 in,
                                                          __half2 twice_low,
                                                          __half2 twice_high) {
  __half2 weight = __hne2(twice_low, twice_high);
  __half2 min = __hmul2(twice_low, __float2half2_rn(0.5));
  __half2 span = __hmul2(__hsub2(twice_high, twice_low), __float2half2_rn(0.5));
  in = __habs2(__hsub2(in, min));

  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  __half2 extra = hmod2(in, span);
  __half2 flips_h2 = h2floor(__h2div(in, span));

  flips_h2 = h2trunc(flips_h2);
  flips_h2 = __heq2(__hmul2(h2rint(__hmul2(flips_h2, __float2half2_rn(0.5f))),
                            __float2half2_rn(2.f)),
                    flips_h2);
  __half2 even = __hadd2(extra, min);
  __half2 odd = __hadd2(__hsub2(span, extra), min);
  flips_h2 = __hmul2(even, flips_h2) +
             __hmul2(odd, __hsub2(__float2half2_rn(1.f), flips_h2));

  return __hmul2(flips_h2, weight);
}

template <typename scalar_t>
__forceinline__ __device__ scalar_t safe_downgrade_to_int_range(scalar_t x) {
  // -100.0 does not have special meaning. This is just to make sure
  // it's not within_bounds_2d or within_bounds_3d, and does not cause
  // undefined behavior. See #35506.
  if (x > INT_MAX - 1 || x < INT_MIN || !::isfinite(static_cast<double>(x)))
    return static_cast<scalar_t>(-100.0);
  return x;
}

template <>
__forceinline__ __device__ __half safe_downgrade_to_int_range(__half x) {
  // -100.0 does not have special meaning. This is just to make sure
  // it's not within_bounds_2d or within_bounds_3d, and does not cause
  // undefined behavior. See #35506.
  if (__hisinf(x))
    return static_cast<__half>(-100.0);
  return x;
}

template <>
__forceinline__ __device__ __half2 safe_downgrade_to_int_range(__half2 x) {
  // -100.0 does not have special meaning. This is just to make sure
  // it's not within_bounds_2d or within_bounds_3d, and does not cause
  // undefined behavior. See #35506.
  __half low, high;

  low = __low2half(x);
  if (__hisinf(low)) {
    low = static_cast<__half>(-100.0);
  }

  high = __high2half(x);
  if (__hisinf(high)) {
    high = static_cast<__half>(-100.0);
  }

  return __halves2half2(low, high);
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t
compute_coordinates(scalar_t coord, int size, GridSamplerPadding padding_mode,
                    bool align_corners) {
  if (padding_mode == GridSamplerPadding::Border) {
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2 * (size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2 * size - 1);
    }
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  }

  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

static __forceinline__ __device__ __half2
compute_coordinates_h2(__half2 coord, __half2 size,
                       GridSamplerPadding padding_mode, bool align_corners) {
  if (padding_mode == GridSamplerPadding::Border) {
    // clip coordinates to image borders
    coord = clip_coordinates_h2(coord, size);
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    // reflect coordinates by image borders

    if (align_corners) {
      coord = reflect_coordinates_h2(
          coord, __float2half2_rn(0.f),
          __hfma2(size, __float2half2_rn(2.f), __float2half2_rn(-2.f)));
    } else {
      coord = reflect_coordinates_h2(
          coord, __float2half2_rn(-1.f),
          __hfma2(size, __float2half2_rn(2.f), __float2half2_rn(-1.f)));
    }
    // clip coordinates to image borders
    coord = clip_coordinates_h2(coord, size);
  }

  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

// Computes the pixel source index value for a grid coordinate
static __forceinline__ __device__ __half2 grid_sampler_compute_source_index_h2(
    __half2 coord, __half2 size, const GridSamplerPadding padding_mode,
    const bool align_corners) {
  coord = grid_sampler_unnormalize_h2(coord, size, align_corners);
  coord = compute_coordinates_h2(coord, size, padding_mode, align_corners);
  return coord;
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t grid_sampler_compute_source_index(
    scalar_t coord, int size, const GridSamplerPadding padding_mode,
    const bool align_corners) {
  coord = grid_sampler_unnormalize(coord, size, align_corners);
  coord = compute_coordinates(coord, size, padding_mode, align_corners);
  return coord;
}

static __forceinline__ __device__ bool within_bounds_2d(int h, int w, int H,
                                                        int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

static __forceinline__ __device__ bool within_bounds_3d(int d, int h, int w,
                                                        int D, int H, int W) {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

static __forceinline__ __device__ __half2 within_bounds_3d_half2(
    __half2 d, __half2 h, __half2 w, __half2 D, __half2 H, __half2 W) {
  return __hmul2(
      __hmul2(__hmul2(__hge2(h, __float2half2_rn(0.)), __hlt2(h, H)),
              __hmul2(__hge2(w, __float2half2_rn(0.)), __hlt2(w, W))),
      __hmul2(__hge2(d, __float2half2_rn(0.)), __hlt2(d, D)));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t cubic_convolution1(scalar_t x, scalar_t A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t cubic_convolution2(scalar_t x, scalar_t A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <>
__device__ __forceinline__ __half cubic_convolution1(__half x, __half A) {
  return __hadd(__hmul(__hsub(__hmul(__hadd(A, __float2half(2.f)), x),
                              __hadd(A, __float2half(3.f))),
                       __hmul(x, x)),
                __float2half(1.f));
}

template <>
__device__ __forceinline__ __half cubic_convolution2(__half x, __half A) {
  return __hsub(
      __hmul(
          __hadd(__hmul(__hsub(__hmul(A, x), __hmul(A, __float2half(5.f))), x),
                 __hmul(A, __float2half(8.f))),
          x),
      __hmul(A, __float2half(4.f)));
}

template <>
__device__ __forceinline__ __half2 cubic_convolution1(__half2 x, __half2 A) {
  return __hadd2(__hmul2(__hsub2(__hmul2(__hadd2(A, __float2half2_rn(2.f)), x),
                                 __hadd2(A, __float2half2_rn(3.f))),
                         __hmul2(x, x)),
                 __float2half2_rn(1.f));
}

template <>
__device__ __forceinline__ __half2 cubic_convolution2(__half2 x, __half2 A) {
  return __hsub2(
      __hmul2(__hadd2(__hmul2(__hsub2(__hmul2(A, x),
                                      __hmul2(A, __float2half2_rn(5.f))),
                              x),
                      __hmul2(A, __float2half2_rn(8.f))),
              x),
      __hmul2(A, __float2half2_rn(4.f)));
}

template <typename scalar_t>
__device__ __forceinline__ void
get_cubic_upsampling_coefficients(scalar_t coeffs[4], scalar_t t) {
  scalar_t A = -0.75;

  scalar_t x1 = t;
  coeffs[0] = cubic_convolution2<scalar_t>(x1 + 1.0, A);
  coeffs[1] = cubic_convolution1<scalar_t>(x1, A);

  // opposite coefficients
  scalar_t x2 = 1.0 - t;
  coeffs[2] = cubic_convolution1<scalar_t>(x2, A);
  coeffs[3] = cubic_convolution2<scalar_t>(x2 + 1.0, A);
}

template <>
__device__ __forceinline__ void
get_cubic_upsampling_coefficients(__half coeffs[4], __half t) {
  __half A = __float2half(-0.75);

  __half x1 = t;
  coeffs[0] = cubic_convolution2<__half>(__hadd(x1, __float2half(1.0)), A);
  coeffs[1] = cubic_convolution1<__half>(x1, A);

  // opposite coefficients
  __half x2 = __hsub(__float2half(1.0), t);
  coeffs[2] = cubic_convolution1<__half>(x2, A);
  coeffs[3] = cubic_convolution2<__half>(__hadd(x2, __float2half(1.0)), A);
}

template <>
__device__ __forceinline__ void
get_cubic_upsampling_coefficients(__half2 coeffs[4], __half2 t) {
  __half2 A = __float2half2_rn(-0.75);

  __half2 x1 = t;
  coeffs[0] =
      cubic_convolution2<__half2>(__hadd2(x1, __float2half2_rn(1.0)), A);
  coeffs[1] = cubic_convolution1<__half2>(x1, A);

  // opposite coefficients
  __half2 x2 = __hsub2(__float2half2_rn(1.0), t);
  coeffs[2] = cubic_convolution1<__half2>(x2, A);
  coeffs[3] =
      cubic_convolution2<__half2>(__hadd2(x2, __float2half2_rn(1.0)), A);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t cubic_interp1d(scalar_t x0, scalar_t x1,
                                                   scalar_t x2, scalar_t x3,
                                                   scalar_t t) {
  scalar_t coeffs[4];
  get_cubic_upsampling_coefficients<scalar_t>(coeffs, t);

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

template <>
__device__ __forceinline__ __half cubic_interp1d(__half x0, __half x1,
                                                 __half x2, __half x3,
                                                 __half t) {
  __half coeffs[4];
  get_cubic_upsampling_coefficients<__half>(coeffs, t);
  return __hadd(__hadd(__hmul(x0, coeffs[0]), __hmul(x1, coeffs[1])),
                __hadd(__hmul(x2, coeffs[2]), __hmul(x3, coeffs[3])));
}

template <>
__device__ __forceinline__ __half2 cubic_interp1d(__half2 x0, __half2 x1,
                                                  __half2 x2, __half2 x3,
                                                  __half2 t) {
  __half2 coeffs[4];
  get_cubic_upsampling_coefficients<__half2>(coeffs, t);
  return __hadd2(__hadd2(__hmul2(x0, coeffs[0]), __hmul2(x1, coeffs[1])),
                 __hadd2(__hmul2(x2, coeffs[2]), __hmul2(x3, coeffs[3])));
}

__device__ __forceinline__ int8_4 cubic_interp1d_int8(int8_4 x0, int8_4 x1,
                                                      int8_4 x2, int8_4 x3,
                                                      float t) {
  float coeffs[4];
  get_cubic_upsampling_coefficients(coeffs, t);
  int8_4 weight = int8_4(int8_t(coeffs[0] * 127), int8_t(coeffs[1] * 127),
                         int8_t(coeffs[2] * 127), int8_t(coeffs[3] * 127));

  int8_4 x, output;
  int32_t temp;

  temp = 0;
  x = {x0.x, x1.x, x2.x, x3.x};
  dp4a((const int32_t *)&x, (const int32_t *)&weight, temp);
  output.x = int8_t(temp / 127);

  temp = 0;
  x = {x0.y, x1.y, x2.y, x3.y};
  dp4a((const int32_t *)&x, (const int32_t *)&weight, temp);
  output.y = int8_t(temp / 127);

  temp = 0;
  x = {x0.z, x1.z, x2.z, x3.z};
  dp4a((const int32_t *)&x, (const int32_t *)&weight, temp);
  output.z = int8_t(temp / 127);

  temp = 0;
  x = {x0.w, x1.w, x2.w, x3.w};
  dp4a((const int32_t *)&x, (const int32_t *)&weight, temp);
  output.w = int8_t(temp / 127);

  return output;
}

template <typename scalar_t>
__forceinline__ __device__ scalar_t get_value_bounded(
    const scalar_t *data, scalar_t x, scalar_t y, int W, int H, int sW, int sH,
    const GridSamplerPadding padding_mode, const bool align_corners) {

  x = compute_coordinates(x, W, padding_mode, align_corners);
  y = compute_coordinates(y, H, padding_mode, align_corners);

  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);

  if (within_bounds_2d(iy, ix, H, W)) {
    return data[iy * sH + ix * sW];
  }
  return static_cast<scalar_t>(0.f);
}

__forceinline__ __device__ __half2 get_value_bounded_h2(
    const __half2 *data, __half2 xy, int W, int H, int sW, int sH,
    const GridSamplerPadding padding_mode, const bool align_corners) {

  xy = compute_coordinates_h2(
      xy, __floats2half2_rn(static_cast<float>(W), static_cast<float>(H)),
      padding_mode, align_corners);

  int ix = static_cast<int>(__low2float(xy));
  int iy = static_cast<int>(__high2float(xy));

  if (within_bounds_2d(iy, ix, H, W)) {
    return data[iy * sH + ix * sW];
  }
  return __float2half2_rn(0.f);
}

__forceinline__ __device__ int8_4 get_value_bounded_int8(
    const int8_4 *data, __half2 xy, int W, int H, int sW, int sH,
    const GridSamplerPadding padding_mode, const bool align_corners) {

  xy = compute_coordinates_h2(
      xy, __floats2half2_rn(static_cast<float>(W), static_cast<float>(H)),
      padding_mode, align_corners);

  int ix = static_cast<int>(__low2float(xy));
  int iy = static_cast<int>(__high2float(xy));

  if (within_bounds_2d(iy, ix, H, W)) {
    return data[iy * sH + ix * sW];
  }
  return 0;
}

template <typename scalar_t>
__global__ void grid_sampler_2d_kernel(
    const int nthreads, const scalar_t *input, const scalar_t *grid,
    scalar_t *output, TensorDesc input_desc, TensorDesc grid_desc,
    TensorDesc output_desc, const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode, const bool align_corners) {
  int C = input_desc.shape[1];
  int inp_H = input_desc.shape[2];
  int inp_W = input_desc.shape[3];
  int out_H = grid_desc.shape[2];
  int out_W = grid_desc.shape[3];
  int inp_sN = input_desc.stride[0];
  int inp_sC = input_desc.stride[1];
  int inp_sH = input_desc.stride[2];
  int inp_sW = input_desc.stride[3];
  int grid_sN = grid_desc.stride[0];
  int grid_sCoor = grid_desc.stride[1];
  int grid_sH = grid_desc.stride[2];
  int grid_sW = grid_desc.stride[3];
  int out_sN = output_desc.stride[0];
  int out_sC = output_desc.stride[1];
  int out_sH = output_desc.stride[2];
  int out_sW = output_desc.stride[3];

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int w = index % out_W;
    const int h = (index / out_W) % out_H;
    const int n = index / (out_H * out_W);
    const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y coordinates from grid
    scalar_t grid_x = grid[grid_offset];
    scalar_t grid_y = grid[grid_offset + grid_sCoor];

    scalar_t ix = grid_sampler_compute_source_index(grid_x, inp_W, padding_mode,
                                                    align_corners);
    scalar_t iy = grid_sampler_compute_source_index(grid_y, inp_H, padding_mode,
                                                    align_corners);

    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      // get NE, NW, SE, SW pixel values from (x, y)
      int ix_nw = static_cast<int>(::floor(ix));
      int iy_nw = static_cast<int>(::floor(iy));
      int ix_ne = ix_nw + 1;
      int iy_ne = iy_nw;
      int ix_sw = ix_nw;
      int iy_sw = iy_nw + 1;
      int ix_se = ix_nw + 1;
      int iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      scalar_t nw = (ix_se - ix) * (iy_se - iy);
      scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
      scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
      scalar_t se = (ix - ix_nw) * (iy - iy_nw);

      // calculate bilinear weighted pixel value and set output pixel
      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        *out_ptr_NCHW = static_cast<scalar_t>(0);
        if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
        }
        if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
        }
        if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
        }
        if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      int ix_nearest = static_cast<int>(::round(ix));
      int iy_nearest = static_cast<int>(::round(iy));

      // assign nearest neighbor pixel value to output pixel
      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
          *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
        } else {
          *out_ptr_NCHW = static_cast<scalar_t>(0);
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
      ix = grid_sampler_unnormalize(grid_x, inp_W, align_corners);
      iy = grid_sampler_unnormalize(grid_y, inp_H, align_corners);

      scalar_t ix_nw = ::floor(ix);
      scalar_t iy_nw = ::floor(iy);

      const scalar_t tx = ix - ix_nw;
      const scalar_t ty = iy - iy_nw;

      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        scalar_t coefficients[4];

#pragma unroll 4
        for (int i = 0; i < 4; ++i) {
          coefficients[i] = cubic_interp1d(
              get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw - 1, iy_nw - 1 + i,
                                          inp_W, inp_H, inp_sW, inp_sH,
                                          padding_mode, align_corners),
              get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 0, iy_nw - 1 + i,
                                          inp_W, inp_H, inp_sW, inp_sH,
                                          padding_mode, align_corners),
              get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 1, iy_nw - 1 + i,
                                          inp_W, inp_H, inp_sW, inp_sH,
                                          padding_mode, align_corners),
              get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 2, iy_nw - 1 + i,
                                          inp_W, inp_H, inp_sW, inp_sH,
                                          padding_mode, align_corners),
              tx);
        }

        *out_ptr_NCHW = cubic_interp1d(coefficients[0], coefficients[1],
                                       coefficients[2], coefficients[3], ty);
      }
    }
  }
}

template <>
__global__ void grid_sampler_2d_kernel(
    const int nthreads, const __half *input, const __half *grid, __half *output,
    TensorDesc input_desc, TensorDesc grid_desc, TensorDesc output_desc,
    const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode, const bool align_corners) {
  int C = input_desc.shape[1];
  int inp_H = input_desc.shape[2];
  int inp_W = input_desc.shape[3];
  int out_H = grid_desc.shape[2];
  int out_W = grid_desc.shape[3];
  int inp_sN = input_desc.stride[0];
  int inp_sC = input_desc.stride[1];
  int inp_sH = input_desc.stride[2];
  int inp_sW = input_desc.stride[3];
  int grid_sN = grid_desc.stride[0];
  int grid_sCoor = grid_desc.stride[1];
  int grid_sH = grid_desc.stride[2];
  int grid_sW = grid_desc.stride[3];
  int out_sN = output_desc.stride[0];
  int out_sC = output_desc.stride[1];
  int out_sH = output_desc.stride[2];
  int out_sW = output_desc.stride[3];

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int w = index % out_W;
    const int h = (index / out_W) % out_H;
    const int n = index / (out_H * out_W);
    const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y coordinates from grid
    __half grid_x = grid[grid_offset];
    __half grid_y = grid[grid_offset + grid_sCoor];

    __half ix = grid_sampler_compute_source_index(grid_x, inp_W, padding_mode,
                                                  align_corners);
    __half iy = grid_sampler_compute_source_index(grid_y, inp_H, padding_mode,
                                                  align_corners);

    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      // get NE, NW, SE, SW pixel values from (x, y)
      int ix_nw = static_cast<int>(hfloor(ix));
      int iy_nw = static_cast<int>(hfloor(iy));
      int ix_ne = ix_nw + 1;
      int iy_ne = iy_nw;
      int ix_sw = ix_nw;
      int iy_sw = iy_nw + 1;
      int ix_se = ix_nw + 1;
      int iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      __half nw = __hmul(__hsub(static_cast<__half>(ix_se), ix),
                         __hsub(static_cast<__half>(iy_se), iy));
      __half ne = __hmul(__hsub(ix, static_cast<__half>(ix_sw)),
                         __hsub(static_cast<__half>(iy_sw), iy));
      __half sw = __hmul(__hsub(static_cast<__half>(ix_ne), ix),
                         __hsub(iy, static_cast<__half>(iy_ne)));
      __half se = __hmul(__hsub(ix, static_cast<__half>(ix_nw)),
                         __hsub(iy, static_cast<__half>(iy_nw)));

      // calculate bilinear weighted pixel value and set output pixel
      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        *out_ptr_NCHW = static_cast<__half>(0);
        if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
          *out_ptr_NCHW = __hfma(inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW],
                                 nw, *out_ptr_NCHW);
        }
        if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
          *out_ptr_NCHW = __hfma(inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW],
                                 ne, *out_ptr_NCHW);
        }
        if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
          *out_ptr_NCHW = __hfma(inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW],
                                 sw, *out_ptr_NCHW);
        }
        if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
          *out_ptr_NCHW = __hfma(inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW],
                                 se, *out_ptr_NCHW);
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      int ix_nearest = static_cast<int>(hrint(ix));
      int iy_nearest = static_cast<int>(hrint(iy));

      // assign nearest neighbor pixel value to output pixel
      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
          *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
        } else {
          *out_ptr_NCHW = static_cast<__half>(0);
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
      ix = grid_sampler_unnormalize(grid_x, inp_W, align_corners);
      iy = grid_sampler_unnormalize(grid_y, inp_H, align_corners);

      __half ix_nw = hfloor(ix);
      __half iy_nw = hfloor(iy);

      const __half tx = __hsub(ix, ix_nw);
      const __half ty = __hsub(iy, iy_nw);

      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        __half coefficients[4];

#pragma unroll 4
        for (int i = 0; i < 4; ++i) {
          coefficients[i] = cubic_interp1d(
              get_value_bounded<__half>(
                  inp_ptr_NC, __hadd(ix_nw, __float2half(-1.f)),
                  __hadd(iy_nw, __float2half(i - 1.f)), inp_W, inp_H, inp_sW,
                  inp_sH, padding_mode, align_corners),
              get_value_bounded<__half>(
                  inp_ptr_NC, __hadd(ix_nw, __float2half(0.f)),
                  __hadd(iy_nw, __float2half(i - 1.f)), inp_W, inp_H, inp_sW,
                  inp_sH, padding_mode, align_corners),
              get_value_bounded<__half>(
                  inp_ptr_NC, __hadd(ix_nw, __float2half(1.f)),
                  __hadd(iy_nw, __float2half(i - 1.f)), inp_W, inp_H, inp_sW,
                  inp_sH, padding_mode, align_corners),
              get_value_bounded<__half>(
                  inp_ptr_NC, __hadd(ix_nw, __float2half(2.f)),
                  __hadd(iy_nw, __float2half(i - 1.f)), inp_W, inp_H, inp_sW,
                  inp_sH, padding_mode, align_corners),
              tx);
        }

        *out_ptr_NCHW = cubic_interp1d(coefficients[0], coefficients[1],
                                       coefficients[2], coefficients[3], ty);
      }
    }
  }
}

template <>
__global__ void grid_sampler_2d_kernel(
    const int nthreads, const __half2 *input, const __half2 *grid,
    __half2 *output, TensorDesc input_desc, TensorDesc grid_desc,
    TensorDesc output_desc, const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode, const bool align_corners) {
  int C = (input_desc.shape[1] + 1) / 2;
  int inp_H = input_desc.shape[2];
  int inp_W = input_desc.shape[3];
  int out_H = grid_desc.shape[2];
  int out_W = grid_desc.shape[3];
  int inp_sC = input_desc.stride[1];
  int inp_sH = input_desc.stride[2];
  int inp_sW = input_desc.stride[3];
  int inp_sN = inp_sC * C;
  int grid_sN = grid_desc.stride[0] / 2;
  int grid_sH = grid_desc.stride[2];
  int grid_sW = grid_desc.stride[3];
  int out_sC = output_desc.stride[1];
  int out_sH = output_desc.stride[2];
  int out_sW = output_desc.stride[3];
  int out_sN = out_sC * C;

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int w = index % out_W;
    const int h = (index / out_W) % out_H;
    const int n = index / (out_H * out_W);
    const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y coordinates from grid
    __half2 grid_xy = grid[grid_offset];

    __half2 ixy = grid_sampler_compute_source_index_h2(
        grid_xy,
        __floats2half2_rn(static_cast<float>(inp_W), static_cast<float>(inp_H)),
        padding_mode, align_corners);

    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      // get NE, NW, SE, SW pixel values from (x, y)
      __half ix = __low2half(ixy);
      __half iy = __high2half(ixy);
      ixy = h2floor(ixy);
      int ix_nw = static_cast<int>(__low2float(ixy));
      int iy_nw = static_cast<int>(__high2float(ixy));
      int ix_ne = ix_nw + 1;
      int iy_ne = iy_nw;
      int ix_sw = ix_nw;
      int iy_sw = iy_nw + 1;
      int ix_se = ix_nw + 1;
      int iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      __half2 nw = __half2half2(__hmul(__hsub(static_cast<__half>(ix_se), ix),
                                       __hsub(static_cast<__half>(iy_se), iy)));
      __half2 ne = __half2half2(__hmul(__hsub(ix, static_cast<__half>(ix_sw)),
                                       __hsub(static_cast<__half>(iy_sw), iy)));
      __half2 sw = __half2half2(__hmul(__hsub(static_cast<__half>(ix_ne), ix),
                                       __hsub(iy, static_cast<__half>(iy_ne))));
      __half2 se = __half2half2(__hmul(__hsub(ix, static_cast<__half>(ix_nw)),
                                       __hsub(iy, static_cast<__half>(iy_nw))));

      // calculate bilinear weighted pixel value and set output pixel
      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        *out_ptr_NCHW = __float2half2_rn(0.f);
        if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
          *out_ptr_NCHW = __hfma2(inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW],
                                  nw, *out_ptr_NCHW);
        }
        if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
          *out_ptr_NCHW = __hfma2(inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW],
                                  ne, *out_ptr_NCHW);
        }
        if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
          *out_ptr_NCHW = __hfma2(inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW],
                                  sw, *out_ptr_NCHW);
        }
        if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
          *out_ptr_NCHW = __hfma2(inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW],
                                  se, *out_ptr_NCHW);
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      ixy = h2rint(ixy);
      int ix_nearest = static_cast<int>(__low2float(ixy));
      int iy_nearest = static_cast<int>(__high2float(ixy));

      // assign nearest neighbor pixel value to output pixel
      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
          *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
        } else {
          *out_ptr_NCHW = __float2half2_rn(0.f);
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
      ixy = grid_sampler_unnormalize_h2(
          grid_xy,
          __floats2half2_rn(static_cast<float>(inp_W),
                            static_cast<float>(inp_H)),
          align_corners);

      __half2 ixy_nw = h2floor(ixy);
      const __half2 txy = __hsub2(ixy, ixy_nw);

      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        __half2 coefficients[4];

#pragma unroll 4
        for (int i = 0; i < 4; ++i) {
          coefficients[i] = cubic_interp1d(
              get_value_bounded_h2(
                  inp_ptr_NC, __hadd2(ixy_nw, __floats2half2_rn(-1.f, i - 1.f)),
                  inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
              get_value_bounded_h2(
                  inp_ptr_NC, __hadd2(ixy_nw, __floats2half2_rn(0.f, i - 1.f)),
                  inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
              get_value_bounded_h2(
                  inp_ptr_NC, __hadd2(ixy_nw, __floats2half2_rn(1.f, i - 1.f)),
                  inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
              get_value_bounded_h2(
                  inp_ptr_NC, __hadd2(ixy_nw, __floats2half2_rn(2.f, i - 1.f)),
                  inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
              __half2half2(__low2half(txy)));
        }

        *out_ptr_NCHW =
            cubic_interp1d(coefficients[0], coefficients[1], coefficients[2],
                           coefficients[3], __half2half2(__high2half(txy)));
      }
    }
  }
}

__global__ void grid_sampler_2d_kernel_int8(
    const int nthreads, const int8_4 *input, const float scale_i,
    const int8_4 *grid, const float scale_g, int8_4 *output,
    const float scale_o, TensorDesc input_desc, TensorDesc grid_desc,
    TensorDesc output_desc, const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode, const bool align_corners) {
  int C = (input_desc.shape[1] + 3) / 4;
  int inp_H = input_desc.shape[2];
  int inp_W = input_desc.shape[3];
  int out_H = grid_desc.shape[2];
  int out_W = grid_desc.shape[3];
  int inp_sC = input_desc.stride[1];
  int inp_sH = input_desc.stride[2];
  int inp_sW = input_desc.stride[3];
  int inp_sN = inp_sC * C;
  int grid_sN = grid_desc.stride[0] / 2;
  int grid_sH = grid_desc.stride[2];
  int grid_sW = grid_desc.stride[3];
  int out_sC = output_desc.stride[1];
  int out_sH = output_desc.stride[2];
  int out_sW = output_desc.stride[3];
  int out_sN = out_sC * C;

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int w = index % out_W;
    const int h = (index / out_W) % out_H;
    const int n = index / (out_H * out_W);
    const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y coordinates from grid
    int8_4 grid_xy_q = grid[grid_offset];
    __half2 grid_xy =
        __floats2half2_rn(grid_xy_q.x * scale_g, grid_xy_q.y * scale_g);

    __half2 ixy = grid_sampler_compute_source_index_h2(
        grid_xy,
        __floats2half2_rn(static_cast<float>(inp_W), static_cast<float>(inp_H)),
        padding_mode, align_corners);

    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      int32_t output_temp;
      // get NE, NW, SE, SW pixel values from (x, y)
      __half ix = __low2half(ixy);
      __half iy = __high2half(ixy);
      ixy = h2floor(ixy);
      int ix_nw = static_cast<int>(__low2float(ixy));
      int iy_nw = static_cast<int>(__high2float(ixy));
      int ix_ne = ix_nw + 1;
      int iy_ne = iy_nw;
      int ix_sw = ix_nw;
      int iy_sw = iy_nw + 1;
      int ix_se = ix_nw + 1;
      int iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      float scale_area = 1 / 127.f;
      float scale_out = scale_area * scale_i / scale_o;
      int8_4 weight;
      weight.x = half2int8(__hmul(__hsub(static_cast<__half>(ix_se), ix),
                                  __hsub(static_cast<__half>(iy_se), iy)),
                           scale_area);
      weight.y = half2int8(__hmul(__hsub(ix, static_cast<__half>(ix_sw)),
                                  __hsub(static_cast<__half>(iy_sw), iy)),
                           scale_area);
      weight.z = half2int8(__hmul(__hsub(static_cast<__half>(ix_ne), ix),
                                  __hsub(iy, static_cast<__half>(iy_ne))),
                           scale_area);
      weight.w = half2int8(__hmul(__hsub(ix, static_cast<__half>(ix_nw)),
                                  __hsub(iy, static_cast<__half>(iy_nw))),
                           scale_area);
      int8_4 inps[4];

      // calculate bilinear weighted pixel value and set output pixel
      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
          const int8_4 &inp = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
          inps[0].x = inp.x;
          inps[1].x = inp.y;
          inps[2].x = inp.z;
          inps[3].x = inp.w;
        }
        if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
          const int8_4 &inp = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
          inps[0].y = inp.x;
          inps[1].y = inp.y;
          inps[2].y = inp.z;
          inps[3].y = inp.w;
        }
        if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
          const int8_4 &inp = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
          inps[0].z = inp.x;
          inps[1].z = inp.y;
          inps[2].z = inp.z;
          inps[3].z = inp.w;
        }
        if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
          const int8_4 &inp = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
          inps[0].w = inp.x;
          inps[1].w = inp.y;
          inps[2].w = inp.z;
          inps[3].w = inp.w;
        }
        output_temp = 0;
        dp4a((const int32_t *)inps, (const int32_t *)&weight, output_temp);
        out_ptr_NCHW->x = T2int8<float>(output_temp * scale_out);

        output_temp = 0;
        dp4a((const int32_t *)(inps + 1), (const int32_t *)&weight,
             output_temp);
        out_ptr_NCHW->y = T2int8<float>(output_temp * scale_out);

        output_temp = 0;
        dp4a((const int32_t *)(inps + 2), (const int32_t *)&weight,
             output_temp);
        out_ptr_NCHW->z = T2int8<float>(output_temp * scale_out);

        output_temp = 0;
        dp4a((const int32_t *)(inps + 3), (const int32_t *)&weight,
             output_temp);
        out_ptr_NCHW->w = T2int8<float>(output_temp * scale_out);
      }
    } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      ixy = h2rint(ixy);
      int ix_nearest = static_cast<int>(__low2float(ixy));
      int iy_nearest = static_cast<int>(__high2float(ixy));
      float scale_out = scale_i / scale_o;

      // assign nearest neighbor pixel value to output pixel
      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
          const int8_4 &inp =
              inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
          qmulf(inp, *out_ptr_NCHW, scale_out);
        } else {
          *out_ptr_NCHW = 0;
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
      ixy = grid_sampler_unnormalize_h2(
          grid_xy,
          __floats2half2_rn(static_cast<float>(inp_W),
                            static_cast<float>(inp_H)),
          align_corners);

      __half2 ixy_nw = h2floor(ixy);
      const __half2 txy = __hsub2(ixy, ixy_nw);
      float scale_out = scale_i / scale_o;
      int8_4 temp;
      int8_4 coefficients[4];

      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {

#pragma unroll 4
        for (int i = 0; i < 4; ++i) {
          coefficients[i] = cubic_interp1d_int8(
              get_value_bounded_int8(
                  inp_ptr_NC, __hadd2(ixy_nw, __floats2half2_rn(-1.f, i - 1.f)),
                  inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
              get_value_bounded_int8(
                  inp_ptr_NC, __hadd2(ixy_nw, __floats2half2_rn(0.f, i - 1.f)),
                  inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
              get_value_bounded_int8(
                  inp_ptr_NC, __hadd2(ixy_nw, __floats2half2_rn(1.f, i - 1.f)),
                  inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
              get_value_bounded_int8(
                  inp_ptr_NC, __hadd2(ixy_nw, __floats2half2_rn(2.f, i - 1.f)),
                  inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
              __half2float(__low2half(txy)));
        }

        temp = cubic_interp1d_int8(coefficients[0], coefficients[1],
                                   coefficients[2], coefficients[3],
                                   __half2float(__high2half(txy)));
        qmulf(temp, *out_ptr_NCHW, scale_out);
      }
    }
  }
}

template <typename scalar_t>
__global__ void grid_sampler_3d_kernel(
    const int nthreads, const scalar_t *input, const scalar_t *grid,
    scalar_t *output, TensorDesc input_desc, TensorDesc grid_desc,
    TensorDesc output_desc, const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode, const bool align_corners) {
  int C = input_desc.shape[1];
  int inp_D = input_desc.shape[2];
  int inp_H = input_desc.shape[3];
  int inp_W = input_desc.shape[4];
  int out_D = grid_desc.shape[2];
  int out_H = grid_desc.shape[3];
  int out_W = grid_desc.shape[4];
  int inp_sN = input_desc.stride[0];
  int inp_sC = input_desc.stride[1];
  int inp_sD = input_desc.stride[2];
  int inp_sH = input_desc.stride[3];
  int inp_sW = input_desc.stride[4];
  int grid_sN = grid_desc.stride[0];
  int grid_sCoor = grid_desc.stride[1];
  int grid_sD = grid_desc.stride[2];
  int grid_sH = grid_desc.stride[3];
  int grid_sW = grid_desc.stride[4];
  int out_sN = output_desc.stride[0];
  int out_sC = output_desc.stride[1];
  int out_sD = output_desc.stride[2];
  int out_sH = output_desc.stride[3];
  int out_sW = output_desc.stride[4];

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int w = index % out_W;
    const int h = (index / out_W) % out_H;
    const int d = (index / (out_H * out_W)) % out_D;
    const int n = index / (out_D * out_H * out_W);
    const int grid_offset =
        n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y, z coordinates from grid
    scalar_t ix = grid[grid_offset];
    scalar_t iy = grid[grid_offset + grid_sCoor];
    scalar_t iz = grid[grid_offset + 2 * grid_sCoor];

    ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode,
                                           align_corners);
    iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode,
                                           align_corners);
    iz = grid_sampler_compute_source_index(iz, inp_D, padding_mode,
                                           align_corners);

    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      // get corner pixel values from (x, y, z)
      // for 4d, we used north-east-south-west
      // for 5d, we add top-bottom
      int ix_tnw = static_cast<int>(::floor(ix));
      int iy_tnw = static_cast<int>(::floor(iy));
      int iz_tnw = static_cast<int>(::floor(iz));

      int ix_tne = ix_tnw + 1;
      int iy_tne = iy_tnw;
      int iz_tne = iz_tnw;

      int ix_tsw = ix_tnw;
      int iy_tsw = iy_tnw + 1;
      int iz_tsw = iz_tnw;

      int ix_tse = ix_tnw + 1;
      int iy_tse = iy_tnw + 1;
      int iz_tse = iz_tnw;

      int ix_bnw = ix_tnw;
      int iy_bnw = iy_tnw;
      int iz_bnw = iz_tnw + 1;

      int ix_bne = ix_tnw + 1;
      int iy_bne = iy_tnw;
      int iz_bne = iz_tnw + 1;

      int ix_bsw = ix_tnw;
      int iy_bsw = iy_tnw + 1;
      int iz_bsw = iz_tnw + 1;

      int ix_bse = ix_tnw + 1;
      int iy_bse = iy_tnw + 1;
      int iz_bse = iz_tnw + 1;

      // get surfaces to each neighbor:
      scalar_t tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
      scalar_t tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
      scalar_t tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
      scalar_t tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
      scalar_t bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
      scalar_t bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
      scalar_t bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
      scalar_t bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCDHW =
          output + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
        //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) *
        //   tne
        // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) *
        // tse
        // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) *
        // bne
        // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) *
        // bse
        *out_ptr_NCDHW = static_cast<scalar_t>(0);
        if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] *
              tnw;
        }
        if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] *
              tne;
        }
        if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] *
              tsw;
        }
        if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] *
              tse;
        }
        if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] *
              bnw;
        }
        if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] *
              bne;
        }
        if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] *
              bsw;
        }
        if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] *
              bse;
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      int ix_nearest = static_cast<int>(::round(ix));
      int iy_nearest = static_cast<int>(::round(iy));
      int iz_nearest = static_cast<int>(::round(iz));

      // assign nearest neighbor pixel value to output pixel
      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCDHW =
          output + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
        if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H,
                             inp_W)) {
          *out_ptr_NCDHW =
              inp_ptr_NC[iz_nearest * inp_sD + iy_nearest * inp_sH +
                         ix_nearest * inp_sW];
        } else {
          *out_ptr_NCDHW = static_cast<scalar_t>(0);
        }
      }
    }
  }
}

template <>
__global__ void grid_sampler_3d_kernel(
    const int nthreads, const __half *input, const __half *grid, __half *output,
    TensorDesc input_desc, TensorDesc grid_desc, TensorDesc output_desc,
    const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode, const bool align_corners) {
  int C = input_desc.shape[1];
  int inp_D = input_desc.shape[2];
  int inp_H = input_desc.shape[3];
  int inp_W = input_desc.shape[4];
  int out_D = grid_desc.shape[2];
  int out_H = grid_desc.shape[3];
  int out_W = grid_desc.shape[4];
  int inp_sN = input_desc.stride[0];
  int inp_sC = input_desc.stride[1];
  int inp_sD = input_desc.stride[2];
  int inp_sH = input_desc.stride[3];
  int inp_sW = input_desc.stride[4];
  int grid_sN = grid_desc.stride[0];
  int grid_sCoor = grid_desc.stride[1];
  int grid_sD = grid_desc.stride[2];
  int grid_sH = grid_desc.stride[3];
  int grid_sW = grid_desc.stride[4];
  int out_sN = output_desc.stride[0];
  int out_sC = output_desc.stride[1];
  int out_sD = output_desc.stride[2];
  int out_sH = output_desc.stride[3];
  int out_sW = output_desc.stride[4];

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int w = index % out_W;
    const int h = (index / out_W) % out_H;
    const int d = (index / (out_H * out_W)) % out_D;
    const int n = index / (out_D * out_H * out_W);
    const int grid_offset =
        n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y, z coordinates from grid
    __half ix = grid[grid_offset];
    __half iy = grid[grid_offset + grid_sCoor];
    __half iz = grid[grid_offset + 2 * grid_sCoor];

    ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode,
                                           align_corners);
    iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode,
                                           align_corners);
    iz = grid_sampler_compute_source_index(iz, inp_D, padding_mode,
                                           align_corners);

    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      // get corner pixel values from (x, y, z)
      // for 4d, we used north-east-south-west
      // for 5d, we add top-bottom
      int ix_tnw = static_cast<int>(hfloor(ix));
      int iy_tnw = static_cast<int>(hfloor(iy));
      int iz_tnw = static_cast<int>(hfloor(iz));

      int ix_tne = ix_tnw + 1;
      int iy_tne = iy_tnw;
      int iz_tne = iz_tnw;

      int ix_tsw = ix_tnw;
      int iy_tsw = iy_tnw + 1;
      int iz_tsw = iz_tnw;

      int ix_tse = ix_tnw + 1;
      int iy_tse = iy_tnw + 1;
      int iz_tse = iz_tnw;

      int ix_bnw = ix_tnw;
      int iy_bnw = iy_tnw;
      int iz_bnw = iz_tnw + 1;

      int ix_bne = ix_tnw + 1;
      int iy_bne = iy_tnw;
      int iz_bne = iz_tnw + 1;

      int ix_bsw = ix_tnw;
      int iy_bsw = iy_tnw + 1;
      int iz_bsw = iz_tnw + 1;

      int ix_bse = ix_tnw + 1;
      int iy_bse = iy_tnw + 1;
      int iz_bse = iz_tnw + 1;

      // get surfaces to each neighbor:
      __half tnw = __hmul(__hmul(__hsub(static_cast<__half>(ix_bse), ix),
                                 __hsub(static_cast<__half>(iy_bse), iy)),
                          __hsub(static_cast<__half>(iz_bse), iz));
      __half tne = __hmul(__hmul(__hsub(ix, static_cast<__half>(ix_bsw)),
                                 __hsub(static_cast<__half>(iy_bsw), iy)),
                          __hsub(static_cast<__half>(iz_bsw), iz));
      __half tsw = __hmul(__hmul(__hsub(static_cast<__half>(ix_bne), ix),
                                 __hsub(iy, static_cast<__half>(iy_bne))),
                          __hsub(static_cast<__half>(iz_bne), iz));
      __half tse = __hmul(__hmul(__hsub(ix, static_cast<__half>(ix_bnw)),
                                 __hsub(iy, static_cast<__half>(iy_bnw))),
                          __hsub(static_cast<__half>(iz_bnw), iz));
      __half bnw = __hmul(__hmul(__hsub(static_cast<__half>(ix_tse), ix),
                                 __hsub(static_cast<__half>(iy_tse), iy)),
                          __hsub(iz, static_cast<__half>(iz_tse)));
      __half bne = __hmul(__hmul(__hsub(ix, static_cast<__half>(ix_tsw)),
                                 __hsub(static_cast<__half>(iy_tsw), iy)),
                          __hsub(iz, static_cast<__half>(iz_tsw)));
      __half bsw = __hmul(__hmul(__hsub(static_cast<__half>(ix_tne), ix),
                                 __hsub(iy, static_cast<__half>(iy_tne))),
                          __hsub(iz, static_cast<__half>(iz_tne)));
      __half bse = __hmul(__hmul(__hsub(ix, static_cast<__half>(ix_tnw)),
                                 __hsub(iy, static_cast<__half>(iy_tnw))),
                          __hsub(iz, static_cast<__half>(iz_tnw)));

      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCDHW =
          output + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
        //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) *
        //   tne
        // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) *
        // tse
        // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) *
        // bne
        // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) *
        // bse
        *out_ptr_NCDHW = static_cast<__half>(0);
        if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW = __hfma(
              inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW],
              tnw, *out_ptr_NCDHW);
        }
        if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW = __hfma(
              inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW],
              tne, *out_ptr_NCDHW);
        }
        if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW = __hfma(
              inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW],
              tsw, *out_ptr_NCDHW);
        }
        if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW = __hfma(
              inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW],
              tse, *out_ptr_NCDHW);
        }
        if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW = __hfma(
              inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW],
              bnw, *out_ptr_NCDHW);
        }
        if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW = __hfma(
              inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW],
              bne, *out_ptr_NCDHW);
        }
        if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW = __hfma(
              inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW],
              bsw, *out_ptr_NCDHW);
        }
        if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW = __hfma(
              inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW],
              bse, *out_ptr_NCDHW);
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      int ix_nearest = static_cast<int>(hrint(ix));
      int iy_nearest = static_cast<int>(hrint(iy));
      int iz_nearest = static_cast<int>(hrint(iz));

      // assign nearest neighbor pixel value to output pixel
      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCDHW =
          output + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
        if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H,
                             inp_W)) {
          *out_ptr_NCDHW =
              inp_ptr_NC[iz_nearest * inp_sD + iy_nearest * inp_sH +
                         ix_nearest * inp_sW];
        } else {
          *out_ptr_NCDHW = static_cast<__half>(0);
        }
      }
    }
  }
}

template <>
__global__ void grid_sampler_3d_kernel(
    const int nthreads, const __half2 *input, const __half2 *grid,
    __half2 *output, TensorDesc input_desc, TensorDesc grid_desc,
    TensorDesc output_desc, const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode, const bool align_corners) {
  int C = input_desc.shape[1];
  int inp_D = input_desc.shape[2];
  int inp_H = input_desc.shape[3];
  int inp_W = input_desc.shape[4];
  int out_D = grid_desc.shape[2];
  int out_H = grid_desc.shape[3];
  int out_W = grid_desc.shape[4];
  int out_W_h = (out_W + 1) / 2;
  int inp_sN = input_desc.stride[0];
  int inp_sC = input_desc.stride[1];
  int inp_sD = input_desc.stride[2];
  int inp_sH = input_desc.stride[3];
  int inp_sW = input_desc.stride[4];
  int grid_sN = grid_desc.stride[0];
  int grid_sCoor = grid_desc.stride[1];
  int grid_sD = grid_desc.stride[2];
  int grid_sH = grid_desc.stride[3];
  int grid_sW = grid_desc.stride[4];
  int out_sN = output_desc.stride[0];
  int out_sC = output_desc.stride[1];
  int out_sD = output_desc.stride[2];
  int out_sH = output_desc.stride[3];
  int out_sW = output_desc.stride[4];

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int w = index % out_W_h;
    const int h = (index / out_W_h) % out_H;
    const int d = (index / (out_H * out_W_h)) % out_D;
    const int n = index / (out_D * out_H * out_W_h);
    const int grid_offset = n * grid_sN + d * grid_sD + h * grid_sH;
    const int grid_w_offset = w * grid_sW;

    const bool last_one = grid_w_offset * 2 == out_W - 1;

    __half2 inp_D_h2 = __float2half2_rn(static_cast<float>(inp_D));
    __half2 inp_H_h2 = __float2half2_rn(static_cast<float>(inp_H));
    __half2 inp_W_h2 = __float2half2_rn(static_cast<float>(inp_W));
    __half2 within_b, out_h2;
    __half inp_l, inp_h;

    // get the corresponding input x, y, z coordinates from grid
    __half2 ix, iy, iz;
    if (last_one) {
      ix = __halves2half2(((__half *)grid + grid_offset)[out_W - 1],
                          static_cast<__half>(0.));
      iy =
          __halves2half2(((__half *)grid + grid_offset + grid_sCoor)[out_W - 1],
                         static_cast<__half>(0.));
      iz = __halves2half2(
          ((__half *)grid + grid_offset + 2 * grid_sCoor)[out_W - 1],
          static_cast<__half>(0.));
    } else {
      ix =
          __halves2half2(((__half *)grid + grid_offset)[grid_w_offset * 2],
                         ((__half *)grid + grid_offset)[grid_w_offset * 2 + 1]);
      iy = __halves2half2(
          ((__half *)grid + grid_offset + grid_sCoor)[grid_w_offset * 2],
          ((__half *)grid + grid_offset + grid_sCoor)[grid_w_offset * 2 + 1]);
      iz = __halves2half2(
          ((__half *)grid + grid_offset + 2 * grid_sCoor)[grid_w_offset * 2],
          ((__half *)grid + grid_offset +
           2 * grid_sCoor)[grid_w_offset * 2 + 1]);
    }

    ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode,
                                           align_corners);
    iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode,
                                           align_corners);
    iz = grid_sampler_compute_source_index(iz, inp_D, padding_mode,
                                           align_corners);

    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      // get corner pixel values from (x, y, z)
      // for 4d, we used north-east-south-west
      // for 5d, we add top-bottom
      __half2 ix_tnw = h2floor(ix);
      __half2 iy_tnw = h2floor(iy);
      __half2 iz_tnw = h2floor(iz);

      __half2 ix_tne = __hadd2(ix_tnw, __float2half2_rn(1.));
      __half2 iy_tne = iy_tnw;
      __half2 iz_tne = iz_tnw;

      __half2 ix_tsw = ix_tnw;
      __half2 iy_tsw = __hadd2(iy_tnw, __float2half2_rn(1.));
      __half2 iz_tsw = iz_tnw;

      __half2 ix_tse = __hadd2(ix_tnw, __float2half2_rn(1.));
      __half2 iy_tse = __hadd2(iy_tnw, __float2half2_rn(1.));
      __half2 iz_tse = iz_tnw;

      __half2 ix_bnw = ix_tnw;
      __half2 iy_bnw = iy_tnw;
      __half2 iz_bnw = __hadd2(iz_tnw, __float2half2_rn(1.));

      __half2 ix_bne = __hadd2(ix_tnw, __float2half2_rn(1.));
      __half2 iy_bne = iy_tnw;
      __half2 iz_bne = __hadd2(iz_tnw, __float2half2_rn(1.));

      __half2 ix_bsw = ix_tnw;
      __half2 iy_bsw = __hadd2(iy_tnw, __float2half2_rn(1.));
      __half2 iz_bsw = __hadd2(iz_tnw, __float2half2_rn(1.));

      __half2 ix_bse = __hadd2(ix_tnw, __float2half2_rn(1.));
      __half2 iy_bse = __hadd2(iy_tnw, __float2half2_rn(1.));
      __half2 iz_bse = __hadd2(iz_tnw, __float2half2_rn(1.));

      // get surfaces to each neighbor:
      __half2 tnw = __hmul2(__hmul2(__hsub2(ix_bse, ix), __hsub2(iy_bse, iy)),
                            __hsub2(iz_bse, iz));
      __half2 tne = __hmul2(__hmul2(__hsub2(ix, ix_bsw), __hsub2(iy_bsw, iy)),
                            __hsub2(iz_bsw, iz));
      __half2 tsw = __hmul2(__hmul2(__hsub2(ix_bne, ix), __hsub2(iy, iy_bne)),
                            __hsub2(iz_bne, iz));
      __half2 tse = __hmul2(__hmul2(__hsub2(ix, ix_bnw), __hsub2(iy, iy_bnw)),
                            __hsub2(iz_bnw, iz));
      __half2 bnw = __hmul2(__hmul2(__hsub2(ix_tse, ix), __hsub2(iy_tse, iy)),
                            __hsub2(iz, iz_tse));
      __half2 bne = __hmul2(__hmul2(__hsub2(ix, ix_tsw), __hsub2(iy_tsw, iy)),
                            __hsub2(iz, iz_tsw));
      __half2 bsw = __hmul2(__hmul2(__hsub2(ix_tne, ix), __hsub2(iy, iy_tne)),
                            __hsub2(iz, iz_tne));
      __half2 bse = __hmul2(__hmul2(__hsub2(ix, ix_tnw), __hsub2(iy, iy_tnw)),
                            __hsub2(iz, iz_tnw));

      __half *inp_ptr_NC = (__half *)input + n * inp_sN;
      __half *out_ptr_NCDHW = (__half *)output + n * out_sN + d * out_sD +
                              h * out_sH + w * out_sW * 2;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
        //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) *
        //   tne
        // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) *
        // tse
        // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) *
        // bne
        // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) *
        // bse
        out_h2 = __float2half2_rn(0.);

        within_b = within_bounds_3d_half2(iz_tnw, iy_tnw, ix_tnw, inp_D_h2,
                                          inp_H_h2, inp_W_h2);
        inp_l = inp_ptr_NC[(static_cast<int>(__low2float(iz_tnw)) * inp_sD +
                            static_cast<int>(__low2float(iy_tnw)) * inp_sH +
                            static_cast<int>(__low2float(ix_tnw)) * inp_sW) *
                           static_cast<int>(__low2float(within_b))];
        inp_h = inp_ptr_NC[(static_cast<int>(__high2float(iz_tnw)) * inp_sD +
                            static_cast<int>(__high2float(iy_tnw)) * inp_sH +
                            static_cast<int>(__high2float(ix_tnw)) * inp_sW) *
                           static_cast<int>(__high2float(within_b))];
        out_h2 = __hfma2(__hmul2(__halves2half2(inp_l, inp_h), tnw), within_b,
                         out_h2);

        within_b = within_bounds_3d_half2(iz_tne, iy_tne, ix_tne, inp_D_h2,
                                          inp_H_h2, inp_W_h2);
        inp_l = inp_ptr_NC[(static_cast<int>(__low2float(iz_tne)) * inp_sD +
                            static_cast<int>(__low2float(iy_tne)) * inp_sH +
                            static_cast<int>(__low2float(ix_tne)) * inp_sW) *
                           static_cast<int>(__low2float(within_b))];
        inp_h = inp_ptr_NC[(static_cast<int>(__high2float(iz_tne)) * inp_sD +
                            static_cast<int>(__high2float(iy_tne)) * inp_sH +
                            static_cast<int>(__high2float(ix_tne)) * inp_sW) *
                           static_cast<int>(__high2float(within_b))];
        out_h2 = __hfma2(__hmul2(__halves2half2(inp_l, inp_h), tne), within_b,
                         out_h2);

        within_b = within_bounds_3d_half2(iz_tsw, iy_tsw, ix_tsw, inp_D_h2,
                                          inp_H_h2, inp_W_h2);
        inp_l = inp_ptr_NC[(static_cast<int>(__low2float(iz_tsw)) * inp_sD +
                            static_cast<int>(__low2float(iy_tsw)) * inp_sH +
                            static_cast<int>(__low2float(ix_tsw)) * inp_sW) *
                           static_cast<int>(__low2float(within_b))];
        inp_h = inp_ptr_NC[(static_cast<int>(__high2float(iz_tsw)) * inp_sD +
                            static_cast<int>(__high2float(iy_tsw)) * inp_sH +
                            static_cast<int>(__high2float(ix_tsw)) * inp_sW) *
                           static_cast<int>(__high2float(within_b))];
        out_h2 = __hfma2(__hmul2(__halves2half2(inp_l, inp_h), tsw), within_b,
                         out_h2);

        within_b = within_bounds_3d_half2(iz_tse, iy_tse, ix_tse, inp_D_h2,
                                          inp_H_h2, inp_W_h2);
        inp_l = inp_ptr_NC[(static_cast<int>(__low2float(iz_tse)) * inp_sD +
                            static_cast<int>(__low2float(iy_tse)) * inp_sH +
                            static_cast<int>(__low2float(ix_tse)) * inp_sW) *
                           static_cast<int>(__low2float(within_b))];
        inp_h = inp_ptr_NC[(static_cast<int>(__high2float(iz_tse)) * inp_sD +
                            static_cast<int>(__high2float(iy_tse)) * inp_sH +
                            static_cast<int>(__high2float(ix_tse)) * inp_sW) *
                           static_cast<int>(__high2float(within_b))];
        out_h2 = __hfma2(__hmul2(__halves2half2(inp_l, inp_h), tse), within_b,
                         out_h2);

        within_b = within_bounds_3d_half2(iz_bnw, iy_bnw, ix_bnw, inp_D_h2,
                                          inp_H_h2, inp_W_h2);
        inp_l = inp_ptr_NC[(static_cast<int>(__low2float(iz_bnw)) * inp_sD +
                            static_cast<int>(__low2float(iy_bnw)) * inp_sH +
                            static_cast<int>(__low2float(ix_bnw)) * inp_sW) *
                           static_cast<int>(__low2float(within_b))];
        inp_h = inp_ptr_NC[(static_cast<int>(__high2float(iz_bnw)) * inp_sD +
                            static_cast<int>(__high2float(iy_bnw)) * inp_sH +
                            static_cast<int>(__high2float(ix_bnw)) * inp_sW) *
                           static_cast<int>(__high2float(within_b))];
        out_h2 = __hfma2(__hmul2(__halves2half2(inp_l, inp_h), bnw), within_b,
                         out_h2);

        within_b = within_bounds_3d_half2(iz_bne, iy_bne, ix_bne, inp_D_h2,
                                          inp_H_h2, inp_W_h2);
        inp_l = inp_ptr_NC[(static_cast<int>(__low2float(iz_bne)) * inp_sD +
                            static_cast<int>(__low2float(iy_bne)) * inp_sH +
                            static_cast<int>(__low2float(ix_bne)) * inp_sW) *
                           static_cast<int>(__low2float(within_b))];
        inp_h = inp_ptr_NC[(static_cast<int>(__high2float(iz_bne)) * inp_sD +
                            static_cast<int>(__high2float(iy_bne)) * inp_sH +
                            static_cast<int>(__high2float(ix_bne)) * inp_sW) *
                           static_cast<int>(__high2float(within_b))];
        out_h2 = __hfma2(__hmul2(__halves2half2(inp_l, inp_h), bne), within_b,
                         out_h2);

        within_b = within_bounds_3d_half2(iz_bsw, iy_bsw, ix_bsw, inp_D_h2,
                                          inp_H_h2, inp_W_h2);
        inp_l = inp_ptr_NC[(static_cast<int>(__low2float(iz_bsw)) * inp_sD +
                            static_cast<int>(__low2float(iy_bsw)) * inp_sH +
                            static_cast<int>(__low2float(ix_bsw)) * inp_sW) *
                           static_cast<int>(__low2float(within_b))];
        inp_h = inp_ptr_NC[(static_cast<int>(__high2float(iz_bsw)) * inp_sD +
                            static_cast<int>(__high2float(iy_bsw)) * inp_sH +
                            static_cast<int>(__high2float(ix_bsw)) * inp_sW) *
                           static_cast<int>(__high2float(within_b))];
        out_h2 = __hfma2(__hmul2(__halves2half2(inp_l, inp_h), bsw), within_b,
                         out_h2);

        within_b = within_bounds_3d_half2(iz_bse, iy_bse, ix_bse, inp_D_h2,
                                          inp_H_h2, inp_W_h2);
        inp_l = inp_ptr_NC[(static_cast<int>(__low2float(iz_bse)) * inp_sD +
                            static_cast<int>(__low2float(iy_bse)) * inp_sH +
                            static_cast<int>(__low2float(ix_bse)) * inp_sW) *
                           static_cast<int>(__low2float(within_b))];
        inp_h = inp_ptr_NC[(static_cast<int>(__high2float(iz_bse)) * inp_sD +
                            static_cast<int>(__high2float(iy_bse)) * inp_sH +
                            static_cast<int>(__high2float(ix_bse)) * inp_sW) *
                           static_cast<int>(__high2float(within_b))];
        out_h2 = __hfma2(__hmul2(__halves2half2(inp_l, inp_h), bse), within_b,
                         out_h2);

        *out_ptr_NCDHW = __low2half(out_h2);
        if (!last_one) {
          *(out_ptr_NCDHW + 1) = __high2half(out_h2);
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      __half2 ix_nearest = h2rint(ix);
      __half2 iy_nearest = h2rint(iy);
      __half2 iz_nearest = h2rint(iz);

      // assign nearest neighbor pixel value to output pixel
      __half *inp_ptr_NC = (__half *)input + n * inp_sN;
      __half *out_ptr_NCDHW = (__half *)output + n * out_sN + d * out_sD +
                              h * out_sH + w * out_sW * 2;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
        out_h2 = __float2half2_rn(0.);

        within_b = within_bounds_3d_half2(iz_nearest, iy_nearest, ix_nearest,
                                          inp_D_h2, inp_H_h2, inp_W_h2);
        inp_l =
            inp_ptr_NC[(static_cast<int>(__low2float(iz_nearest)) * inp_sD +
                        static_cast<int>(__low2float(iy_nearest)) * inp_sH +
                        static_cast<int>(__low2float(ix_nearest)) * inp_sW) *
                       static_cast<int>(__low2float(within_b))];
        inp_h =
            inp_ptr_NC[(static_cast<int>(__high2float(iz_nearest)) * inp_sD +
                        static_cast<int>(__high2float(iy_nearest)) * inp_sH +
                        static_cast<int>(__high2float(ix_nearest)) * inp_sW) *
                       static_cast<int>(__high2float(within_b))];
        out_h2 = __hmul2(__halves2half2(inp_l, inp_h), within_b);

        *out_ptr_NCDHW = __low2half(out_h2);
        if (!last_one) {
          *(out_ptr_NCDHW + 1) = __high2half(out_h2);
        }
      }
    }
  }
}

void create_desc(const int *dims, int nb_dims, TensorDesc &desc) {
  memcpy(&desc.shape[0], dims, sizeof(int) * nb_dims);
  desc.stride[nb_dims - 1] = 1;
  for (int i = nb_dims - 2; i >= 0; --i) {
    desc.stride[i] = desc.stride[i + 1] * desc.shape[i + 1];
  }
}

template <typename T>
void grid_sample(T *output, const T *input, const T *grid, int *output_dims,
                 int *input_dims, int *grid_dims, int nb_dims,
                 GridSamplerInterpolation interp, GridSamplerPadding padding,
                 bool align_corners, cudaStream_t stream) {
  TensorDesc input_desc;
  create_desc(input_dims, nb_dims, input_desc);

  TensorDesc output_desc;
  create_desc(output_dims, nb_dims, output_desc);

  TensorDesc grid_desc;
  create_desc(grid_dims, nb_dims, grid_desc);

  int count = 1;
  for (int i = 0; i < nb_dims; ++i) {
    if (i == 1) {
      continue;
    }
    count *= output_desc.shape[i];
  }

  if (nb_dims == 4) {
    grid_sampler_2d_kernel<T>
        <<<GET_BLOCKS(count), THREADS_PER_BLOCK, 0, stream>>>(
            count, input, grid, output, input_desc, grid_desc, output_desc,
            interp, padding, align_corners);
  } else if (nb_dims == 5) {
    grid_sampler_3d_kernel<T>
        <<<GET_BLOCKS(count), THREADS_PER_BLOCK, 0, stream>>>(
            count, input, grid, output, input_desc, grid_desc, output_desc,
            interp, padding, align_corners);
  } else {
    printf("input and grid dims should be 4 or 5\n");
  }
}

template <>
void grid_sample(__half2 *output, const __half2 *input, const __half2 *grid,
                 int *output_dims, int *input_dims, int *grid_dims, int nb_dims,
                 GridSamplerInterpolation interp, GridSamplerPadding padding,
                 bool align_corners, cudaStream_t stream) {
  TensorDesc input_desc;
  create_desc(input_dims, nb_dims, input_desc);

  TensorDesc output_desc;
  create_desc(output_dims, nb_dims, output_desc);

  TensorDesc grid_desc;
  create_desc(grid_dims, nb_dims, grid_desc);

  int count = 1;
  for (int i = 0; i < nb_dims; ++i) {
    if (i == 1) {
      continue;
    } else if (i == nb_dims - 1 and nb_dims == 5) {
      count *= (output_desc.shape[i] + 1) / 2;
    } else {
      count *= output_desc.shape[i];
    }
  }

  if (nb_dims == 4) {
    grid_sampler_2d_kernel<__half2>
        <<<GET_BLOCKS(count), THREADS_PER_BLOCK, 0, stream>>>(
            count, input, grid, output, input_desc, grid_desc, output_desc,
            interp, padding, align_corners);
  } else if (nb_dims == 5) {
    grid_sampler_3d_kernel<__half2>
        <<<GET_BLOCKS(count), THREADS_PER_BLOCK, 0, stream>>>(
            count, input, grid, output, input_desc, grid_desc, output_desc,
            interp, padding, align_corners);
  } else {
    printf("input and grid dims should be 4 or 5\n");
  }
}

void grid_sample_int8(int8_4 *output, const float &scale_o, const int8_4 *input,
                      const float &scale_i, const int8_4 *grid,
                      const float &scale_g, int *output_dims, int *input_dims,
                      int *grid_dims, int nb_dims,
                      GridSamplerInterpolation interp,
                      GridSamplerPadding padding, bool align_corners,
                      cudaStream_t stream) {
  TensorDesc input_desc;
  create_desc(input_dims, nb_dims, input_desc);

  TensorDesc output_desc;
  create_desc(output_dims, nb_dims, output_desc);

  TensorDesc grid_desc;
  create_desc(grid_dims, nb_dims, grid_desc);

  int count = 1;
  for (int i = 0; i < nb_dims; ++i) {
    if (i == 1) {
      continue;
    } else {
      count *= output_desc.shape[i];
    }
  }

  if (nb_dims == 4) {
    grid_sampler_2d_kernel_int8<<<GET_BLOCKS(count), THREADS_PER_BLOCK, 0,
                                  stream>>>(
        count, input, scale_i, grid, scale_g, output, scale_o, input_desc,
        grid_desc, output_desc, interp, padding, align_corners);
  } else {
    printf("input and grid dims should be 4\n");
  }
}

template void grid_sample<float>(float *output, const float *input,
                                 const float *grid, int *output_dims,
                                 int *input_dims, int *grid_dims, int nb_dims,
                                 GridSamplerInterpolation interp,
                                 GridSamplerPadding padding, bool align_corners,
                                 cudaStream_t stream);

template void grid_sample<__half>(__half *output, const __half *input,
                                  const __half *grid, int *output_dims,
                                  int *input_dims, int *grid_dims, int nb_dims,
                                  GridSamplerInterpolation interp,
                                  GridSamplerPadding padding,
                                  bool align_corners, cudaStream_t stream);

template void grid_sample<__half2>(__half2 *output, const __half2 *input,
                                   const __half2 *grid, int *output_dims,
                                   int *input_dims, int *grid_dims, int nb_dims,
                                   GridSamplerInterpolation interp,
                                   GridSamplerPadding padding,
                                   bool align_corners, cudaStream_t stream);
