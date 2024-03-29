//
// Created by Derry Lin on 2023/5/11.
//

#include "helper.h"
#include "inverseKernel.h"
#include <cublas_v2.h>

template <typename T>
void inverse(T *output, T *input, void *workspace, T **inpPtrArr, T **outPtrArr,
             const int &matrix_n, const int &matrix_dim,
             cublasHandle_t cublas_handle, cudaStream_t stream) {
  T **inputPtr = (T **)workspace;
  workspace = (T **)workspace + matrix_n;
  T **outputPtr = (T **)workspace;
  workspace = (T **)workspace + matrix_n;

  int pow_dim = matrix_dim * matrix_dim;
  for (int i = 0; i < matrix_n; i++) {
    inpPtrArr[i] = input + i * pow_dim;
    outPtrArr[i] = output + i * pow_dim;
  }
  cudaMemcpyAsync(inputPtr, inpPtrArr, sizeof(T *) * matrix_n,
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(outputPtr, outPtrArr, sizeof(T *) * matrix_n,
                  cudaMemcpyHostToDevice, stream);

  int *infoArray = (int *)workspace;
  workspace = (int *)workspace + matrix_n;
  int *pivotArray = (int *)workspace;

  cublasStatus_t status;
  status = cublasSgetrfBatched(cublas_handle, matrix_dim, inputPtr, matrix_dim,
                               pivotArray, infoArray, matrix_n);
  ASSERT(status == CUBLAS_STATUS_SUCCESS);
  status = cublasSgetriBatched(cublas_handle, matrix_dim, (const T **)inputPtr,
                               matrix_dim, pivotArray, outputPtr, matrix_dim,
                               infoArray, matrix_n);
  ASSERT(status == CUBLAS_STATUS_SUCCESS);
}

template void inverse(float *output, float *input, void *workspace,
                      float **inpPtrArr, float **outPtrArr, const int &matrix_n,
                      const int &matrix_dim, cublasHandle_t cublas_handle,
                      cudaStream_t stream);
