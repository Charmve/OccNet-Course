//
// Created by Derry Lin on 2023/6/16.
//

#include "multiHeadAttnPlugin.h"
#include "checkMacrosPlugin.h"
#include "cuda_helper.h"
#include "multiHeadAttnKernel.h"
#include "multiHeadFlashAttnKernel.h"
#include "serialize.h"
#include <cuda_fp16.h>
#include <stdexcept>

using trt_plugin::QKVPlugin;
using trt_plugin::QKVPluginCreator;
using trt_plugin::QKVPluginCreator2;
using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace {
constexpr char const *QKV_PLUGIN_VERSION{"1"};
constexpr char const *QKV_PLUGIN_NAME{"QKVTRT"};
constexpr char const *QKV_PLUGIN_NAME2{"QKVTRT2"};
} // namespace

PluginFieldCollection QKVPluginCreator::mFC{};
std::vector<PluginField> QKVPluginCreator::mPluginAttributes;

PluginFieldCollection QKVPluginCreator2::mFC{};
std::vector<PluginField> QKVPluginCreator2::mPluginAttributes;

QKVPlugin::QKVPlugin(bool use_h2) : use_h2(use_h2) {}

QKVPlugin::QKVPlugin(cublasGemmAlgo_t algo_1, cublasGemmAlgo_t algo_2,
                     bool use_h2)
    : mAlgo_1(algo_1), mAlgo_2(algo_2), use_h2(use_h2) {}

QKVPlugin::QKVPlugin(const void *serialData, size_t serialLength, bool use_h2)
    : use_h2(use_h2) {
  deserialize_value(&serialData, &serialLength, &mAlgo_1);
  deserialize_value(&serialData, &serialLength, &mAlgo_2);
  deserialize_value(&serialData, &serialLength, &mCfgAlgo);
}

QKVPlugin::~QKVPlugin() { terminate(); }

int32_t QKVPlugin::getNbOutputs() const noexcept { return 1; }

DimsExprs QKVPlugin::getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) noexcept {
  return inputs[0];
}

int32_t QKVPlugin::initialize() noexcept { return 0; }

void QKVPlugin::terminate() noexcept {}

size_t QKVPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                   int32_t nbInputs,
                                   const nvinfer1::PluginTensorDesc *outputs,
                                   int32_t nbOutputs) const noexcept {
  if (mFMHA) {
    return 0;
  }
  const int batch = inputs[0].dims.d[0];
  const int q_len = inputs[0].dims.d[1];
  const int kv_len = inputs[1].dims.d[1];
  auto data_type = inputs[0].type;
  int size;
  switch (data_type) {
  case DataType::kFLOAT:
    size = 4;
    break;
  case DataType::kHALF:
    size = 2;
    break;
  case DataType::kINT8:
    size = 6;
    break;
  default:
    size = 0;
    break;
  }
  return batch * q_len * kv_len * size +
         batch * q_len * sizeof(float) * (size == 6);
}

int32_t QKVPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                           const nvinfer1::PluginTensorDesc *outputDesc,
                           const void *const *inputs, void *const *outputs,
                           void *workspace, cudaStream_t stream) noexcept {
  const int batch = inputDesc[0].dims.d[0];
  const int q_len = inputDesc[0].dims.d[1];
  const int embed_dim = inputDesc[0].dims.d[2];
  const int kv_len = inputDesc[1].dims.d[1];
  const float scale_q = inputDesc[0].scale;
  const float scale_k = inputDesc[1].scale;
  const float scale_v = inputDesc[2].scale;
  const float scale_o = outputDesc[0].scale;
  auto data_type = inputDesc[0].type;
  switch (data_type) {
  case nvinfer1::DataType::kFLOAT:
    if (mFMHA) {
      qkv_flash((const float *)inputs[0], (const float *)inputs[1],
                (const float *)inputs[2], (float *)outputs[0], batch, q_len,
                kv_len, embed_dim, stream);
    } else {
      qkv((float *)workspace, (const float *)inputs[0],
          (const float *)inputs[1], (const float *)inputs[2],
          (float *)outputs[0], batch, q_len, kv_len, embed_dim, m_cublas_handle,
          stream, mAlgo_1, mAlgo_2);
    }
    break;
  case nvinfer1::DataType::kHALF:
    if (mFMHA) {
      if (use_h2) {
        qkv_flash((const __half2 *)inputs[0], (const __half2 *)inputs[1],
                  (const __half2 *)inputs[2], (__half2 *)outputs[0], batch,
                  q_len, kv_len, embed_dim, stream);
      } else {
        qkv_flash((const __half *)inputs[0], (const __half *)inputs[1],
                  (const __half *)inputs[2], (__half *)outputs[0], batch, q_len,
                  kv_len, embed_dim, stream);
      }
    } else {
      if (use_h2) {
        qkv((__half2 *)workspace, (const __half2 *)inputs[0],
            (const __half2 *)inputs[1], (const __half2 *)inputs[2],
            (__half2 *)outputs[0], batch, q_len, kv_len, embed_dim,
            m_cublas_handle, stream, mAlgo_1, mAlgo_2);
      } else {
        qkv((__half *)workspace, (const __half *)inputs[0],
            (const __half *)inputs[1], (const __half *)inputs[2],
            (__half *)outputs[0], batch, q_len, kv_len, embed_dim,
            m_cublas_handle, stream, mAlgo_1, mAlgo_2);
      }
    }
    break;
  case nvinfer1::DataType::kINT8:
    if (mFMHA) {
      qkv_flash_int8((const int8_4 *)inputs[0], scale_q,
                     (const int8_4 *)inputs[1], scale_k,
                     (const int8_4 *)inputs[2], scale_v, (int8_4 *)outputs[0],
                     scale_o, batch, q_len, kv_len, embed_dim, stream);
    } else {
      qkv_int8((int8_4 *)workspace, (const int8_4 *)inputs[0], scale_q,
               (const int8_4 *)inputs[1], scale_k, (const int8_4 *)inputs[2],
               scale_v, (int8_4 *)outputs[0], scale_o, batch, q_len, kv_len,
               embed_dim, m_cublas_handle, stream, mAlgo_1, mAlgo_2);
    }
    break;
  default:
    return 1;
  }
  return 0;
}

size_t QKVPlugin::getSerializationSize() const noexcept {
  return serialized_size(mAlgo_1) + serialized_size(mAlgo_2) +
         serialized_size(mCfgAlgo);
}

void QKVPlugin::serialize(void *buffer) const noexcept {
  serialize_value(&buffer, mAlgo_1);
  serialize_value(&buffer, mAlgo_2);
  serialize_value(&buffer, mCfgAlgo);
}

bool QKVPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc *inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept {
  if (pos == 0) {
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
             inOut[pos].type == nvinfer1::DataType::kHALF ||
             (inOut[pos].type == nvinfer1::DataType::kINT8 && use_int8)) &&
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == inOut[0].format;
  }
}

char const *QKVPlugin::getPluginType() const noexcept {
  return use_h2 ? QKV_PLUGIN_NAME2 : QKV_PLUGIN_NAME;
}

char const *QKVPlugin::getPluginVersion() const noexcept {
  return QKV_PLUGIN_VERSION;
}

void QKVPlugin::destroy() noexcept { delete this; }

IPluginV2DynamicExt *QKVPlugin::clone() const noexcept {
  try {
    auto *plugin = new QKVPlugin(mAlgo_1, mAlgo_2, use_h2);
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    plugin->initialize();
    plugin->setCfgAlgo(mCfgAlgo);
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

void QKVPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
  mPluginNamespace = pluginNamespace;
}

char const *QKVPlugin::getPluginNamespace() const noexcept {
  return mPluginNamespace.c_str();
}

DataType QKVPlugin::getOutputDataType(int32_t index,
                                      const nvinfer1::DataType *inputTypes,
                                      int32_t nbInputs) const noexcept {
  PLUGIN_ASSERT(inputTypes && nbInputs > 0 && index == 0)
  return inputTypes[0];
}

void QKVPlugin::attachToContext(cudnnContext *cudnn, cublasContext *cublas,
                                nvinfer1::IGpuAllocator *allocator) noexcept {
  m_cublas_handle = cublas;
}

void QKVPlugin::detachFromContext() noexcept {}

void QKVPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                                int32_t nbInputs,
                                const nvinfer1::DynamicPluginTensorDesc *out,
                                int32_t nbOutputs) noexcept {
  PLUGIN_ASSERT(nbInputs == 3);
  PLUGIN_ASSERT(nbOutputs == 1);

  int q_len = in[0].desc.dims.d[1];
  int embed_dim = in[0].desc.dims.d[2];
  int kv_len = in[1].desc.dims.d[1];

  checkFMHA(q_len, kv_len, embed_dim);
  if (mFMHA)
    return;

  auto data_type = in[0].desc.type;

  if (!mCfgAlgo) {
    switch (data_type) {
    case DataType::kFLOAT:
      chooseAlgo<float>(in, out);
      break;
    case DataType::kHALF:
      chooseAlgo<__half>(in, out);
      break;
    case DataType::kINT8:
      chooseAlgoINT8(in, out);
      break;
    default:
      exit(1);
    }
    mCfgAlgo = true;
  }

  if (q_len % 2 != 0 || embed_dim % 2 != 0 || kv_len % 2 != 0) {
    use_h2 = false;
  }
  if (q_len % 4 != 0 || embed_dim % 4 != 0 || kv_len % 4 != 0) {
    use_int8 = false;
  }
}

template <class T>
void QKVPlugin::chooseAlgo(
    nvinfer1::DynamicPluginTensorDesc const *in,
    nvinfer1::DynamicPluginTensorDesc const *out) noexcept {
  float minTime = INFINITY, time;
  int batch = in[0].desc.dims.d[0];
  int q_len = in[0].desc.dims.d[1];
  int embed_dim = in[0].desc.dims.d[2];
  int kv_len = in[1].desc.dims.d[1];
  cublasStatus_t status;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cublasHandle_t cublasHandle;
  cublasCreate_v2(&cublasHandle);
  T *A, *B, *C;
  T alpha = 1.f, beta = 0.f;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaMallocAsync(&A, batch * q_len * embed_dim * sizeof(T), stream);
  cudaMallocAsync(&B, batch * kv_len * embed_dim * sizeof(T), stream);
  cudaMallocAsync(&C, batch * q_len * kv_len * sizeof(T), stream);

  printf("[QKVPlugin::configurePlugin]: *******************Testing Algo "
         "1*******************\n");
  for (cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT; algo <= CUBLAS_GEMM_ALGO23;
       algo = (cublasGemmAlgo_t)(algo + 1)) {
    cudaEventRecord(start, stream);
    status = cublasGemmStridedBatchedWrap<T>(
        cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, kv_len, q_len, embed_dim,
        &alpha, B, embed_dim, kv_len * embed_dim, A, embed_dim,
        q_len * embed_dim, &beta, C, kv_len, q_len * kv_len, batch, algo);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

    if (status == CUBLAS_STATUS_SUCCESS) {
      printf("[QKVPlugin::configurePlugin]: CUBLAS SUCCESS, TIME: %f\n", time);
      if (time < minTime) {
        minTime = time;
        mAlgo_1 = algo;
      }
    } else {
      printf("[QKVPlugin::configurePlugin]: CUBLAS FAIL, Code: %d\n", status);
    }
  }
  for (cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
       algo <= CUBLAS_GEMM_ALGO15_TENSOR_OP;
       algo = (cublasGemmAlgo_t)(algo + 1)) {
    cudaEventRecord(start, stream);
    status = cublasGemmStridedBatchedWrap<T>(
        cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, kv_len, q_len, embed_dim,
        &alpha, B, embed_dim, kv_len * embed_dim, A, embed_dim,
        q_len * embed_dim, &beta, C, kv_len, q_len * kv_len, batch, algo);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

    if (status == CUBLAS_STATUS_SUCCESS) {
      printf("[QKVPlugin::configurePlugin]: CUBLAS SUCCESS, TIME: %f\n", time);
      if (time < minTime) {
        minTime = time;
        mAlgo_1 = algo;
      }
    } else {
      printf("[QKVPlugin::configurePlugin]: CUBLAS FAIL, Code: %d\n", status);
    }
  }
  printf("[QKVPlugin::configurePlugin]: *******************Fastest Algo_1 %d, "
         "Time: %f*******************\n",
         mAlgo_1, minTime);

  cudaFreeAsync(A, stream);
  cudaFreeAsync(B, stream);
  cudaFreeAsync(C, stream);
  cudaMallocAsync(&A, batch * kv_len * embed_dim * sizeof(T), stream);
  cudaMallocAsync(&B, batch * kv_len * q_len * sizeof(T), stream);
  cudaMallocAsync(&C, batch * q_len * embed_dim * sizeof(T), stream);

  minTime = INFINITY;
  printf("[QKVPlugin::configurePlugin]: *******************Testing Algo "
         "2*******************\n");
  for (cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT; algo <= CUBLAS_GEMM_ALGO23;
       algo = (cublasGemmAlgo_t)(algo + 1)) {
    cudaEventRecord(start, stream);
    status = cublasGemmStridedBatchedWrap<T>(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, embed_dim, q_len, kv_len,
        &alpha, A, embed_dim, kv_len * embed_dim, B, kv_len, q_len * kv_len,
        &beta, C, embed_dim, q_len * embed_dim, batch, algo);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

    if (status == CUBLAS_STATUS_SUCCESS) {
      printf("[QKVPlugin::configurePlugin]: CUBLAS SUCCESS, TIME: %f\n", time);
      if (time < minTime) {
        minTime = time;
        mAlgo_2 = algo;
      }
    } else {
      printf("[QKVPlugin::configurePlugin]: CUBLAS FAIL, Code: %d\n", status);
    }
  }
  for (cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
       algo <= CUBLAS_GEMM_ALGO15_TENSOR_OP;
       algo = (cublasGemmAlgo_t)(algo + 1)) {
    cudaEventRecord(start, stream);
    status = cublasGemmStridedBatchedWrap<T>(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, embed_dim, q_len, kv_len,
        &alpha, A, embed_dim, kv_len * embed_dim, B, kv_len, q_len * kv_len,
        &beta, C, embed_dim, q_len * embed_dim, batch, algo);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

    if (status == CUBLAS_STATUS_SUCCESS) {
      printf("[QKVPlugin::configurePlugin]: CUBLAS SUCCESS, TIME: %f\n", time);
      if (time < minTime) {
        minTime = time;
        mAlgo_2 = algo;
      }
    } else {
      printf("[QKVPlugin::configurePlugin]: CUBLAS FAIL, Code: %d\n", status);
    }
  }
  printf("[QKVPlugin::configurePlugin]: *******************Fastest Algo_2 %d, "
         "Time: %f*******************\n",
         mAlgo_1, minTime);
  cudaFreeAsync(A, stream);
  cudaFreeAsync(B, stream);
  cudaFreeAsync(C, stream);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cublasDestroy_v2(cublasHandle);
  cudaStreamDestroy(stream);
}

void QKVPlugin::chooseAlgoINT8(
    nvinfer1::DynamicPluginTensorDesc const *in,
    nvinfer1::DynamicPluginTensorDesc const *out) noexcept {
  float minTime = INFINITY, time;
  int batch = in[0].desc.dims.d[0];
  int q_len = in[0].desc.dims.d[1];
  int embed_dim = in[0].desc.dims.d[2];
  int kv_len = in[1].desc.dims.d[1];
  cublasStatus_t status;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cublasHandle_t cublasHandle;
  cublasCreate_v2(&cublasHandle);
  int8_t *A, *B;
  int32_t *C;
  int32_t alpha = 1, beta = 0;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaMallocAsync(&A, batch * q_len * embed_dim * sizeof(int8_t), stream);
  cudaMallocAsync(&B, batch * kv_len * embed_dim * sizeof(int8_t), stream);
  cudaMallocAsync(&C, batch * q_len * kv_len * sizeof(int32_t), stream);

  printf("[QKVPlugin::configurePlugin]: *******************Testing Algo "
         "1*******************\n");
  for (cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT; algo <= CUBLAS_GEMM_ALGO23;
       algo = (cublasGemmAlgo_t)(algo + 1)) {
    cudaEventRecord(start, stream);
    status = cublasGemmStridedBatchedWrap_int8(
        cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, kv_len, q_len, embed_dim,
        &alpha, B, embed_dim, kv_len * embed_dim, A, embed_dim,
        q_len * embed_dim, &beta, C, kv_len, q_len * kv_len, batch, algo);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

    if (status == CUBLAS_STATUS_SUCCESS) {
      printf("[QKVPlugin::configurePlugin]: CUBLAS SUCCESS, TIME: %f\n", time);
      if (time < minTime) {
        minTime = time;
        mAlgo_1 = algo;
      }
    } else {
      printf("[QKVPlugin::configurePlugin]: CUBLAS FAIL, Code: %d\n", status);
    }
  }
  for (cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
       algo <= CUBLAS_GEMM_ALGO15_TENSOR_OP;
       algo = (cublasGemmAlgo_t)(algo + 1)) {
    cudaEventRecord(start, stream);
    status = cublasGemmStridedBatchedWrap_int8(
        cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, kv_len, q_len, embed_dim,
        &alpha, B, embed_dim, kv_len * embed_dim, A, embed_dim,
        q_len * embed_dim, &beta, C, kv_len, q_len * kv_len, batch, algo);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

    if (status == CUBLAS_STATUS_SUCCESS) {
      printf("[QKVPlugin::configurePlugin]: CUBLAS SUCCESS, TIME: %f\n", time);
      if (time < minTime) {
        minTime = time;
        mAlgo_1 = algo;
      }
    } else {
      printf("[QKVPlugin::configurePlugin]: CUBLAS FAIL, Code: %d\n", status);
    }
  }
  printf("[QKVPlugin::configurePlugin]: *******************Fastest Algo_1 %d, "
         "Time: %f*******************\n",
         mAlgo_1, minTime);

  cudaFreeAsync(A, stream);
  cudaFreeAsync(B, stream);
  cudaFreeAsync(C, stream);
  cudaMallocAsync(&A, batch * kv_len * embed_dim * sizeof(int8_t), stream);
  cudaMallocAsync(&B, batch * kv_len * q_len * sizeof(int8_t), stream);
  cudaMallocAsync(&C, batch * q_len * embed_dim * sizeof(int32_t), stream);

  minTime = INFINITY;
  printf("[QKVPlugin::configurePlugin]: *******************Testing Algo "
         "2*******************\n");
  for (cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT; algo <= CUBLAS_GEMM_ALGO23;
       algo = (cublasGemmAlgo_t)(algo + 1)) {
    cudaEventRecord(start, stream);
    status = cublasGemmStridedBatchedWrap_int8(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, embed_dim, q_len, kv_len,
        &alpha, A, embed_dim, kv_len * embed_dim, B, kv_len, q_len * kv_len,
        &beta, C, embed_dim, q_len * embed_dim, batch, algo);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

    if (status == CUBLAS_STATUS_SUCCESS) {
      printf("[QKVPlugin::configurePlugin]: CUBLAS SUCCESS, TIME: %f\n", time);
      if (time < minTime) {
        minTime = time;
        mAlgo_2 = algo;
      }
    } else {
      printf("[QKVPlugin::configurePlugin]: CUBLAS FAIL, Code: %d\n", status);
    }
  }
  for (cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
       algo <= CUBLAS_GEMM_ALGO15_TENSOR_OP;
       algo = (cublasGemmAlgo_t)(algo + 1)) {
    cudaEventRecord(start, stream);
    status = cublasGemmStridedBatchedWrap_int8(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, embed_dim, q_len, kv_len,
        &alpha, A, embed_dim, kv_len * embed_dim, B, kv_len, q_len * kv_len,
        &beta, C, embed_dim, q_len * embed_dim, batch, algo);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

    if (status == CUBLAS_STATUS_SUCCESS) {
      printf("[QKVPlugin::configurePlugin]: CUBLAS SUCCESS, TIME: %f\n", time);
      if (time < minTime) {
        minTime = time;
        mAlgo_2 = algo;
      }
    } else {
      printf("[QKVPlugin::configurePlugin]: CUBLAS FAIL, Code: %d\n", status);
    }
  }
  printf("[QKVPlugin::configurePlugin]: *******************Fastest Algo_2 %d, "
         "Time: %f*******************\n",
         mAlgo_1, minTime);
  cudaFreeAsync(A, stream);
  cudaFreeAsync(B, stream);
  cudaFreeAsync(C, stream);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cublasDestroy_v2(cublasHandle);
  cudaStreamDestroy(stream);
}

QKVPluginCreator::QKVPluginCreator() {
  mPluginAttributes.clear();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

char const *QKVPluginCreator::getPluginName() const noexcept {
  return QKV_PLUGIN_NAME;
}

char const *QKVPluginCreator::getPluginVersion() const noexcept {
  return QKV_PLUGIN_VERSION;
}

PluginFieldCollection const *QKVPluginCreator::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2DynamicExt *QKVPluginCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  try {
    auto *plugin = new QKVPlugin(false);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV2DynamicExt *
QKVPluginCreator::deserializePlugin(const char *name, const void *serialData,
                                    size_t serialLength) noexcept {
  try {
    auto *plugin = new QKVPlugin{serialData, serialLength, false};
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

QKVPluginCreator2::QKVPluginCreator2() {
  mPluginAttributes.clear();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

char const *QKVPluginCreator2::getPluginName() const noexcept {
  return QKV_PLUGIN_NAME2;
}

char const *QKVPluginCreator2::getPluginVersion() const noexcept {
  return QKV_PLUGIN_VERSION;
}

PluginFieldCollection const *QKVPluginCreator2::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2DynamicExt *QKVPluginCreator2::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  try {
    auto *plugin = new QKVPlugin(true);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV2DynamicExt *
QKVPluginCreator2::deserializePlugin(const char *name, const void *serialData,
                                     size_t serialLength) noexcept {
  try {
    auto *plugin = new QKVPlugin{serialData, serialLength, true};
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

REGISTER_TENSORRT_PLUGIN(QKVPluginCreator);
REGISTER_TENSORRT_PLUGIN(QKVPluginCreator2);
