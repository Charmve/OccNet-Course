//
// Created by Derry Lin on 2023/05/11.
//

#include "inversePlugin.h"
#include "checkMacrosPlugin.h"
#include "inverseKernel.h"
#include "serialize.h"
#include <cuda_fp16.h>
#include <stdexcept>

using trt_plugin::InversePlugin;
using trt_plugin::InversePluginCreator;
using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace {
constexpr char const *BP_PLUGIN_VERSION{"1"};
constexpr char const *BP_PLUGIN_NAME{"InverseTRT"};
} // namespace

PluginFieldCollection InversePluginCreator::mFC{};
std::vector<PluginField> InversePluginCreator::mPluginAttributes;

InversePlugin::InversePlugin() {}

InversePlugin::InversePlugin(const void *serialData, size_t serialLength) {}

InversePlugin::~InversePlugin() { terminate(); }

int32_t InversePlugin::getNbOutputs() const noexcept { return 1; }

DimsExprs InversePlugin::getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) noexcept {
  return inputs[0];
}

int32_t InversePlugin::initialize() noexcept { return 0; }

void InversePlugin::terminate() noexcept {
  if (config) {
    delete[] inpPtrArr;
    delete[] outPtrArr;
  }
  config = false;
}

size_t
InversePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                int32_t nbInputs,
                                const nvinfer1::PluginTensorDesc *outputs,
                                int32_t nbOutputs) const noexcept {
  return matrix_n * 2 * sizeof(float *) +
         matrix_n * (matrix_dim + 1) * sizeof(int);
}

int32_t InversePlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                               const nvinfer1::PluginTensorDesc *outputDesc,
                               const void *const *inputs, void *const *outputs,
                               void *workspace, cudaStream_t stream) noexcept {
  auto data_type = inputDesc[0].type;
  switch (data_type) {
  case DataType::kFLOAT:
    inverse((float *)outputs[0], (float *)inputs[0], workspace,
            (float **)inpPtrArr, (float **)outPtrArr, matrix_n, matrix_dim,
            m_cublas_handle, stream);
    break;
  default:
    return 1;
  }
  return 0;
}

size_t InversePlugin::getSerializationSize() const noexcept { return 0; }

void InversePlugin::serialize(void *buffer) const noexcept {}

bool InversePlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc *inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept {
  if (pos == 0) {
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT &&
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  }
}

char const *InversePlugin::getPluginType() const noexcept {
  return BP_PLUGIN_NAME;
}

char const *InversePlugin::getPluginVersion() const noexcept {
  return BP_PLUGIN_VERSION;
}

void InversePlugin::destroy() noexcept { delete this; }

IPluginV2DynamicExt *InversePlugin::clone() const noexcept {
  try {
    auto *plugin = new InversePlugin();
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

void InversePlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
  mPluginNamespace = pluginNamespace;
}

char const *InversePlugin::getPluginNamespace() const noexcept {
  return mPluginNamespace.c_str();
}

DataType InversePlugin::getOutputDataType(int32_t index,
                                          const nvinfer1::DataType *inputTypes,
                                          int32_t nbInputs) const noexcept {
  PLUGIN_ASSERT(inputTypes && nbInputs > 0 && index == 0)
  return inputTypes[0];
}

void InversePlugin::attachToContext(
    cudnnContext *cudnn, cublasContext *cublas,
    nvinfer1::IGpuAllocator *allocator) noexcept {
  m_cublas_handle = cublas;
}

void InversePlugin::detachFromContext() noexcept {}

void InversePlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
  PLUGIN_ASSERT(nbInputs == 1);
  PLUGIN_ASSERT(in[0].desc.dims.d[in[0].desc.dims.nbDims - 1] ==
                in[0].desc.dims.d[in[0].desc.dims.nbDims - 2]);
  matrix_n = 1;
  for (int i = 0; i < in[0].desc.dims.nbDims - 2; i++) {
    matrix_n *= in[0].desc.dims.d[i];
  }
  matrix_dim = in[0].desc.dims.d[in[0].desc.dims.nbDims - 1];

  inpPtrArr = new float *[matrix_n];
  outPtrArr = new float *[matrix_n];

  config = true;
}

InversePluginCreator::InversePluginCreator() {
  mPluginAttributes.clear();

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

char const *InversePluginCreator::getPluginName() const noexcept {
  return BP_PLUGIN_NAME;
}

char const *InversePluginCreator::getPluginVersion() const noexcept {
  return BP_PLUGIN_VERSION;
}

PluginFieldCollection const *InversePluginCreator::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2DynamicExt *InversePluginCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  try {
    auto *plugin = new InversePlugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV2DynamicExt *InversePluginCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) noexcept {
  try {
    auto *plugin = new InversePlugin{serialData, serialLength};
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

REGISTER_TENSORRT_PLUGIN(InversePluginCreator);
