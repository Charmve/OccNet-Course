//
// Created by Derry Lin on 2022/11/7.
//

#include "multiScaleDeformableAttnPlugin.h"
#include "checkMacrosPlugin.h"
#include "multiScaleDeformableAttnKernel.h"
#include "serialize.h"
#include <cuda_fp16.h>
#include <iostream>
#include <stdexcept>

using trt_plugin::MultiScaleDeformableAttnPlugin;
using trt_plugin::MultiScaleDeformableAttnPluginCreator;
using trt_plugin::MultiScaleDeformableAttnPluginCreator2;
using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace {
constexpr char const *MSDA_PLUGIN_VERSION{"1"};
constexpr char const *MSDA_PLUGIN_NAME{"MultiScaleDeformableAttnTRT"};
constexpr char const *MSDA_PLUGIN_NAME2{"MultiScaleDeformableAttnTRT2"};
} // namespace

PluginFieldCollection MultiScaleDeformableAttnPluginCreator::mFC{};
std::vector<PluginField>
    MultiScaleDeformableAttnPluginCreator::mPluginAttributes;

PluginFieldCollection MultiScaleDeformableAttnPluginCreator2::mFC{};
std::vector<PluginField>
    MultiScaleDeformableAttnPluginCreator2::mPluginAttributes;

MultiScaleDeformableAttnPlugin::MultiScaleDeformableAttnPlugin(bool use_h2)
    : use_h2(use_h2) {}

MultiScaleDeformableAttnPlugin::MultiScaleDeformableAttnPlugin(
    const void *serialData, size_t serialLength, bool use_h2)
    : use_h2(use_h2) {}

MultiScaleDeformableAttnPlugin::~MultiScaleDeformableAttnPlugin() throw() {
  terminate();
}

int32_t MultiScaleDeformableAttnPlugin::getNbOutputs() const noexcept {
  return 1;
}

DimsExprs MultiScaleDeformableAttnPlugin::getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) noexcept {
  DimsExprs outputDim;
  outputDim.nbDims = 4;
  outputDim.d[0] = inputs[0].d[0];
  outputDim.d[1] = inputs[3].d[1];
  outputDim.d[2] = inputs[0].d[2];
  outputDim.d[3] = inputs[0].d[3];
  return outputDim;
}

int32_t MultiScaleDeformableAttnPlugin::initialize() noexcept { return 0; }

void MultiScaleDeformableAttnPlugin::terminate() noexcept {}

size_t MultiScaleDeformableAttnPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int32_t nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int32_t nbOutputs) const noexcept {
  return 0;
}

int32_t MultiScaleDeformableAttnPlugin::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  float scale_value = inputDesc[0].scale, scale_ref = inputDesc[2].scale,
        scale_offset = inputDesc[3].scale, scale_weight = inputDesc[4].scale,
        scale_out = outputDesc[0].scale;
  Dims value_dims = inputDesc[0].dims;
  const int batch = value_dims.d[0];
  const int spatial_size = value_dims.d[1];
  const int num_heads = value_dims.d[2];
  const int channels = value_dims.d[3];

  const int num_levels = inputDesc[1].dims.d[0];

  const int points_per_group = inputDesc[2].dims.d[3] / 2;
  const int num_query = inputDesc[3].dims.d[1];
  const int num_point = inputDesc[4].dims.d[3] / num_levels;

  auto data_type = inputDesc[0].type;
  auto data_type_rp = inputDesc[2].type;
  ASSERT(data_type == DataType::kFLOAT || data_type == DataType::kHALF ||
         data_type == DataType::kINT8)
  ASSERT(data_type_rp == DataType::kFLOAT || data_type_rp == DataType::kHALF)

  switch (data_type) {
  case DataType::kFLOAT:
    ms_deformable_im2col_cuda<float>(
        (float *)inputs[0], (int32_t *)inputs[1], (float *)inputs[2],
        (float *)inputs[3], (float *)inputs[4], batch, spatial_size, num_heads,
        channels, num_levels, num_query, num_point, points_per_group,
        (float *)outputs[0], stream);
    break;
  case DataType::kHALF:
    if (use_h2) {
      ms_deformable_im2col_cuda_h2(
          (__half2 *)inputs[0], (int32_t *)inputs[1], (__half2 *)inputs[2],
          (__half2 *)inputs[3], (__half *)inputs[4], batch, spatial_size,
          num_heads, channels, num_levels, num_query, num_point,
          points_per_group, (__half2 *)outputs[0], stream);
    } else {
      ms_deformable_im2col_cuda<__half>(
          (__half *)inputs[0], (int32_t *)inputs[1], (__half *)inputs[2],
          (__half *)inputs[3], (__half *)inputs[4], batch, spatial_size,
          num_heads, channels, num_levels, num_query, num_point,
          points_per_group, (__half *)outputs[0], stream);
    }
    break;
  case DataType::kINT8:
    if (data_type_rp == DataType::kHALF) {
      ms_deformable_im2col_cuda_int8<__half2>(
          (int8_4 *)inputs[0], scale_value, (int32_t *)inputs[1],
          (__half2 *)inputs[2], (int8_4 *)inputs[3], scale_offset,
          (int8_4 *)inputs[4], scale_weight, batch, spatial_size, num_heads,
          channels, num_levels, num_query, num_point, points_per_group,
          (int8_4 *)outputs[0], scale_out, stream);
    } else {
      ms_deformable_im2col_cuda_int8<float>(
          (int8_4 *)inputs[0], scale_value, (int32_t *)inputs[1],
          (float *)inputs[2], (int8_4 *)inputs[3], scale_offset,
          (int8_4 *)inputs[4], scale_weight, batch, spatial_size, num_heads,
          channels, num_levels, num_query, num_point, points_per_group,
          (int8_4 *)outputs[0], scale_out, stream);
    }
    break;
  default:
    return 1;
  }
  return 0;
}

size_t MultiScaleDeformableAttnPlugin::getSerializationSize() const noexcept {
  return 0;
}

void MultiScaleDeformableAttnPlugin::serialize(void *buffer) const noexcept {}

bool MultiScaleDeformableAttnPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc *inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept {
  const int channels = inOut[0].dims.d[3];
  const int point_num = inOut[4].dims.d[3] / inOut[1].dims.d[0];
  bool use_int8 = true;
  if (channels % 4 != 0 || point_num % 4 != 0) {
    use_int8 = false;
  }
  switch (pos) {
  case 0:
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT &&
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR) ||
           (inOut[pos].type == nvinfer1::DataType::kHALF &&
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR) ||
           (inOut[pos].type == nvinfer1::DataType::kINT8 &&
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR && use_int8);
  case 1:
    return inOut[pos].type == nvinfer1::DataType::kINT32 &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  case 2:
    if (inOut[0].type == nvinfer1::DataType::kFLOAT ||
        inOut[0].type == nvinfer1::DataType::kHALF) {
      return inOut[pos].type == inOut[0].type &&
             inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }
    return (inOut[pos].type == nvinfer1::DataType::kHALF ||
            inOut[pos].type == nvinfer1::DataType::kFLOAT) &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  case 3:
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  case 4:
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  case 5:
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == inOut[0].format;
  default:
    return false;
  }
}

char const *MultiScaleDeformableAttnPlugin::getPluginType() const noexcept {
  return use_h2 ? MSDA_PLUGIN_NAME2 : MSDA_PLUGIN_NAME;
}

char const *MultiScaleDeformableAttnPlugin::getPluginVersion() const noexcept {
  return MSDA_PLUGIN_VERSION;
}

void MultiScaleDeformableAttnPlugin::destroy() noexcept { delete this; }

IPluginV2DynamicExt *MultiScaleDeformableAttnPlugin::clone() const noexcept {
  try {
    auto *plugin = new MultiScaleDeformableAttnPlugin{use_h2};
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

void MultiScaleDeformableAttnPlugin::setPluginNamespace(
    const char *pluginNamespace) noexcept {
  mPluginNamespace = pluginNamespace;
}

char const *
MultiScaleDeformableAttnPlugin::getPluginNamespace() const noexcept {
  return mPluginNamespace.c_str();
}

DataType MultiScaleDeformableAttnPlugin::getOutputDataType(
    int32_t index, const nvinfer1::DataType *inputTypes,
    int32_t nbInputs) const noexcept {
  PLUGIN_ASSERT(inputTypes && nbInputs > 0 && index == 0)
  return inputTypes[0];
}

void MultiScaleDeformableAttnPlugin::attachToContext(
    cudnnContext *cudnn, cublasContext *cublas,
    nvinfer1::IGpuAllocator *allocator) noexcept {}

void MultiScaleDeformableAttnPlugin::detachFromContext() noexcept {}

void MultiScaleDeformableAttnPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
  PLUGIN_ASSERT(nbInputs == 5)
  const int channels = in[0].desc.dims.d[3];
  if (use_h2) {
    if (channels % 2 != 0) {
      use_h2 = false;
    }
  }
}

MultiScaleDeformableAttnPluginCreator::MultiScaleDeformableAttnPluginCreator() {
  mPluginAttributes.clear();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

char const *
MultiScaleDeformableAttnPluginCreator::getPluginName() const noexcept {
  return MSDA_PLUGIN_NAME;
}

char const *
MultiScaleDeformableAttnPluginCreator::getPluginVersion() const noexcept {
  return MSDA_PLUGIN_VERSION;
}

PluginFieldCollection const *
MultiScaleDeformableAttnPluginCreator::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2DynamicExt *MultiScaleDeformableAttnPluginCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  try {
    auto *plugin = new MultiScaleDeformableAttnPlugin(false);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV2DynamicExt *MultiScaleDeformableAttnPluginCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) noexcept {
  try {
    auto *plugin =
        new MultiScaleDeformableAttnPlugin{serialData, serialLength, false};
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

MultiScaleDeformableAttnPluginCreator2::
    MultiScaleDeformableAttnPluginCreator2() {
  mPluginAttributes.clear();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

char const *
MultiScaleDeformableAttnPluginCreator2::getPluginName() const noexcept {
  return MSDA_PLUGIN_NAME2;
}

char const *
MultiScaleDeformableAttnPluginCreator2::getPluginVersion() const noexcept {
  return MSDA_PLUGIN_VERSION;
}

PluginFieldCollection const *
MultiScaleDeformableAttnPluginCreator2::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2DynamicExt *MultiScaleDeformableAttnPluginCreator2::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  try {
    auto *plugin = new MultiScaleDeformableAttnPlugin(true);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV2DynamicExt *MultiScaleDeformableAttnPluginCreator2::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) noexcept {
  try {
    auto *plugin =
        new MultiScaleDeformableAttnPlugin{serialData, serialLength, true};
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

REGISTER_TENSORRT_PLUGIN(MultiScaleDeformableAttnPluginCreator);
REGISTER_TENSORRT_PLUGIN(MultiScaleDeformableAttnPluginCreator2);
