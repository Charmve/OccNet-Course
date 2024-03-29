//
// Created by Derry Lin on 2022/10/22.
//
#include "gridSamplerPlugin.h"
#include "checkMacrosPlugin.h"
#include "gridSamplerKernel.h"
#include "serialize.h"
#include <cuda_fp16.h>
#include <iostream>
#include <stdexcept>

using trt_plugin::GridSampler2DPluginCreator;
using trt_plugin::GridSampler2DPluginCreator2;
using trt_plugin::GridSampler3DPluginCreator;
using trt_plugin::GridSampler3DPluginCreator2;
using trt_plugin::GridSamplerPlugin;
using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace {
constexpr char const *GS_PLUGIN_VERSION{"1"};
constexpr char const *GS2D_PLUGIN_NAME{"GridSampler2DTRT"};
constexpr char const *GS2D_PLUGIN_NAME2{"GridSampler2DTRT2"};
constexpr char const *GS3D_PLUGIN_NAME{"GridSampler3DTRT"};
constexpr char const *GS3D_PLUGIN_NAME2{"GridSampler3DTRT2"};
} // namespace

PluginFieldCollection GridSampler2DPluginCreator::mFC{};
std::vector<PluginField> GridSampler2DPluginCreator::mPluginAttributes;

PluginFieldCollection GridSampler2DPluginCreator2::mFC{};
std::vector<PluginField> GridSampler2DPluginCreator2::mPluginAttributes;

PluginFieldCollection GridSampler3DPluginCreator::mFC{};
std::vector<PluginField> GridSampler3DPluginCreator::mPluginAttributes;

PluginFieldCollection GridSampler3DPluginCreator2::mFC{};
std::vector<PluginField> GridSampler3DPluginCreator2::mPluginAttributes;

GridSamplerPlugin::GridSamplerPlugin(int mode, int paddingMode,
                                     bool alignCorners, bool use_h2, bool _3D)
    : mAlignCorners(alignCorners), use_h2(use_h2), m3D(_3D) {
  ASSERT(mode == 0 || mode == 1 || mode == 2)
  ASSERT(paddingMode == 0 || paddingMode == 1 || paddingMode == 2)
  switch (mode) {
  case 0:
    mMode = GridSamplerInterpolation::Bilinear;
    break;
  case 1:
    mMode = GridSamplerInterpolation::Nearest;
    break;
  case 2:
    mMode = GridSamplerInterpolation::Bicubic;
    break;
  default:
    break;
  }
  switch (paddingMode) {
  case 0:
    mPaddingMode = GridSamplerPadding::Zeros;
    break;
  case 1:
    mPaddingMode = GridSamplerPadding::Border;
    break;
  case 2:
    mPaddingMode = GridSamplerPadding::Reflection;
    break;
  default:
    break;
  }
}

GridSamplerPlugin::GridSamplerPlugin(const void *serialData,
                                     size_t serialLength, bool use_h2, bool _3D)
    : use_h2(use_h2), m3D(_3D) {
  deserialize_value(&serialData, &serialLength, &mMode);
  deserialize_value(&serialData, &serialLength, &mPaddingMode);
  deserialize_value(&serialData, &serialLength, &mAlignCorners);
}

GridSamplerPlugin::~GridSamplerPlugin() { terminate(); }

int32_t GridSamplerPlugin::getNbOutputs() const noexcept { return 1; }

DimsExprs GridSamplerPlugin::getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) noexcept {
  DimsExprs outputDim;
  outputDim.nbDims = inputs[0].nbDims;
  outputDim.d[0] = inputs[0].d[0];
  outputDim.d[1] = inputs[0].d[1];
  for (int i = 2; i < outputDim.nbDims; i++) {
    outputDim.d[i] = inputs[1].d[i];
  }
  return outputDim;
}

int32_t GridSamplerPlugin::initialize() noexcept { return 0; }

void GridSamplerPlugin::terminate() noexcept {}

size_t
GridSamplerPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                    int32_t nbInputs,
                                    const nvinfer1::PluginTensorDesc *outputs,
                                    int32_t nbOutputs) const noexcept {
  return 0;
}

int32_t GridSamplerPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                   const nvinfer1::PluginTensorDesc *outputDesc,
                                   const void *const *inputs,
                                   void *const *outputs, void *workspace,
                                   cudaStream_t stream) noexcept {
  Dims input_dims = inputDesc[0].dims;
  Dims grid_dims = inputDesc[1].dims;
  Dims output_dims = outputDesc[0].dims;
  const float scale_o = outputDesc[0].scale, scale_i = inputDesc[0].scale,
              scale_g = inputDesc[1].scale;

  auto data_type = inputDesc[0].type;
  ASSERT(data_type == DataType::kFLOAT || data_type == DataType::kHALF ||
         data_type == DataType::kINT8)

  switch (data_type) {
  case DataType::kFLOAT:
    grid_sample<float>((float *)outputs[0], (float *)inputs[0],
                       (float *)inputs[1], &(output_dims.d[0]),
                       &(input_dims.d[0]), &(grid_dims.d[0]), input_dims.nbDims,
                       mMode, mPaddingMode, mAlignCorners, stream);
    break;
  case DataType::kHALF:
    if (use_h2) {
      grid_sample<__half2>(
          (__half2 *)outputs[0], (__half2 *)inputs[0], (__half2 *)inputs[1],
          &(output_dims.d[0]), &(input_dims.d[0]), &(grid_dims.d[0]),
          input_dims.nbDims, mMode, mPaddingMode, mAlignCorners, stream);
    } else {
      grid_sample<__half>(
          (__half *)outputs[0], (__half *)inputs[0], (__half *)inputs[1],
          &(output_dims.d[0]), &(input_dims.d[0]), &(grid_dims.d[0]),
          input_dims.nbDims, mMode, mPaddingMode, mAlignCorners, stream);
    }
    break;
  case DataType::kINT8:
    grid_sample_int8((int8_4 *)outputs[0], scale_o, (int8_4 *)inputs[0],
                     scale_i, (int8_4 *)inputs[1], scale_g, &(output_dims.d[0]),
                     &(input_dims.d[0]), &(grid_dims.d[0]), input_dims.nbDims,
                     mMode, mPaddingMode, mAlignCorners, stream);
    break;
  default:
    return 1;
  }
  return 0;
}

size_t GridSamplerPlugin::getSerializationSize() const noexcept {
  return serialized_size(mMode) + serialized_size(mPaddingMode) +
         serialized_size(mAlignCorners);
}

void GridSamplerPlugin::serialize(void *buffer) const noexcept {
  serialize_value(&buffer, mMode);
  serialize_value(&buffer, mPaddingMode);
  serialize_value(&buffer, mAlignCorners);
}

bool GridSamplerPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc *inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept {
  if (pos == 0) {
    if (use_h2 && !m3D) {
      return (inOut[pos].type == nvinfer1::DataType::kFLOAT &&
              inOut[pos].format == nvinfer1::TensorFormat::kLINEAR) ||
             (inOut[pos].type == nvinfer1::DataType::kHALF &&
              inOut[pos].format == nvinfer1::TensorFormat::kCHW2) ||
             (inOut[pos].type == nvinfer1::DataType::kINT8 &&
              inOut[pos].format == nvinfer1::TensorFormat::kCHW4);
    }
    if (!m3D) {
      return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
               inOut[pos].type == nvinfer1::DataType::kHALF) &&
              inOut[pos].format == nvinfer1::TensorFormat::kLINEAR) ||
             (inOut[pos].type == nvinfer1::DataType::kINT8 &&
              inOut[pos].format == nvinfer1::TensorFormat::kCHW4);
    }
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
             inOut[pos].type == nvinfer1::DataType::kHALF) &&
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == inOut[0].format;
  }
}

char const *GridSamplerPlugin::getPluginType() const noexcept {
  return use_h2 ? (m3D ? GS3D_PLUGIN_NAME2 : GS2D_PLUGIN_NAME2)
                : (m3D ? GS3D_PLUGIN_NAME : GS2D_PLUGIN_NAME);
}

char const *GridSamplerPlugin::getPluginVersion() const noexcept {
  return GS_PLUGIN_VERSION;
}

void GridSamplerPlugin::destroy() noexcept { delete this; }

IPluginV2DynamicExt *GridSamplerPlugin::clone() const noexcept {
  try {
    int mode, paddingMode;
    switch (mMode) {
    case GridSamplerInterpolation::Bilinear:
      mode = 0;
      break;
    case GridSamplerInterpolation::Nearest:
      mode = 1;
      break;
    case GridSamplerInterpolation::Bicubic:
      mode = 2;
      break;
    default:
      break;
    }
    switch (mPaddingMode) {
    case GridSamplerPadding::Zeros:
      paddingMode = 0;
      break;
    case GridSamplerPadding::Border:
      paddingMode = 1;
      break;
    case GridSamplerPadding::Reflection:
      paddingMode = 2;
      break;
    default:
      break;
    }
    auto *plugin =
        new GridSamplerPlugin{mode, paddingMode, mAlignCorners, use_h2, m3D};
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

void GridSamplerPlugin::setPluginNamespace(
    const char *pluginNamespace) noexcept {
  mPluginNamespace = pluginNamespace;
}

char const *GridSamplerPlugin::getPluginNamespace() const noexcept {
  return mPluginNamespace.c_str();
}

DataType
GridSamplerPlugin::getOutputDataType(int32_t index,
                                     const nvinfer1::DataType *inputTypes,
                                     int32_t nbInputs) const noexcept {
  PLUGIN_ASSERT(inputTypes && nbInputs > 0 && index == 0)
  return inputTypes[0];
}

void GridSamplerPlugin::attachToContext(
    cudnnContext *cudnn, cublasContext *cublas,
    nvinfer1::IGpuAllocator *allocator) noexcept {}

void GridSamplerPlugin::detachFromContext() noexcept {}

void GridSamplerPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
    PLUGIN_ASSERT(nbInputs == 2)
        PLUGIN_ASSERT(!(mMode == GridSamplerInterpolation::Bicubic &&
                        in[0].desc.dims.nbDims != 4))}

GridSampler2DPluginCreator::GridSampler2DPluginCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(
      PluginField("interpolation_mode", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("padding_mode", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("align_corners", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

char const *GridSampler2DPluginCreator::getPluginName() const noexcept {
  return GS2D_PLUGIN_NAME;
}

char const *GridSampler2DPluginCreator::getPluginVersion() const noexcept {
  return GS_PLUGIN_VERSION;
}

PluginFieldCollection const *
GridSampler2DPluginCreator::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2DynamicExt *GridSampler2DPluginCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  try {
    int mode, paddingMode;
    bool alignCorners;
    PluginField const *fields = fc->fields;
    for (int i = 0; i < fc->nbFields; i++) {
      char const *attrName = fields[i].name;
      if (!strcmp(attrName, "interpolation_mode")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        mode = *(static_cast<int const *>(fields[i].data));
      } else if (!strcmp(attrName, "padding_mode")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        paddingMode = *(static_cast<int const *>(fields[i].data));
      } else if (!strcmp(attrName, "align_corners")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        alignCorners = (bool)(*(static_cast<int const *>(fields[i].data)));
      }
    }

    auto *plugin =
        new GridSamplerPlugin(mode, paddingMode, alignCorners, false, false);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV2DynamicExt *GridSampler2DPluginCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) noexcept {
  try {
    auto *plugin =
        new GridSamplerPlugin{serialData, serialLength, false, false};
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

GridSampler2DPluginCreator2::GridSampler2DPluginCreator2() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(
      PluginField("interpolation_mode", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("padding_mode", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("align_corners", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

char const *GridSampler2DPluginCreator2::getPluginName() const noexcept {
  return GS2D_PLUGIN_NAME2;
}

char const *GridSampler2DPluginCreator2::getPluginVersion() const noexcept {
  return GS_PLUGIN_VERSION;
}

PluginFieldCollection const *
GridSampler2DPluginCreator2::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2DynamicExt *GridSampler2DPluginCreator2::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  try {
    int mode, paddingMode;
    bool alignCorners;
    PluginField const *fields = fc->fields;
    for (int i = 0; i < fc->nbFields; i++) {
      char const *attrName = fields[i].name;
      if (!strcmp(attrName, "interpolation_mode")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        mode = *(static_cast<int const *>(fields[i].data));
      } else if (!strcmp(attrName, "padding_mode")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        paddingMode = *(static_cast<int const *>(fields[i].data));
      } else if (!strcmp(attrName, "align_corners")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        alignCorners = (bool)(*(static_cast<int const *>(fields[i].data)));
      }
    }

    auto *plugin =
        new GridSamplerPlugin(mode, paddingMode, alignCorners, true, false);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV2DynamicExt *GridSampler2DPluginCreator2::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) noexcept {
  try {
    auto *plugin = new GridSamplerPlugin{serialData, serialLength, true, false};
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

GridSampler3DPluginCreator::GridSampler3DPluginCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(
      PluginField("interpolation_mode", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("padding_mode", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("align_corners", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

char const *GridSampler3DPluginCreator::getPluginName() const noexcept {
  return GS3D_PLUGIN_NAME;
}

char const *GridSampler3DPluginCreator::getPluginVersion() const noexcept {
  return GS_PLUGIN_VERSION;
}

PluginFieldCollection const *
GridSampler3DPluginCreator::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2DynamicExt *GridSampler3DPluginCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  try {
    int mode, paddingMode;
    bool alignCorners;
    PluginField const *fields = fc->fields;
    for (int i = 0; i < fc->nbFields; i++) {
      char const *attrName = fields[i].name;
      if (!strcmp(attrName, "interpolation_mode")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        mode = *(static_cast<int const *>(fields[i].data));
      } else if (!strcmp(attrName, "padding_mode")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        paddingMode = *(static_cast<int const *>(fields[i].data));
      } else if (!strcmp(attrName, "align_corners")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        alignCorners = (bool)(*(static_cast<int const *>(fields[i].data)));
      }
    }

    auto *plugin =
        new GridSamplerPlugin(mode, paddingMode, alignCorners, false, true);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV2DynamicExt *GridSampler3DPluginCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) noexcept {
  try {
    auto *plugin = new GridSamplerPlugin{serialData, serialLength, false, true};
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

GridSampler3DPluginCreator2::GridSampler3DPluginCreator2() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(
      PluginField("interpolation_mode", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("padding_mode", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("align_corners", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

char const *GridSampler3DPluginCreator2::getPluginName() const noexcept {
  return GS3D_PLUGIN_NAME2;
}

char const *GridSampler3DPluginCreator2::getPluginVersion() const noexcept {
  return GS_PLUGIN_VERSION;
}

PluginFieldCollection const *
GridSampler3DPluginCreator2::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2DynamicExt *GridSampler3DPluginCreator2::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  try {
    int mode, paddingMode;
    bool alignCorners;
    PluginField const *fields = fc->fields;
    for (int i = 0; i < fc->nbFields; i++) {
      char const *attrName = fields[i].name;
      if (!strcmp(attrName, "interpolation_mode")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        mode = *(static_cast<int const *>(fields[i].data));
      } else if (!strcmp(attrName, "padding_mode")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        paddingMode = *(static_cast<int const *>(fields[i].data));
      } else if (!strcmp(attrName, "align_corners")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        alignCorners = (bool)(*(static_cast<int const *>(fields[i].data)));
      }
    }

    auto *plugin =
        new GridSamplerPlugin(mode, paddingMode, alignCorners, true, true);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV2DynamicExt *GridSampler3DPluginCreator2::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) noexcept {
  try {
    auto *plugin = new GridSamplerPlugin{serialData, serialLength, true, true};
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

REGISTER_TENSORRT_PLUGIN(GridSampler2DPluginCreator);
REGISTER_TENSORRT_PLUGIN(GridSampler2DPluginCreator2);
REGISTER_TENSORRT_PLUGIN(GridSampler3DPluginCreator);
REGISTER_TENSORRT_PLUGIN(GridSampler3DPluginCreator2);
