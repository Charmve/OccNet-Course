//
// Created by Derry Lin on 2022/10/22.
//

#ifndef TENSORRT_OPS_GRIDSAMPLERPLUGIN_H
#define TENSORRT_OPS_GRIDSAMPLERPLUGIN_H

#include "gridSamplerKernel.h"
#include "helper.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <string>
#include <vector>

namespace trt_plugin {

class GridSamplerPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
  GridSamplerPlugin(int mode, int paddingMode, bool alignCorners, bool use_h2,
                    bool _3D);
  GridSamplerPlugin(const void *serialData, size_t serialLength, bool use_h2,
                    bool _3D);
  GridSamplerPlugin() = delete;
  ~GridSamplerPlugin() override;

  int32_t getNbOutputs() const noexcept override;

  nvinfer1::DimsExprs
  getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const *inputs,
                      int32_t nbInputs,
                      nvinfer1::IExprBuilder &exprBuilder) noexcept override;

  int32_t initialize() noexcept override;

  void terminate() noexcept override;

  size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const *inputs,
                          int32_t nbInputs,
                          nvinfer1::PluginTensorDesc const *outputs,
                          int32_t nbOutputs) const noexcept override;

  int32_t enqueue(nvinfer1::PluginTensorDesc const *inputDesc,
                  nvinfer1::PluginTensorDesc const *outputDesc,
                  void const *const *inputs, void *const *outputs,
                  void *workspace, cudaStream_t stream) noexcept override;

  size_t getSerializationSize() const noexcept override;

  void serialize(void *buffer) const noexcept override;

  bool supportsFormatCombination(int32_t pos,
                                 nvinfer1::PluginTensorDesc const *inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override;

  char const *getPluginType() const noexcept override;

  char const *getPluginVersion() const noexcept override;

  void destroy() noexcept override;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;

  void setPluginNamespace(char const *pluginNamespace) noexcept override;

  char const *getPluginNamespace() const noexcept override;

  nvinfer1::DataType
  getOutputDataType(int32_t index, nvinfer1::DataType const *inputTypes,
                    int32_t nbInputs) const noexcept override;

  void attachToContext(cudnnContext *cudnn, cublasContext *cublas,
                       nvinfer1::IGpuAllocator *allocator) noexcept override;

  void detachFromContext() noexcept override;

  void configurePlugin(nvinfer1::DynamicPluginTensorDesc const *in,
                       int32_t nbInputs,
                       nvinfer1::DynamicPluginTensorDesc const *out,
                       int32_t nbOutputs) noexcept override;

private:
  GridSamplerInterpolation mMode;
  GridSamplerPadding mPaddingMode;
  bool mAlignCorners{};
  bool m3D{false};

  std::string mPluginNamespace;
  std::string mNamespace;
  bool use_h2;
};

class GridSampler2DPluginCreator : public trt_plugin::BaseCreator {
public:
  GridSampler2DPluginCreator();
  ~GridSampler2DPluginCreator() override = default;

  char const *getPluginName() const noexcept override;

  char const *getPluginVersion() const noexcept override;

  nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override;

  nvinfer1::IPluginV2DynamicExt *
  createPlugin(char const *name,
               const nvinfer1::PluginFieldCollection *fc) noexcept override;

  nvinfer1::IPluginV2DynamicExt *
  deserializePlugin(char const *name, void const *serialData,
                    size_t serialLength) noexcept override;

private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

class GridSampler2DPluginCreator2 : public trt_plugin::BaseCreator {
public:
  GridSampler2DPluginCreator2();
  ~GridSampler2DPluginCreator2() override = default;

  char const *getPluginName() const noexcept override;

  char const *getPluginVersion() const noexcept override;

  nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override;

  nvinfer1::IPluginV2DynamicExt *
  createPlugin(char const *name,
               const nvinfer1::PluginFieldCollection *fc) noexcept override;

  nvinfer1::IPluginV2DynamicExt *
  deserializePlugin(char const *name, void const *serialData,
                    size_t serialLength) noexcept override;

private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

class GridSampler3DPluginCreator : public trt_plugin::BaseCreator {
public:
  GridSampler3DPluginCreator();
  ~GridSampler3DPluginCreator() override = default;

  char const *getPluginName() const noexcept override;

  char const *getPluginVersion() const noexcept override;

  nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override;

  nvinfer1::IPluginV2DynamicExt *
  createPlugin(char const *name,
               const nvinfer1::PluginFieldCollection *fc) noexcept override;

  nvinfer1::IPluginV2DynamicExt *
  deserializePlugin(char const *name, void const *serialData,
                    size_t serialLength) noexcept override;

private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

class GridSampler3DPluginCreator2 : public trt_plugin::BaseCreator {
public:
  GridSampler3DPluginCreator2();
  ~GridSampler3DPluginCreator2() override = default;

  char const *getPluginName() const noexcept override;

  char const *getPluginVersion() const noexcept override;

  nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override;

  nvinfer1::IPluginV2DynamicExt *
  createPlugin(char const *name,
               const nvinfer1::PluginFieldCollection *fc) noexcept override;

  nvinfer1::IPluginV2DynamicExt *
  deserializePlugin(char const *name, void const *serialData,
                    size_t serialLength) noexcept override;

private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace trt_plugin

#endif // TENSORRT_OPS_GRIDSAMPLERPLUGIN_H
