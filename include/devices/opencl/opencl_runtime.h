
#ifndef FASTLLM_OPENCL_RUNTIME_H
#define FASTLLM_OPENCL_RUNTIME_H

#include <CL/opencl.hpp>
#include <map>
#include <memory>

namespace fastllm {

const std::string OpenCLErrorToString(cl_int error);

enum GPUType {
  GPU_ARM_MALI, GPU_QCOM_ADERNO, GPU_OTHER
};

class OpenCLRuntime {
public:
  static OpenCLRuntime* GetGlobalOpenCLRuntime();

  ~OpenCLRuntime();

  OpenCLRuntime(const OpenCLRuntime &r) = delete;

  OpenCLRuntime &operator=(const OpenCLRuntime &r) = delete;

  cl::Context &context() const;

  cl::CommandQueue &command_queue() const;

  cl::Device &device() const;

  void BuildKernel(const std::string &programName,
                   const std::vector<std::string> &buildOptions,
                   const std::string &kernelName, cl::Kernel *kernel);

  std::string GetPlatformInfo() const;
  uint64_t GetDeviceMaxWorkGroupSize() const;
  uint64_t GetDeviceMaxMemAllocSize() const;
  uint64_t GetKernelMaxWorkGroupSize(const cl::Kernel &kernel) const;

private:
  OpenCLRuntime();

  void BuildProgram(const std::string &programName,
                    const std::string &buildOptions, cl::Program *program);

  std::vector<unsigned char> *GetOpenCLPrograms(const std::string &name);

  std::shared_ptr<cl::Device> device_;
  std::shared_ptr<cl::Context> context_;
  std::shared_ptr<cl::CommandQueue> queue_;
  std::map<std::string, cl::Program> builtPrograms_;

  std::string platformInfo_;
  uint64_t globalMemCacheLineSize_;
  uint32_t deviceComputeUnits_;
  GPUType gpuType_;
};



}; // namespace fastllm

#endif
