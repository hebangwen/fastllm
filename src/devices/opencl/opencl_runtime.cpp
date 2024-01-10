#include "devices/opencl/opencl_runtime.h"
#include "spdlog/fmt/bundled/core.h"
#include "utils.h"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/opencl.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <spdlog/spdlog.h>
#include <string>

#define OPENCL_BASE_BUILD_OPTIONS                                              \
  "-cl-std=CL3.0 -cl-mad-enable -cl-fast-relaxed-math"

namespace fastllm {

#ifdef __aarch64__
void printf_callback( const char *buffer, size_t len, size_t complete, void *user_data ) {
    printf( "%.*s", len, buffer );
}
#endif

GPUType ParseGPUType(const std::string &device_name) {
  constexpr const char *kQualcommAdrenoGPUStr = "QUALCOMM Adreno(TM)";
  constexpr const char *kMaliGPUStr = "Mali";

  if (device_name == kQualcommAdrenoGPUStr) {
    return GPUType::GPU_QCOM_ADERNO;
  } else if (device_name.find(kMaliGPUStr) != std::string::npos) {
    return GPUType::GPU_ARM_MALI;
  } else {
    return GPUType::GPU_OTHER;
  }
}



OpenCLRuntime::OpenCLRuntime() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  auto &platform = platforms[0];
  platformInfo_ =
      fmt::format("OpenCL name: {}, profile: {}, version: {}, vendor: {}",
                  platform.getInfo<CL_PLATFORM_NAME>(),
                  platform.getInfo<CL_PLATFORM_PROFILE>(),
                  platform.getInfo<CL_PLATFORM_VERSION>(),
                  platform.getInfo<CL_PLATFORM_VENDOR>());

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  device_ = std::make_shared<cl::Device>();
  *device_ = devices[0];
  device_->getInfo(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                   &globalMemCacheLineSize_);
  device_->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &deviceComputeUnits_);
  printf("device extensions: %s\n", device_->getInfo<CL_DEVICE_EXTENSIONS>().c_str());
  gpuType_ = ParseGPUType(device_->getInfo<CL_DEVICE_NAME>());

  cl_context_properties *properties = nullptr;
#ifdef __aarch64__
  if (gpuType_ == GPU_ARM_MALI) {
    static cl_context_properties innerProperties[] = {
        CL_PRINTF_CALLBACK_ARM,   (cl_context_properties) printf_callback,
        CL_PRINTF_BUFFERSIZE_ARM, (cl_context_properties) 0x100000,
        CL_CONTEXT_PLATFORM,      (cl_context_properties) platform(),
        0
    };

    properties = innerProperties;
  } else if (gpuType_ == GPU_QCOM_ADERNO) {
    static cl_context_properties innerProperties[] = {
        CL_CONTEXT_PLATFORM,      (cl_context_properties) platform(), 0
    };
    properties = innerProperties;
  }
#endif
  if (gpuType_ == GPU_OTHER) {
    static cl_context_properties innerProperties[] = {
        CL_CONTEXT_PLATFORM,      (cl_context_properties) platform(),
        0
    };

    properties = innerProperties;
  }

  context_ = std::make_shared<cl::Context>(*device_, properties, nullptr, nullptr, nullptr);

  queue_ = std::make_shared<cl::CommandQueue>(*context_, *device_);
}

OpenCLRuntime::~OpenCLRuntime() {}

std::string OpenCLRuntime::GetPlatformInfo() const { return platformInfo_; }

uint64_t OpenCLRuntime::GetDeviceMaxWorkGroupSize() const {
  uint64_t size = 0;
  cl_int err = device_->getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);
  AssertInFastLLM(err == CL_SUCCESS, std::to_string(err));
  return size;
}

uint64_t OpenCLRuntime::GetDeviceMaxMemAllocSize() const {
  uint64_t size = 0;
  cl_int err = device_->getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &size);
  AssertInFastLLM(err == CL_SUCCESS, std::to_string(err));
  return size;
}

uint64_t
OpenCLRuntime::GetKernelMaxWorkGroupSize(const cl::Kernel &kernel) const {
  size_t size;
  cl_int err =
      kernel.getWorkGroupInfo(*device_, CL_KERNEL_WORK_GROUP_SIZE, &size);
  AssertInFastLLM(err == CL_SUCCESS, std::to_string(err));
  return size;
}

void OpenCLRuntime::BuildProgram(const std::string &programName,
                                 const std::string &buildOptions,
                                 cl::Program *program) {
  auto it = GetOpenCLPrograms(programName);
  AssertInFastLLM(it != nullptr, "program not found");
  std::string programSource(it->begin(), it->end());

  *program = cl::Program(*context_, programSource);
  cl_int ret = program->build(*device_, buildOptions.c_str());
  if (ret != CL_SUCCESS) {
    size_t buildLogSize;
    clGetProgramBuildInfo((cl_program)(*program)(), (cl_device_id)(*device_)(),
                          CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);

    char *log = new char[buildLogSize];
    clGetProgramBuildInfo((cl_program)(*program)(), (cl_device_id)(*device_)(),
                          CL_PROGRAM_BUILD_LOG, buildLogSize, log, NULL);

    spdlog::error("build error: {}, error code: {}", log, OpenCLErrorToString(ret));
  }
  AssertInFastLLM(ret == CL_SUCCESS, "build error");
}

void OpenCLRuntime::BuildKernel(const std::string &programName,
                                const std::vector<std::string> &buildOptions,
                                const std::string &kernelName,
                                cl::Kernel *kernel) {
  std::string concatBuildOptions = buildOptions[0];
  for (int i = 1; i < buildOptions.size(); i++) {
    concatBuildOptions += " " + buildOptions[i];
  }
  std::string key = programName + " " + concatBuildOptions;

  auto it = builtPrograms_.find(key);
  if (it != builtPrograms_.end()) {
    *kernel = cl::Kernel(it->second, kernelName.c_str());
    return;
  }

  cl::Program program;
  std::string finalBuildOptions = " " + concatBuildOptions;
  finalBuildOptions = OPENCL_BASE_BUILD_OPTIONS + finalBuildOptions;

  BuildProgram(programName, finalBuildOptions, &program);
  builtPrograms_[key] = program;

  cl_int err;
  *kernel = cl::Kernel(program, kernelName.c_str(), &err);
  AssertInFastLLM(err == CL_SUCCESS, OpenCLErrorToString(err));
}

cl::Context &OpenCLRuntime::context() const { return *context_; }

cl::CommandQueue &OpenCLRuntime::command_queue() const { return *queue_; }

cl::Device &OpenCLRuntime::device() const { return *device_; }

GPUType OpenCLRuntime::gpu_type() const { return gpuType_; }

OpenCLRuntime *OpenCLRuntime::GetGlobalOpenCLRuntime() {
  static OpenCLRuntime runtime;
  return &runtime;
}

const std::string OpenCLErrorToString(cl_int error) {
  switch (error) {
  case CL_SUCCESS:
    return "CL_SUCCESS";
  case CL_DEVICE_NOT_FOUND:
    return "CL_DEVICE_NOT_FOUND";
  case CL_DEVICE_NOT_AVAILABLE:
    return "CL_DEVICE_NOT_AVAILABLE";
  case CL_COMPILER_NOT_AVAILABLE:
    return "CL_COMPILER_NOT_AVAILABLE";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case CL_OUT_OF_RESOURCES:
    return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY:
    return "CL_OUT_OF_HOST_MEMORY";
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case CL_MEM_COPY_OVERLAP:
    return "CL_MEM_COPY_OVERLAP";
  case CL_IMAGE_FORMAT_MISMATCH:
    return "CL_IMAGE_FORMAT_MISMATCH";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case CL_BUILD_PROGRAM_FAILURE:
    return "CL_BUILD_PROGRAM_FAILURE";
  case CL_MAP_FAILURE:
    return "CL_MAP_FAILURE";
  case CL_MISALIGNED_SUB_BUFFER_OFFSET:
    return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case CL_COMPILE_PROGRAM_FAILURE:
    return "CL_COMPILE_PROGRAM_FAILURE";
  case CL_LINKER_NOT_AVAILABLE:
    return "CL_LINKER_NOT_AVAILABLE";
  case CL_LINK_PROGRAM_FAILURE:
    return "CL_LINK_PROGRAM_FAILURE";
  case CL_DEVICE_PARTITION_FAILED:
    return "CL_DEVICE_PARTITION_FAILED";
  case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
    return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
  case CL_INVALID_VALUE:
    return "CL_INVALID_VALUE";
  case CL_INVALID_DEVICE_TYPE:
    return "CL_INVALID_DEVICE_TYPE";
  case CL_INVALID_PLATFORM:
    return "CL_INVALID_PLATFORM";
  case CL_INVALID_DEVICE:
    return "CL_INVALID_DEVICE";
  case CL_INVALID_CONTEXT:
    return "CL_INVALID_CONTEXT";
  case CL_INVALID_QUEUE_PROPERTIES:
    return "CL_INVALID_QUEUE_PROPERTIES";
  case CL_INVALID_COMMAND_QUEUE:
    return "CL_INVALID_COMMAND_QUEUE";
  case CL_INVALID_HOST_PTR:
    return "CL_INVALID_HOST_PTR";
  case CL_INVALID_MEM_OBJECT:
    return "CL_INVALID_MEM_OBJECT";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case CL_INVALID_IMAGE_SIZE:
    return "CL_INVALID_IMAGE_SIZE";
  case CL_INVALID_SAMPLER:
    return "CL_INVALID_SAMPLER";
  case CL_INVALID_BINARY:
    return "CL_INVALID_BINARY";
  case CL_INVALID_BUILD_OPTIONS:
    return "CL_INVALID_BUILD_OPTIONS";
  case CL_INVALID_PROGRAM:
    return "CL_INVALID_PROGRAM";
  case CL_INVALID_PROGRAM_EXECUTABLE:
    return "CL_INVALID_PROGRAM_EXECUTABLE";
  case CL_INVALID_KERNEL_NAME:
    return "CL_INVALID_KERNEL_NAME";
  case CL_INVALID_KERNEL_DEFINITION:
    return "CL_INVALID_KERNEL_DEFINITION";
  case CL_INVALID_KERNEL:
    return "CL_INVALID_KERNEL";
  case CL_INVALID_ARG_INDEX:
    return "CL_INVALID_ARG_INDEX";
  case CL_INVALID_ARG_VALUE:
    return "CL_INVALID_ARG_VALUE";
  case CL_INVALID_ARG_SIZE:
    return "CL_INVALID_ARG_SIZE";
  case CL_INVALID_KERNEL_ARGS:
    return "CL_INVALID_KERNEL_ARGS";
  case CL_INVALID_WORK_DIMENSION:
    return "CL_INVALID_WORK_DIMENSION";
  case CL_INVALID_WORK_GROUP_SIZE:
    return "CL_INVALID_WORK_GROUP_SIZE";
  case CL_INVALID_WORK_ITEM_SIZE:
    return "CL_INVALID_WORK_ITEM_SIZE";
  case CL_INVALID_GLOBAL_OFFSET:
    return "CL_INVALID_GLOBAL_OFFSET";
  case CL_INVALID_EVENT_WAIT_LIST:
    return "CL_INVALID_EVENT_WAIT_LIST";
  case CL_INVALID_EVENT:
    return "CL_INVALID_EVENT";
  case CL_INVALID_OPERATION:
    return "CL_INVALID_OPERATION";
  case CL_INVALID_GL_OBJECT:
    return "CL_INVALID_GL_OBJECT";
  case CL_INVALID_BUFFER_SIZE:
    return "CL_INVALID_BUFFER_SIZE";
  case CL_INVALID_MIP_LEVEL:
    return "CL_INVALID_MIP_LEVEL";
  case CL_INVALID_GLOBAL_WORK_SIZE:
    return "CL_INVALID_GLOBAL_WORK_SIZE";
  case CL_INVALID_PROPERTY:
    return "CL_INVALID_PROPERTY";
  case CL_INVALID_IMAGE_DESCRIPTOR:
    return "CL_INVALID_IMAGE_DESCRIPTOR";
  case CL_INVALID_COMPILER_OPTIONS:
    return "CL_INVALID_COMPILER_OPTIONS";
  case CL_INVALID_LINKER_OPTIONS:
    return "CL_INVALID_LINKER_OPTIONS";
  case CL_INVALID_DEVICE_PARTITION_COUNT:
    return "CL_INVALID_DEVICE_PARTITION_COUNT";
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
  case CL_INVALID_PIPE_SIZE:
    return "CL_INVALID_PIPE_SIZE";
  case CL_INVALID_DEVICE_QUEUE:
    return "CL_INVALID_DEVICE_QUEUE";
#endif
  default:
    return spdlog::fmt_lib::format("UNKNOWN: {}", error);
  }
}

} // namespace fastllm
