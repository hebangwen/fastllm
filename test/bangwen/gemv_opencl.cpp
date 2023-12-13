
#include "mace/core/device_context.h"
#include "mace/core/registry/op_delegator_registry.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/core/runtime/opencl/gpu_device.h"
#include "mace/core/runtime/opencl/opencl_allocator.h"
#include "mace/core/runtime/opencl/opencl_util.h"
#include "mace/core/tensor.h"
#include "mace/core/types.h"
#include "mace/ops/common/activation_type.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/opencl/buffer/buffer_transform.h"
#include "mace/proto/mace.pb.h"
#include "mace/public/mace.h"
#include "mace/utils/thread_pool.h"
#include "spdlog/fmt/bundled/core.h"
#include "spdlog/fmt/bundled/format.h"
#include "utils.h"

#include "mace/core/ops/op_context.h"
#include "mace/core/workspace.h"
#include "mace/ops/opencl/buffer/conv_2d.h"
#include "mace/ops/opencl/image/matmul.h"
#include "spdlog/common.h"
#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <cstddef>
#include <cstdint>
#include <fastllm.h>
#include <fstream>
#include <mutex>
#include <numeric>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <vector>

#define DEFAULT_OPENCL_STORAGE_PATH "/tmp/opencl.lib"

#define COMPILE_OPTIONS "-cl-std=CL3.0 -cl-mad-enable -cl-fast-relaxed-math"

#define FASTLLM_CHECK_CL_SUCCESS(error, info)                                  \
  if (error != CL_SUCCESS) {                                                   \
    SPDLOG_ERROR("CL ERROR CODE: {}, info: {}\n", (int)error, info);           \
  }

std::string readKernelContents(std::initializer_list<std::string> kernelFiles) {
  std::string begin;
  auto contents =
      std::accumulate(kernelFiles.begin(), kernelFiles.end(), begin,
                      [](std::string &init, const std::string &filepath) {
                        std::ifstream ifs(filepath);
                        std::istreambuf_iterator<char> beginFile(ifs), eof;
                        std::copy(beginFile, eof, std::back_inserter(init));
                        init.append("\n");
                        return init;
                      });

  return contents;
}

cl::Program buildProgram(cl::Context &context, cl::Device &targetDevice,
                         std::initializer_list<std::string> kernelFilePaths,
                         const std::string &options) {
  auto contents = readKernelContents(kernelFilePaths);
  cl::Program program(context, contents);

  auto ret = program.build(targetDevice, options.c_str());
  if (ret != CL_SUCCESS) {
    // If the build fails, print the build log
    size_t logSize;
    clGetProgramBuildInfo((cl_program)program(), (cl_device_id)targetDevice(),
                          CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

    char *log = new char[logSize];
    clGetProgramBuildInfo((cl_program)program(), (cl_device_id)targetDevice(),
                          CL_PROGRAM_BUILD_LOG, logSize, log, NULL);

    spdlog::error("build error: {}", log);
    delete[] log;
  }

  return program;
}

void ConvertInt4NoZeroToFloat(fastllm::Data &weight,
                              fastllm::Data &weightFloat) {
  weightFloat.dataType = fastllm::FLOAT32;
  weightFloat.Resize(weight.dims);
  weightFloat.Allocate();

  float *weightFloatPtr = (float *)weightFloat.cpuData;
  uint8_t *weightPtr = weight.cpuData;
  int k = weight.dims[0], m = weight.dims[1];

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < (m >> 1); j++) {
      weightFloatPtr[i * m + j * 2] =
          weight.perChannelsConfigs[i].invQuantization(
              weightPtr[i * (m >> 1) + j] >> 4);
      weightFloatPtr[i * m + j * 2 + 1] =
          weight.perChannelsConfigs[i].invQuantization(
              weightPtr[i * (m >> 1) + j] & 15);
    }
  }
}

void PrintMaceTensor(mace::Tensor *tensor) {
  mace::Tensor::MappingGuard guard(tensor);
  std::cout << "shape: " << tensor->buffer_shape() << std::endl;
  const float *data = tensor->data<float>();
  for (int i = 0; i < 10; i++) {
    printf("%f ", data[i]);
  }
  printf("\n");
}

void CopyDataFromArray(mace::Tensor *tensor, const std::vector<int64_t> &shape,
                       const std::vector<float> &data) {
  tensor->Resize(shape);
  mace::Tensor::MappingGuard guard{tensor};
  float *tensorData = tensor->mutable_data<float>();
  auto size = tensor->size();
  for (int64_t i = 0; i < size; i++) {
    tensorData[i] = data[i];
  }
}

int main() {
  spdlog::set_level(spdlog::level::debug);
  std::string kernelFilePath = "linear.cl";
  int benchmarkRounds = 20;

  int m = 4096, k = 4608;
  fastllm::Data input{fastllm::FLOAT32, {1, m}};
  input.Allocate(0.5f);
  fastllm::Data weight{fastllm::INT4_NOZERO, {k, m}};
  weight.Allocate();
  weight.RandomizeData();
  fastllm::Data bias{fastllm::FLOAT32, {k}};
  bias.Allocate();
  bias.RandomizeData();

#ifdef USE_CUDA
  input.ToDevice(fastllm::CUDA);
  weight.ToDevice(fastllm::CUDA);
  bias.ToDevice(fastllm::CUDA);

  fastllm::TimeRecord recorder;
  fastllm::Data result;
  for (int i = 0; i < benchmarkRounds; i++) {
    recorder.Record();
    fastllm::Linear(input, weight, bias, result);
    recorder.Record(spdlog::fmt_lib::format("CUDA {:02d}", i));
  }

  input.ToDevice(fastllm::CPU);
  weight.ToDevice(fastllm::CPU);
  bias.ToDevice(fastllm::CPU);
  result.ToDevice(fastllm::CPU);
#else
  fastllm::Data result{fastllm::FLOAT32, {1, 4608}};
  result.Allocate(0.5f);
#endif

  fastllm::ApplyDeviceMap({{"cpu", 10}}, 1, 10);
  fastllm::Data result1;
  for (int i = 0; i < benchmarkRounds; i++) {
    recorder.Record();
    fastllm::Linear(input, weight, bias, result1);
    recorder.Record(spdlog::fmt_lib::format("CPU {:02d}", i));
  }

  fastllm::Data result2{fastllm::FLOAT32, {1, 4096}};
  result2.Allocate();

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  auto platform = platforms[0];

  spdlog::debug("platform: {}", platform.getInfo<CL_PLATFORM_NAME>());

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  auto device = devices[0];
  // (local) workgroup 存在一个最大值, 超过就会报错
  // work items 的最大数量由 arr 的乘积决定
  spdlog::debug(
      "workgroup size: {}, work item dimension: {}, work item sizes: {}",
      device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(),
      device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>(),
      device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>());

  cl::Context context{
      device, new cl_context_properties[]{
                  CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0}};
  cl::Program program =
      buildProgram(context, device, {kernelFilePath}, COMPILE_OPTIONS);
  cl::CommandQueue queue{context, device};

  cl::Buffer bufferA{context, CL_MEM_READ_WRITE, input.GetBytes()};
  cl::Buffer bufferB{context, CL_MEM_READ_WRITE, weight.GetBytes()};
  cl::Buffer bufferC{context, CL_MEM_READ_WRITE, result2.GetBytes()};
  cl::Buffer bufferBias{context, CL_MEM_READ_WRITE, bias.GetBytes()};
  cl::Buffer bufferScales{context, CL_MEM_READ_WRITE,
                          weight.scales.size() * sizeof(float)};
  cl::Buffer bufferMins{context, CL_MEM_READ_WRITE,
                        weight.mins.size() * sizeof(float)};

  cl_int ret = CL_SUCCESS;
  ret |= queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, input.GetBytes(),
                                  input.cpuData);
  ret |= queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, weight.GetBytes(),
                                  weight.cpuData);
  ret |= queue.enqueueWriteBuffer(bufferBias, CL_TRUE, 0, bias.GetBytes(),
                                  bias.cpuData);
  ret |= queue.enqueueWriteBuffer(bufferScales, CL_TRUE, 0, m * sizeof(float),
                                  weight.scales.data());
  ret |= queue.enqueueWriteBuffer(bufferMins, CL_TRUE, 0, m * sizeof(float),
                                  weight.mins.data());

  FASTLLM_CHECK_CL_SUCCESS(ret, "enqueue write buffer");

  cl::Kernel kernel{program, "GemvFloatInt4NoZero"};
  int idx = 0;
  ret |= kernel.setArg(idx++, bufferA);
  ret |= kernel.setArg(idx++, bufferB);
  ret |= kernel.setArg(idx++, bufferC);
  ret |= kernel.setArg(idx++, bufferBias);
  ret |= kernel.setArg(idx++, bufferScales);
  ret |= kernel.setArg(idx++, bufferMins);
  ret |= kernel.setArg(idx++, m);
  ret |= kernel.setArg(idx++, k);
  FASTLLM_CHECK_CL_SUCCESS(ret, "kernel set args");

  int warpSize = 256;

  for (int i = 0; i < benchmarkRounds; i++) {
    recorder.Record();
    ret |= queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(k, warpSize),
                                      cl::NDRange(1, warpSize));
    // ret |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(k),
    //                                   cl::NullRange);
    ret |= queue.finish();
    recorder.Record(spdlog::fmt_lib::format("OpenCL {:02d}", i));
  }
  FASTLLM_CHECK_CL_SUCCESS(ret, "enqueue kernel");

  ret |= queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, result2.GetBytes(),
                                 result2.cpuData);
  FASTLLM_CHECK_CL_SUCCESS(ret, "enqueue read buffer");

  mace::ops::opencl::buffer::Conv2dKernel maceConvKernel;
  // buffer: [O, I, H, W] -> [H, W, (O+3)/4, I, 4]
  mace::ops::opencl::buffer::BufferTransform transform;

  mace::OpDelegatorRegistry delegatorRegistry;
  mace::Workspace space{&delegatorRegistry};
  auto gpuContext = mace::GPUContextBuilder{}.Finalize();

  mace::utils::ThreadPool pool{4, mace::AFFINITY_BIG_ONLY};

  mace::GPUDevice gpuDevice =
      mace::GPUDevice{gpuContext->opencl_tuner(),
                      gpuContext->opencl_cache_storage(),
                      mace::GPUPriorityHint::PRIORITY_LOW,
                      mace::PERF_NORMAL,
                      gpuContext->opencl_binary_storage(),
                      4,
                      mace::AFFINITY_BIG_ONLY,
                      &pool};

  mace::OpContext opContext{&space, &gpuDevice};

  mace::OpenCLAllocator clAllocator{gpuDevice.gpu_runtime()->opencl_runtime()};
  auto *maceInput =
      space.CreateTensor("conv_input", &clAllocator, mace::DT_FLOAT);
  auto *maceWeight =
      space.CreateTensor("conv_weight", &clAllocator, mace::DT_FLOAT, true);
  auto *maceBias =
      space.CreateTensor("conv_bias", &clAllocator, mace::DT_FLOAT, true);
  auto *maceOutput =
      space.CreateTensor("conv_output", &clAllocator, mace::DT_FLOAT);
  auto *maceWeightTrans = space.CreateTensor("conv_weight_trans", &clAllocator, mace::DT_FLOAT, true);
  std::vector<int> strides = {1, 1};
  std::vector<int> paddings = {0, 0};
  std::vector<int> dilations = {1, 1};

  fastllm::Data unquantWeight;
  ConvertInt4NoZeroToFloat(weight, unquantWeight);
  maceInput->Resize({1, 1, 1, m});
  maceInput->set_data_format(mace::DataFormat::NHWC);
  maceWeight->Resize({k, m, 1, 1});
  maceWeight->set_data_format(mace::DataFormat::OIHW);
  maceBias->Resize({k});
  maceBias->set_data_format(mace::DataFormat::AUTO);
  maceOutput->Resize({1, 1, 1, k});
  maceOutput->set_data_format(mace::DataFormat::NHWC);

  fastllm::Data out2;
  for (int i = 0; i < benchmarkRounds; i++) {
    recorder.Record();
    fastllm::Linear(input, unquantWeight, bias, out2);
    recorder.Record(spdlog::fmt_lib::format("CPUF {:02d}", i));
  }

  maceInput->CopyBytes(input.cpuData, input.GetBytes());
  maceWeight->CopyBytes(unquantWeight.cpuData, unquantWeight.GetBytes());
  maceBias->CopyBytes(bias.cpuData, bias.GetBytes());

  for (int i = 0; i < benchmarkRounds; i++) {
    recorder.Record();

    transform.Compute(&opContext, maceWeight, mace::CONV2D_FILTER, 0, maceWeightTrans);
    maceConvKernel.Compute(&opContext, maceInput, maceWeightTrans, maceBias,
                           strides.data(), mace::VALID, paddings,
                           dilations.data(), mace::ops::NOOP, 0, 0, 0,
                           maceOutput);

    recorder.Record(spdlog::fmt_lib::format("mace {:02d}", i));
  }

  {
    mace::Tensor::MappingGuard mapper{maceOutput};
    float *res0 = (float *)result.cpuData;
    float *res1 = (float *)result1.cpuData;
    float *res2 = (float *)result2.cpuData;
    const float *res3 = maceOutput->data<float>();
    float *res4 = (float *)out2.cpuData;
    for (int i = 0; i < 24; i++) {
      spdlog::info("{} {} {} {} {} {}", i, res0[i], res1[i], res2[i], res3[i],
                   res4[i]);
    }
  }

  recorder.Print();
  return 0;
}