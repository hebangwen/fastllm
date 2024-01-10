#include "devices/opencl/opencl_allocator.h"
#include "devices/opencl/opencl_runtime.h"
#include "devices/opencl/opencldevice.h"
#include "spdlog/fmt/bundled/core.h"
#include "spdlog/fmt/bundled/format.h"
#include "utils.h"
#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>
// #define CL_TARGET_OPENCL_VERSION 300
// #define CL_HPP_TARGET_OPENCL_VERSION 300

#include "spdlog/common.h"
#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <fastllm.h>
#include <fstream>
#include <mutex>
#include <numeric>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

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

template <typename T>
void PrintOutputValues(std::initializer_list<fastllm::Data *> datas) {
  int n = datas.size();
  std::vector<float *> ptrs(n);
  std::vector<size_t> sizes(n);
  auto begin = datas.begin();
  for (int i = 0; i < n; i++) {
    ptrs[i] = (T *)(*begin)->cpuData;
    sizes[i] = (*begin)->Count(0);
    begin++;
  }

  for (int i = 0; i < 10; i++) {
    printf("%d: ", i);
    for (int j = 0; j < n; j++) {
      printf("%f%c", ptrs[j][i], " \n"[j == n - 1]);
    }
  }
}

uint64_t GetKernelMaxWorkGroupSize(cl::Kernel &kernel, cl::Device &device) {
  uint64_t size = 0;
  auto ret = kernel.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &size);
  FASTLLM_CHECK_CL_SUCCESS(ret, "get kernel workgroup info error");
  return size;
}

std::vector<int> FindKernelWorkgroupSize(cl::Kernel &kernel,
                                              cl::Device &device,
                                              cl::CommandQueue &queue,
                                              const std::vector<int> &gws) {
  const int kwg_size =
      static_cast<int>(GetKernelMaxWorkGroupSize(kernel, device));
  std::vector<std::vector<int>> results;
  std::vector<std::vector<int>> candidates = {{kwg_size / 2, 2, 0},
                                                   {kwg_size / 4, 4, 0},
                                                   {kwg_size / 8, 8, 0},
                                                   {kwg_size / 16, 16, 0},
                                                   {kwg_size / 32, 32, 0},
                                                   {kwg_size / 64, 64, 0},
                                                   {kwg_size / 128, 128, 0},
                                                   {kwg_size / 256, 256, 0},
                                                   {kwg_size, 1, 0},
                                                   {1, kwg_size, 0},
                                                   {4, 4, 0},
                                                   {16, 4, 0},
                                                   {4, 16, 0},
                                                   {16, 16, 0}};
  for (auto &ele : candidates) {
    const int tmp = ele[0] * ele[1];
    if (0 < tmp && tmp <= kwg_size) {
      results.push_back(ele);
    }
  }

  std::vector<int> *bestLWS;
  long bestTime;

  auto internalGws = gws;
  int n = results.size();
  for (int i = 0; i < n; i++) {
    auto &candidate = candidates[i];
    // warmup
    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(gws[0], gws[1]),
                               cl::NDRange(candidate[0], candidate[1]));
    queue.finish();

    internalGws[1] = RoundUp(internalGws[1], candidate[1]);
    auto start = std::chrono::system_clock::now();
    for (int j = 0; j < 10; j++) {
      queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                 cl::NDRange(internalGws[0], internalGws[1]),
                                 cl::NDRange(candidate[0], candidate[1]));
      queue.finish();
    }
    auto end = std::chrono::system_clock::now();
    auto span =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // spdlog::info("local_workgroup_size {}, time: {}ms", candidate,
    // span.count() / 1000.0 / 10.0);

    if (i == 0 || span.count() < bestTime) {
      bestLWS = &candidate;
      bestTime = span.count();
    }
  }

  return {bestLWS->begin(), bestLWS->end()};
}

int main(int argc, char *argv[]) {
  spdlog::set_level(spdlog::level::debug);
  int benchmarkRounds = 10;
  if (argc > 1)
    benchmarkRounds = std::stoi(argv[1]);
  int m = 4096, k = 4608;

  fastllm::SetThreads(4);
  fastllm::PrintInstructionInfo();
  fastllm::Data input{fastllm::FLOAT32, {1, m}};
  input.Allocate(0.5f);
  // input.RandomizeData();
  fastllm::Data weight{fastllm::INT4_NOZERO, {k, m}};
  weight.Allocate();
  weight.RandomizeData();
  weight.Allocate((uint8_t)1);
  fastllm::Data bias{fastllm::FLOAT32, {k}};
  bias.Allocate(0.0f);

  fastllm::TimeRecord recorder;
#ifdef USE_CUDA
  input.ToDevice(fastllm::CUDA);
  weight.ToDevice(fastllm::CUDA);
  bias.ToDevice(fastllm::CUDA);

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

  int warpSize = 256;

  fastllm::OpenCLRuntime *runtime =
      fastllm::OpenCLRuntime::GetGlobalOpenCLRuntime();
  fastllm::OpenCLAllocator *allocator =
      fastllm::OpenCLAllocator::GetGlobalOpenCLAllocator();
  auto &context = runtime->context();
  auto &device = runtime->device();
  spdlog::info("OpenCL platform: {}", runtime->GetPlatformInfo());

  std::string compileOptions{COMPILE_OPTIONS};

  auto &queue = runtime->command_queue();
  cl::Buffer *bufferInput;
  allocator->New(input.GetBytes(), (void **)&bufferInput);
  CopyBufferFromCPU(allocator, bufferInput, input.cpuData, input.GetBytes());

  cl::Buffer *bufferWeightInt4;
  allocator->New(weight.GetBytes(), (void **)&bufferWeightInt4);
  CopyBufferFromCPU(allocator, bufferWeightInt4, weight.cpuData,
                    weight.GetBytes());

  cl::Buffer *bufferOutput;
  allocator->New(result.GetBytes(), (void **)&bufferOutput);
  CopyBufferFromCPU(allocator, bufferOutput, result.cpuData, result.GetBytes());

  cl::Buffer *bufferBias;
  allocator->New(bias.GetBytes(), (void **)&bufferBias);
  CopyBufferFromCPU(allocator, bufferBias, bias.cpuData, bias.GetBytes());

  cl::Buffer *bufferScales;
  allocator->New(k * sizeof(float), (void **)&bufferScales);
  CopyBufferFromCPU(allocator, bufferScales, weight.scales.data(),
                    k * sizeof(float));

  cl::Buffer *bufferMins;
  allocator->New(k * sizeof(float), (void **)&bufferMins);
  CopyBufferFromCPU(allocator, bufferMins, weight.mins.data(),
                    k * sizeof(float));

  int idx = 0;
  fastllm::Data gemvConvOut(fastllm::FLOAT32, {1, k});
  gemvConvOut.Allocate(0.5f);
  cl::Kernel gemvConvKernel;
  runtime->BuildKernel("gemv", {"-DOP=Linear", "-DHAS_BIAS"},
                        "GemvConv1x1Impl", &gemvConvKernel);

  std::vector<int> gws = {k >> 2, 1};
  idx = 0;
  gemvConvKernel.setArg(idx++, gws[0]);
  gemvConvKernel.setArg(idx++, gws[1]);
  gemvConvKernel.setArg(idx++, *bufferInput);
  gemvConvKernel.setArg(idx++, *bufferWeightInt4);
  gemvConvKernel.setArg(idx++, *bufferOutput);
  gemvConvKernel.setArg(idx++, *bufferBias);
  gemvConvKernel.setArg(idx++, *bufferScales);
  gemvConvKernel.setArg(idx++, *bufferMins);
  gemvConvKernel.setArg(idx++, k);
  gemvConvKernel.setArg(idx++, m);

  auto lws = FindKernelWorkgroupSize(gemvConvKernel, device, queue, gws);
  gws[1] = RoundUp(gws[1], lws[1]);
  spdlog::info("gemvConv kernel: {}", lws);

  for (int i = 0; i < benchmarkRounds; i++) {
    recorder.Record();
    queue.enqueueNDRangeKernel(gemvConvKernel, cl::NullRange,
                                cl::NDRange(gws[0], gws[1]),
                                cl::NDRange(lws[0], lws[1]));
    queue.finish();
    recorder.Record(fmt::format("GemvConv {:02d}", i));
  }

  CopyBufferToCPU(allocator, gemvConvOut.cpuData, bufferOutput, gemvConvOut.GetBytes());

#ifdef USE_OPENCL
  fastllm::ApplyDeviceMap({{"opencl", 10}}, 0, 0);
  input.ToDevice(fastllm::DataDevice::OPENCL);
  weight.ToDevice(fastllm::DataDevice::OPENCL);
  bias.ToDevice(fastllm::DataDevice::OPENCL);

  fastllm::Data output(fastllm::FLOAT32, {1, k});

  for (int i = 0; i < benchmarkRounds; i++) {
    recorder.Record();
    fastllm::Linear(input, weight, bias, output);
    recorder.Record(spdlog::fmt_lib::format("OpenCLIntgrated {:02d}", i));
  }

  runtime->command_queue().finish();
  output.ToDevice(fastllm::DataDevice::CPU);
#else
  fastllm::Data output{fastllm::FLOAT32, {1, k}};
  output.Allocate(0.5f);
#endif
  
  output.Print();
  PrintOutputValues<float>({&result, &result1, &gemvConvOut, &output});
  spdlog::debug("mins: {}, scales: {}", weight.mins[0], weight.scales[0]);

  recorder.Print();

  // 这个 kernel 与 tid 有关, 所以不能修改 tid, 也就不能用 tune
  // FindKernelWorkgroupSize(kernel, device, queue, {k, warpSize});

  return 0;
}