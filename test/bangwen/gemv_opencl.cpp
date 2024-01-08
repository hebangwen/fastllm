#include "spdlog/fmt/bundled/core.h"
#include "spdlog/fmt/bundled/format.h"
#include "utils.h"
#include <chrono>
#include <cstdint>
#include <initializer_list>
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
void PrintOutputValues(std::initializer_list<fastllm::Data> datas) {
  int n = datas.size();
  std::vector<float *> ptrs(n);
  std::vector<size_t> sizes(n);
  auto begin = datas.begin();
  for (int i = 0; i < n; i++) {
    ptrs[i] = (T *)begin->cpuData;
    sizes[i] = begin->Count(0);
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

std::vector<uint32_t> FindKernelWorkgroupSize(cl::Kernel &kernel,
                                              cl::Device &device,
                                              cl::CommandQueue &queue,
                                              const std::vector<int> &gws) {
  const uint32_t kwg_size =
      static_cast<uint32_t>(GetKernelMaxWorkGroupSize(kernel, device));
  std::vector<std::vector<uint32_t>> results;
  std::vector<std::vector<uint32_t>> candidates = {{kwg_size / 2, 2, 0},
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
    const uint32_t tmp = ele[0] * ele[1];
    if (0 < tmp && tmp <= kwg_size) {
      results.push_back(ele);
    }
  }

  std::vector<uint32_t> *bestLWS;
  long bestTime;

  int n = results.size();
  for (int i = 0; i < n; i++) {
    auto &candidate = candidates[i];
    // warmup
    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(gws[0], gws[1]),
                               cl::NDRange(candidate[0], candidate[1]));
    queue.finish();

    auto start = std::chrono::system_clock::now();
    for (int j = 0; j < 10; j++) {
      queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                 cl::NDRange(gws[0], gws[1]),
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
  spdlog::debug("{}:{}", __FILE__, __LINE__);
  spdlog::set_level(spdlog::level::debug);
  std::string kernelFilePath = "linear.cl";
  int benchmarkRounds = 10;
  if (argc > 1)
    benchmarkRounds = std::stoi(argv[1]);
  int m = 4096, k = 4608;

  spdlog::debug("{}:{}", __FILE__, __LINE__);
  fastllm::Data input{fastllm::FLOAT32, {1, m}};
  input.Allocate(0.5f);
  input.RandomizeData();
  fastllm::Data weight{fastllm::INT4_NOZERO, {k, m}};
  weight.Allocate();
  weight.RandomizeData();
  // weight.Allocate((uint8_t)1);
  fastllm::Data bias{fastllm::FLOAT32, {k}};
  bias.Allocate(0.0f);

  spdlog::debug("{}:{}", __FILE__, __LINE__);
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

  // int warpSize = 64;
  int warpSize = 256;
  fastllm::Data input2(input);
  fastllm::Data result2{fastllm::FLOAT32, {1, k}};
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
  std::string compileOptions{COMPILE_OPTIONS};
  compileOptions += fmt::format(" -DLOCAL_SIZE={}", warpSize);
  cl::Program program =
      buildProgram(context, device, {kernelFilePath}, compileOptions);
  cl::CommandQueue queue{context, device};

  cl::Buffer bufferA{context, CL_MEM_READ_WRITE, input.GetBytes()};
  cl::Buffer bufferB{context, CL_MEM_READ_WRITE, weight.GetBytes()};
  cl::Buffer bufferC{context, CL_MEM_READ_WRITE, result2.GetBytes()};
  cl::Buffer bufferBias{context, CL_MEM_READ_WRITE, bias.GetBytes()};
  cl::Buffer bufferScales{context, CL_MEM_READ_WRITE,
                          weight.scales.size() * sizeof(float)};
  cl::Buffer bufferMins{context, CL_MEM_READ_WRITE,
                        weight.mins.size() * sizeof(float)};
  cl::Buffer memAligned{context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                        input.GetBytes()};

  cl_int ret = CL_SUCCESS;
  ret |= queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, input.GetBytes(),
                                  input.cpuData);
  ret |= queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, weight.GetBytes(),
                                  weight.cpuData);
  ret |= queue.enqueueWriteBuffer(bufferBias, CL_TRUE, 0, bias.GetBytes(),
                                  bias.cpuData);
  ret |= queue.enqueueWriteBuffer(bufferScales, CL_TRUE, 0, k * sizeof(float),
                                  weight.scales.data());
  ret |= queue.enqueueWriteBuffer(bufferMins, CL_TRUE, 0, k * sizeof(float),
                                  weight.mins.data());

  float *inputMap = (float *)queue.enqueueMapBuffer(
      memAligned, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, input.GetBytes());
  // std::memcpy(inputMap, input.cpuData, input.GetBytes());
  std::memcpy(inputMap, input2.cpuData, input2.GetBytes());
  ret |= queue.enqueueUnmapMemObject(memAligned, inputMap);
  ret |= queue.finish();

  FASTLLM_CHECK_CL_SUCCESS(ret, "enqueue write buffer");
  cl::Kernel kernel{program, "GemvFloatInt4NoZero"};
  // cl::Kernel kernel{program, "GemvFloatInt4NoZero2"};
  // cl::Kernel kernel{program, "gemv_quantized"};
  // cl::Kernel kernel{program, "gemv_kernel"};

  int idx = 0;
  // ret |= kernel.setArg(idx++, bufferA);
  ret |= kernel.setArg(idx++, memAligned);
  ret |= kernel.setArg(idx++, bufferB);
  ret |= kernel.setArg(idx++, bufferC);
  ret |= kernel.setArg(idx++, bufferBias);
  ret |= kernel.setArg(idx++, bufferScales);
  ret |= kernel.setArg(idx++, bufferMins);
  ret |= kernel.setArg(idx++, m);
  ret |= kernel.setArg(idx++, k);
  FASTLLM_CHECK_CL_SUCCESS(ret, "kernel set args");

  for (int i = 0; i < benchmarkRounds; i++) {
    recorder.Record();
    ret |= queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(k, warpSize),
                                      cl::NDRange(1, warpSize));
    ret |= queue.finish();
    recorder.Record(spdlog::fmt_lib::format("OpenCL {:02d}", i));
  }
  FASTLLM_CHECK_CL_SUCCESS(ret, "enqueue kernel");

  ret |= queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, result2.GetBytes(),
                                 result2.cpuData);
  FASTLLM_CHECK_CL_SUCCESS(ret, "enqueue read buffer");

  fastllm::Data unquantWeight;
  ConvertInt4NoZeroToFloat(weight, unquantWeight);
  fastllm::Data unquantWeightOCL(unquantWeight.dataType, unquantWeight.dims);
  unquantWeightOCL.Allocate(0.0f);

  fastllm::Data gemvConvOut(fastllm::FLOAT32, {1, k});
  gemvConvOut.Allocate();
  {
    cl::Kernel gemvConvKernel{program, "GemvConv1x1Impl"};

    idx = 0;
    gemvConvKernel.setArg(idx++, bufferA);
    gemvConvKernel.setArg(idx++, bufferB);
    gemvConvKernel.setArg(idx++, bufferC);
    gemvConvKernel.setArg(idx++, bufferBias);
    gemvConvKernel.setArg(idx++, bufferScales);
    gemvConvKernel.setArg(idx++, bufferMins);
    gemvConvKernel.setArg(idx++, k);
    gemvConvKernel.setArg(idx++, m);

    std::vector<int> gws = {k >> 2, 1};
    auto lws = FindKernelWorkgroupSize(gemvConvKernel, device, queue, gws);
    spdlog::info("gemvConv kernel: {}", lws);

    for (int i = 0; i < benchmarkRounds; i++) {
      recorder.Record();
      queue.enqueueNDRangeKernel(gemvConvKernel, cl::NullRange,
                                 cl::NDRange(gws[0], gws[1]),
                                 cl::NDRange(lws[0], lws[1]));
      queue.finish();
      recorder.Record(fmt::format("GemvConv {:02d}", i));
    }

    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, gemvConvOut.GetBytes(),
                            gemvConvOut.cpuData);
  }

#ifdef USE_OPENCL
  fastllm::ApplyDeviceMap({{"opencl", 20}, {"cpu", 10}}, 0, 0);
  input.ToDevice(fastllm::DataDevice::OPENCL);
  weight.ToDevice(fastllm::DataDevice::OPENCL);
  bias.ToDevice(fastllm::DataDevice::OPENCL);

  fastllm::Data output(fastllm::FLOAT32, {1, k});
  for (int i = 0; i < benchmarkRounds; i++) {
    recorder.Record();
    fastllm::Linear(input, weight, bias, output);
    recorder.Record(spdlog::fmt_lib::format("OpenCLIntgrated {:02d}", i));
  }

  output.ToDevice(fastllm::DataDevice::CPU);
#else
  fastllm::Data output{fastllm::FLOAT32, {1, k}};
  output.Allocate(0.5f);
#endif

  // convOutput.Print();
  PrintOutputValues<float>({result, result1, result2, gemvConvOut, output});
  spdlog::debug("mins: {}, scales: {}", weight.mins[0], weight.scales[0]);

  recorder.Print();

  // 这个 kernel 与 tid 有关, 所以不能修改 tid, 也就不能用 tune
  // FindKernelWorkgroupSize(kernel, device, queue, {k, warpSize});

  return 0;
}