#include "spdlog/fmt/bundled/core.h"
#include "spdlog/fmt/bundled/format.h"
#include "utils.h"
#define CL_TARGET_OPENCL_VERSION 300
#define CL_HPP_TARGET_OPENCL_VERSION 300

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

int main() {
  spdlog::set_level(spdlog::level::debug);
  std::string kernelFilePath = "linear.cl";
  int benchmarkRounds = 20;

  fastllm::Data input{fastllm::FLOAT32, {1, 4096}};
  input.Allocate(0.5f);
  fastllm::Data weight{fastllm::INT4_NOZERO, {4608, 4096}};
  weight.Allocate();
  weight.RandomizeData();
  fastllm::Data bias{fastllm::FLOAT32, {4608}};
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
  int m = weight.dims[1], k = weight.dims[0];

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

  float *res0 = (float *)result.cpuData;
  float *res1 = (float *)result1.cpuData;
  float *res2 = (float *)result2.cpuData;
  for (int i = 0; i < 10; i++) {
    spdlog::info("{} {} {} {}", i, res0[i], res1[i], res2[i]);
  }

  recorder.Print();
  return 0;
}