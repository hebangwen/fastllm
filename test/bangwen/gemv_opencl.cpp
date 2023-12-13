
#include "mace/core/device.h"
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
#include <memory>
#include <mutex>
#include <numeric>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <vector>
#include "mace_arena.h"

#define DEFAULT_OPENCL_STORAGE_PATH "/tmp/opencl.lib"

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

  mace::ops::opencl::buffer::Conv2dKernel maceConvKernel;
  // buffer: [O, I, H, W] -> [H, W, (O+3)/4, I, 4]
  mace::ops::opencl::buffer::BufferTransform transform;

  fastllm::MaceArena arena{4, mace::AFFINITY_BIG_ONLY,
                           fastllm::MaceArena::MACE_OPENCL};
  auto *opContext = arena.GetOpContext();
  auto *gpuDevice = arena.GetGPUDevice();
  auto *space = arena.GetWorkspace();

  mace::OpenCLAllocator clAllocator{gpuDevice->gpu_runtime()->opencl_runtime()};
  auto *maceInput =
      space->CreateTensor("conv_input", &clAllocator, mace::DT_FLOAT);
  auto *maceWeight =
      space->CreateTensor("conv_weight", &clAllocator, mace::DT_FLOAT, true);
  auto *maceBias =
      space->CreateTensor("conv_bias", &clAllocator, mace::DT_FLOAT, true);
  auto *maceOutput =
      space->CreateTensor("conv_output", &clAllocator, mace::DT_FLOAT);
  auto *maceWeightTrans = space->CreateTensor("conv_weight_trans", &clAllocator,
                                              mace::DT_FLOAT, true);
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

    transform.Compute(opContext, maceWeight, mace::CONV2D_FILTER, 0,
                      maceWeightTrans);
    maceConvKernel.Compute(opContext, maceInput, maceWeightTrans, maceBias,
                           strides.data(), mace::VALID, paddings,
                           dilations.data(), mace::ops::NOOP, 0, 0, 0,
                           maceOutput);

    recorder.Record(spdlog::fmt_lib::format("mace {:02d}", i));
  }

  {
    mace::Tensor::MappingGuard mapper{maceOutput};
    float *res0 = (float *)result.cpuData;
    float *res1 = (float *)result1.cpuData;
    const float *res3 = maceOutput->data<float>();
    float *res4 = (float *)out2.cpuData;
    for (int i = 0; i < 24; i++) {
      spdlog::info("{} {} {} {} {} ", i, res0[i], res1[i], res3[i], res4[i]);
    }
  }

  recorder.Print();
  return 0;
}