#include "basellm.h"
#include "mace/core/runtime/opencl/opencl_allocator.h"
#include "mace/core/runtime/opencl/opencl_util.h"
#include "mace/ops/common/activation_type.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/opencl/buffer/conv_2d.h"
#include "mace/proto/mace.pb.h"
#include "spdlog/fmt/bundled/format.h"

#ifdef USE_CUDA
#include "cudadevice.h"
#endif

#include "executor.h"
#include "fastllm.h"
#include "mace_arena.h"
#include "model.h"
#include "spdlog/common.h"
#include "utils.h"
#include <iostream>
#include <memory>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

struct BenchmarkConfig {
  int n_;
  int m_;
  int k_;
  fastllm::DataType weightType_;
  fastllm::DataType biasType_;
  bool hasBias_;
  float partRatio_ = 1.0f;
  enum Device {
    BENCH_FASTLLM_CPU,
    BENCH_FASTLLM_CUDA,
    BENCH_MACE_OPENCL
  } device_;
};

enum PartDim { PART_M = 0, PART_K = 1 };

struct RunConfig {
  std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
  int threads = 4;                          // 使用的线程数
  bool lowMemMode = false;                  // 是否使用低内存模式
  int benchmark_rounds = 10;                // benchmark 运行次数
};

void Usage() {
  std::cout << "Usage:" << std::endl;
  std::cout << "[-h|--help]:                  显示帮助" << std::endl;
  std::cout << "<-t|--threads> <args>:        使用的线程数量" << std::endl;
  std::cout << "<-l|--low>:                   使用低内存模式" << std::endl;
  // std::cout << "<--pratio> <args>:            CPU 上的计算比例" << std::endl;
}

void ParseArgs(int argc, char **argv, RunConfig &config,
               fastllm::GenerationConfig &generationConfig) {
  std::vector<std::string> sargv;
  for (int i = 0; i < argc; i++) {
    sargv.push_back(std::string(argv[i]));
  }
  for (int i = 1; i < argc; i++) {
    if (sargv[i] == "-h" || sargv[i] == "--help") {
      Usage();
      exit(0);
    } else if (sargv[i] == "-t" || sargv[i] == "--threads") {
      config.threads = atoi(sargv[++i].c_str());
    } else if (sargv[i] == "-l" || sargv[i] == "--low") {
      config.lowMemMode = true;
    } else {
      Usage();
      exit(-1);
    }
  }
}

// int CalculatePartResult(int k, float ratio) { return int(k * ratio); }

std::vector<int> CalculatePartResult(int k, int m, float ratio, PartDim dim) {
  if (dim == PART_K) {
    return {int(k * ratio), m};
  } else if (dim == PART_M) {
    return {k, int(ratio * m)};
  }

  return {};
}

/*
input: [n, m], weight: [k, m], output: [n, k]
*/
float BenchmarkByPartRatio(const RunConfig &runCfg,
                           const BenchmarkConfig &benchmarkCfg) {
  fastllm::Data input{fastllm::FLOAT32, {benchmarkCfg.n_, benchmarkCfg.m_}};
  input.Allocate(0.5f);
  fastllm::Data middle;

  if (benchmarkCfg.device_ == BenchmarkConfig::BENCH_FASTLLM_CUDA) {
#ifdef USE_CUDA
    input.ToDevice(fastllm::CUDA);
#else
    return 0.0f;
#endif
  }

  fastllm::TimeRecord recorder;
  for (int i = 0; i < runCfg.benchmark_rounds; i++) {
    fastllm::Data weight{benchmarkCfg.weightType_,
                         CalculatePartResult(benchmarkCfg.k_, benchmarkCfg.m_,
                                             benchmarkCfg.partRatio_, PART_K)};
    weight.Allocate();
    weight.RandomizeData();
    fastllm::Data bias{benchmarkCfg.biasType_};
    if (benchmarkCfg.hasBias_) {
      bias.Resize({weight.dims[0]});
      bias.Allocate();
      bias.RandomizeData();
    }

    if (benchmarkCfg.device_ != BenchmarkConfig::BENCH_MACE_OPENCL) {
#ifdef USE_CUDA
      weight.ToDevice(fastllm::CUDA);
      bias.ToDevice(fastllm::CUDA);
#else
      return 0.0f;
#endif
      recorder.Record();
      fastllm::Linear(input, weight, bias, middle);
      recorder.Record(spdlog::fmt_lib::format("Linear {:02d}", i));
    } else {
      static fastllm::MaceArena arena{
          4, mace::AFFINITY_BIG_ONLY,
          fastllm::MaceArena::ComputeHint::MACE_OPENCL};

      static mace::ops::opencl::buffer::BufferTransform transform;
      static mace::ops::opencl::buffer::Conv2dKernel conv2d;
      static int strides[2] = {1, 1};
      static std::vector<int> paddings = {1, 1};
      static int dilation[2] = {1, 1};

      mace::OpenCLAllocator clAllocator{
          arena.GetGPUDevice()->gpu_runtime()->opencl_runtime()};
      auto *ws = arena.GetWorkspace();
      auto *opContext = arena.GetOpContext();
      auto *maceInput =
          ws->CreateTensor("inp_cl_buffer", &clAllocator, mace::DT_FLOAT);
      auto *inputTrans =
          ws->CreateTensor("inp_cl_buffer_trans", &clAllocator, mace::DT_FLOAT);
      auto *filter =
          ws->CreateTensor("filter_cl_buffer", &clAllocator, mace::DT_FLOAT);
      auto *maceBias =
          ws->CreateTensor("bias_cl_buffer", &clAllocator, mace::DT_FLOAT);
      auto *output =
          ws->CreateTensor("out_cl_buffer", &clAllocator, mace::DT_FLOAT);

      fastllm::Data weightFloat;
      fastllm::ConvertInt4NoZeroToFloat(weight, weightFloat);
      maceInput->Resize({1, 1, benchmarkCfg.n_, benchmarkCfg.m_});
      maceInput->CopyBytes(input.cpuData, input.GetBytes());
      filter->Resize({weight.dims[0], weight.dims[1], 1, 1});
      filter->CopyBytes(weightFloat.cpuData, weightFloat.GetBytes());
      if (benchmarkCfg.hasBias_) {
        maceBias->Resize({weight.dims[0]});
        maceBias->CopyBytes(bias.cpuData, bias.GetBytes());
      } else {
        maceBias = nullptr;
      }

      transform.Compute(opContext, maceInput, mace::IN_OUT_CHANNEL, 0,
                        inputTrans);
      conv2d.Compute(opContext, inputTrans, filter, maceBias, strides,
                     mace::VALID, paddings, dilation, mace::ops::NOOP, 0, 0, 0,
                     output);
      recorder.Record();
      transform.Compute(opContext, maceInput, mace::IN_OUT_CHANNEL, 0,
                        inputTrans);
      conv2d.Compute(opContext, inputTrans, filter, maceBias, strides,
                     mace::VALID, paddings, dilation, mace::ops::NOOP, 0, 0, 0,
                     output);
      recorder.Record(spdlog::fmt_lib::format("Linear {:02d}", i));
    }
  }

  return recorder.StatAvg();
}

int main(int argc, char **argv) {
  RunConfig config;
  fastllm::GenerationConfig generationConfig;
  ParseArgs(argc, argv, config, generationConfig);

  spdlog::set_level(spdlog::level::debug);

  fastllm::PrintInstructionInfo();
  fastllm::SetThreads(config.threads);
  fastllm::SetLowMemMode(config.lowMemMode);

  std::vector<BenchmarkConfig> benchmarkConfigs = {
      {1, 4096, 4608, fastllm::INT4_NOZERO, fastllm::FLOAT32, true},
      {1, 4096, 4096, fastllm::INT4_NOZERO, fastllm::FLOAT32, false},
      {1, 27392, 4096, fastllm::INT4_NOZERO, fastllm::FLOAT32, false},
      {1, 4096, 13696, fastllm::INT4_NOZERO, fastllm::FLOAT32, false},
  };

  int ratioParts = 10;
  std::vector<std::string> csvResult;
  csvResult.push_back("k,m,weighttype,biastype,hasbias,device,ratio,time");

  std::vector<BenchmarkConfig::Device> devices = {
      BenchmarkConfig::BENCH_FASTLLM_CPU, BenchmarkConfig::BENCH_MACE_OPENCL};
  std::vector<std::string> deviceIdToName = {"CPU", "CUDA", "OPENCL"};

  for (auto &device : devices) {
    for (auto &cfg : benchmarkConfigs) {
      cfg.device_ = device;

      for (int i = 0; i < ratioParts; i++) {
        cfg.partRatio_ = float(i + 1) / float(ratioParts);
        if (device != BenchmarkConfig::BENCH_FASTLLM_CPU) {
          cfg.partRatio_ = 1 - cfg.partRatio_ + 0.1;
        }
        auto dims = CalculatePartResult(cfg.k_, cfg.m_, cfg.partRatio_, PART_K);
        float timeMs = BenchmarkByPartRatio(config, cfg);

        csvResult.push_back(spdlog::fmt_lib::format(
            "{},{},{},{},{},{},{},{}", dims[0], dims[1], cfg.weightType_,
            cfg.biasType_, cfg.hasBias_, deviceIdToName[cfg.device_], cfg.partRatio_, timeMs));

        spdlog::debug(csvResult.back());
      }
    }
  }

  spdlog::info("result: {}", csvResult);

  return 0;
}
