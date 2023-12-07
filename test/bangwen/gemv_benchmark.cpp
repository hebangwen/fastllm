#include "basellm.h"

#ifdef USE_CUDA
#include "cudadevice.h"
#include "fastllm-cuda.cuh"
#endif

#include "executor.h"
#include "fastllm.h"
#include "model.h"
#include "spdlog/common.h"
#include "utils.h"
#include <iostream>
#include <memory>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

// #define USE_DISCRETE_CUDA

struct BenchmarkConfig {
  int n_;
  int m_;
  int k_;
  fastllm::DataType weightType_;
  fastllm::DataType biasType_;
  bool hasBias_;
  float partRatio_ = 1.0f;
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
  fastllm::Data middle;

#ifdef USE_CUDA
#ifndef USE_DISCRETE_CUDA
  // discrete_cuda 不使用统一内存，但是速度较快
  input.dataDevice = fastllm::CUDA;
  input.directMemory = true;
#endif

  input.Allocate();
  input.RandomizeData();
#ifdef USE_DISCRETE_CUDA
  input.ToDevice(fastllm::CUDA);
#endif
#endif

  fastllm::TimeRecord recorder;
  for (int i = 0; i < runCfg.benchmark_rounds; i++) {
    fastllm::Data weight{benchmarkCfg.weightType_,
                         CalculatePartResult(benchmarkCfg.k_, benchmarkCfg.m_,
                                             benchmarkCfg.partRatio_, PART_K)};
    fastllm::Data bias{benchmarkCfg.biasType_};
#ifdef USE_CUDA
#ifndef USE_DISCRETE_CUDA
    weight.dataDevice = fastllm::CUDA;
    weight.directMemory = true;
#endif
#endif

    weight.Allocate();
    weight.RandomizeData();
    if (benchmarkCfg.hasBias_) {
      bias.Resize({benchmarkCfg.k_});
      bias.Allocate();
      bias.RandomizeData();
    }
#ifdef USE_CUDA
#ifdef USE_DISCRETE_CUDA
    weight.ToDevice(fastllm::CUDA);
    bias.ToDevice(fastllm::CUDA);
#endif

    // 先分配好 extradata，避免在 kernel 前要分配数据
    FastllmCreateInt4ExtraData(weight, weight.dims[0], bias);
#endif

    recorder.Record();
    fastllm::Linear(input, weight, bias, middle);
    recorder.Record(spdlog::fmt_lib::format("Linear {:02d}", i));
  }

  return recorder.StatAvg();
}

int main(int argc, char **argv) {
  RunConfig config;
  fastllm::GenerationConfig generationConfig;
  ParseArgs(argc, argv, config, generationConfig);

  // spdlog::set_level(spdlog::level::debug);

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
  csvResult.push_back("k,m,weighttype,biastype,hasbias,ratio,time");

  for (auto &cfg : benchmarkConfigs) {
    for (int i = 0; i < ratioParts; i++) {
      cfg.partRatio_ = float(i + 1) / float(ratioParts);
      auto dims = CalculatePartResult(cfg.k_, cfg.m_, cfg.partRatio_, PART_K);
      float timeMs = BenchmarkByPartRatio(config, cfg);

      csvResult.push_back(spdlog::fmt_lib::format(
          "{},{},{},{},{},{},{}", dims[0], dims[1], cfg.weightType_,
          cfg.biasType_, cfg.hasBias_, cfg.partRatio_, timeMs));

      spdlog::debug(csvResult.back());
    }
  }

  spdlog::info("result: {}", csvResult);

  return 0;
}

