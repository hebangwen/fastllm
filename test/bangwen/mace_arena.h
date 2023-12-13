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

namespace fastllm {
struct MaceArena {
  enum ComputeHint { MACE_CPU, MACE_OPENCL };

  MaceArena(int threadCount, mace::CPUAffinityPolicy policy, ComputeHint hint)
      : threadCount_(threadCount), cpuAffinityPolicy_(policy),
        computeHint_(hint) {
    Init();
  }

  mace::OpContext *GetOpContext() { return opContext_.get(); }

  mace::CPUDevice *GetCPUDevice() { return cpuDevice_.get(); }

  mace::GPUDevice *GetGPUDevice() { return gpuDevice_.get(); }

  mace::Workspace *GetWorkspace() { return space_.get(); }

private:
  inline void Init() {
    mace::OpDelegatorRegistry registry;
    space_ = std::make_shared<mace::Workspace>(&registry);
    pool_ = std::make_shared<mace::utils::ThreadPool>(threadCount_,
                                                      cpuAffinityPolicy_);

    gpuContext_ = mace::GPUContextBuilder{}.Finalize();

    cpuDevice_ = std::make_shared<mace::CPUDevice>(
        threadCount_, cpuAffinityPolicy_, pool_.get());
    gpuDevice_ = std::make_shared<mace::GPUDevice>(
        gpuContext_->opencl_tuner(), gpuContext_->opencl_cache_storage(),
        mace::GPUPriorityHint::PRIORITY_NORMAL, mace::GPUPerfHint::PERF_NORMAL,
        gpuContext_->opencl_binary_storage(), threadCount_, cpuAffinityPolicy_,
        pool_.get());

    if (computeHint_ == MACE_CPU) {
      opContext_ =
          std::make_shared<mace::OpContext>(space_.get(), cpuDevice_.get());
    } else if (computeHint_ == MACE_OPENCL) {
      opContext_ =
          std::make_shared<mace::OpContext>(space_.get(), gpuDevice_.get());
    }
  }

  int threadCount_;
  mace::CPUAffinityPolicy cpuAffinityPolicy_;
  MaceArena::ComputeHint computeHint_;
  std::shared_ptr<mace::utils::ThreadPool> pool_;

  std::shared_ptr<mace::GPUDevice> gpuDevice_;
  std::shared_ptr<mace::CPUDevice> cpuDevice_;

  std::shared_ptr<mace::GPUContext> gpuContext_;
  std::shared_ptr<mace::OpContext> opContext_;

  std::shared_ptr<mace::Workspace> space_;
};
} // namespace fastllm
