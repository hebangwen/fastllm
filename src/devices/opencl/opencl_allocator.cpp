// Copyright 2018 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <CL/cl.h>
#include <memory>

#include "CL/opencl.hpp"
#include "devices/opencl/opencl_allocator.h"
#include "devices/opencl/opencl_runtime.h"
#include "spdlog/fmt/bundled/core.h"
#include "utils.h"
#include <spdlog/spdlog.h>

namespace fastllm {

OpenCLAllocator::OpenCLAllocator()
    : opencl_runtime_(OpenCLRuntime::GetGlobalOpenCLRuntime()) {}

OpenCLAllocator::~OpenCLAllocator() {}

void OpenCLAllocator::New(size_t nbytes, void **result) {
  if (nbytes == 0) {
    return;
  }

  cl_int error = CL_SUCCESS;
  cl::Buffer *buffer = nullptr;

  buffer = new cl::Buffer(opencl_runtime_->context(),
                          CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, nbytes,
                          nullptr, &error);
  if (error != CL_SUCCESS) {
    delete buffer;
    *result = nullptr;
    AssertInFastLLM(
        error == CL_SUCCESS,
        spdlog::fmt_lib::format("opencl allocation error, error: {}",
                                OpenCLErrorToString(error)));
  } else {
    *result = buffer;
  }
}

void OpenCLAllocator::Delete(void *buffer) {
  if (buffer != nullptr) {
    cl::Buffer *cl_buffer = static_cast<cl::Buffer *>(buffer);
    delete cl_buffer;
  }
}

void *OpenCLAllocator::Map(void *buffer, size_t offset, size_t nbytes,
                           bool finish_cmd_queue) const {
  void *mapped_ptr = nullptr;
  auto cl_buffer = static_cast<cl::Buffer *>(buffer);
  auto &queue = opencl_runtime_->command_queue();
  // TODO(heliangliang) Non-blocking call
  cl_int error;
  mapped_ptr =
      queue.enqueueMapBuffer(*cl_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                             offset, nbytes, nullptr, nullptr, &error);
  AssertInFastLLM(error == CL_SUCCESS,
                  spdlog::fmt_lib::format("Map buffer failed, error: {} {}",
                                          error, OpenCLErrorToString(error)));
  return mapped_ptr;
}

void OpenCLAllocator::Unmap(void *buffer, void *mapped_ptr) const {

  auto cl_buffer = static_cast<cl::Buffer *>(buffer);
  auto &queue = opencl_runtime_->command_queue();
  cl_int error =
      queue.enqueueUnmapMemObject(*cl_buffer, mapped_ptr, nullptr, nullptr);
  AssertInFastLLM(error == CL_SUCCESS,
                  spdlog::fmt_lib::format("Unmap buffer failed, error: {} {}",
                                          error, OpenCLErrorToString(error)));
}

bool OpenCLAllocator::OnHost() const { return false; }

OpenCLAllocator *OpenCLAllocator::GetGlobalOpenCLAllocator() {
  static OpenCLAllocator allocator;
  return &allocator;
}

} // namespace fastllm
