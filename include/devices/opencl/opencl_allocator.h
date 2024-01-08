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

#ifndef FASTLLM_OPENCL_ALLOCATOR_H
#define FASTLLM_OPENCL_ALLOCATOR_H

#include <memory>
#include <unordered_map>
#include <vector>
#include "opencl_runtime.h"

namespace fastllm {

class OpenCLAllocator {
public:
  static OpenCLAllocator *GetGlobalOpenCLAllocator();

  ~OpenCLAllocator();

  void New(size_t nbytes, void **result);

  void Delete(void *buffer);

  void *Map(void *buffer, size_t offset, size_t nbytes,
            bool finish_cmd_queue) const;

  void Unmap(void *buffer, void *mapped_ptr) const;

  bool OnHost() const;

private:
  explicit OpenCLAllocator();

  OpenCLRuntime *opencl_runtime_;
};

} // namespace fastllm

#endif // FASTLLM_OPENCL_ALLOCATOR_H
