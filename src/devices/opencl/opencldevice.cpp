#include "devices/opencl/opencldevice.h"
#include "CL/cl.h"
#include "device.h"
#include "devices/opencl/opencl_allocator.h"
#include "devices/opencl/opencl_runtime.h"
#include "fastllm.h"
#include "utils.h"
#include <CL/opencl.hpp>
#include <memory>


namespace fastllm {
void CopyBufferFromCPU(OpenCLAllocator *allocator, void *dst, void *src,
                       size_t size) {
  void *oclPtr = allocator->Map(dst, 0, size, true);
  std::memcpy(oclPtr, src, size);
  allocator->Unmap(dst, oclPtr);
}

void CopyBufferToCPU(OpenCLAllocator *allocator, void *dst, void *src,
                     size_t size) {
  void *oclPtr = allocator->Map(src, 0, size, true);
  std::memcpy(dst, oclPtr, size);
  allocator->Unmap(src, oclPtr);
}

OpenCLDevice::OpenCLDevice() {
  deviceType = "opencl";
  deviceName = "opencl";
  oclAllocator_ = OpenCLAllocator::GetGlobalOpenCLAllocator();

  this->ops["Linear"] = (BaseOperator *)new OpenCLLinearOp();
}

bool OpenCLDevice::Malloc(void **ret, size_t size) {
  oclAllocator_->New(size, ret);
  return true;
}

bool OpenCLDevice::Malloc(void **ret, Data &data) {
  return Malloc(ret, data.expansionBytes);
}

bool OpenCLDevice::Free(void *ret) {
  oclAllocator_->Delete(ret);
  return true;
}

bool OpenCLDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
  CopyBufferToCPU(oclAllocator_, dst, src, size);

  return true;
}

bool OpenCLDevice::CopyDataToCPU(Data &data) {
  AssertInFastLLM(data.cpuData == nullptr,
                  "Copy data from " + this->deviceName +
                      " to cpu failed: cpu's data is not null.\n");
  AssertInFastLLM(data.deviceData != nullptr,
                  "Copy data from " + this->deviceName +
                      " to cpu failed: device's data is null.\n");
  data.cpuData = new uint8_t[data.expansionBytes];
  CopyDataToCPU(data.cpuData, data.openclData_, data.expansionBytes);
  Free(data.openclData_);
  data.openclData_ = nullptr;
  return true;
}

bool OpenCLDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
  CopyBufferFromCPU(oclAllocator_, dst, src, size);
  return true;
}

bool OpenCLDevice::CopyDataFromCPU(Data &data) {
  AssertInFastLLM(data.openclData_ != nullptr,
                  "Copy data to " + this->deviceName +
                      " from cpu failed: cpu's data is null.\n");
  AssertInFastLLM(data.deviceData == nullptr,
                  "Copy data to " + this->deviceName +
                      " from cpu failed: device's data is not null.\n");

  Malloc(&data.openclData_, data.expansionBytes);
  CopyDataFromCPU(data.openclData_, data.cpuData, data.expansionBytes);
  delete[] data.cpuData;
  data.cpuData = nullptr;
  return true;
}

void FastllmOpenCLMatVecMulFloatInt4NoZero(cl::Kernel *kernel,
                                           fastllm::Data &input,
                                           fastllm::Data &weight,
                                           fastllm::Data &bias,
                                           fastllm::Data &output, int n, int m,
                                           int k, bool hasBias) {
  OpenCLRuntime *runtime = OpenCLRuntime::GetGlobalOpenCLRuntime();
  OpenCLAllocator *allocator = OpenCLAllocator::GetGlobalOpenCLAllocator();

  cl::Buffer *scales, *mins;
  allocator->New(k * sizeof(float), (void **)&scales);
  allocator->New(k * sizeof(float), (void **)&mins);
  CopyBufferFromCPU(allocator, scales, weight.scales.data(), k * sizeof(float));
  CopyBufferFromCPU(allocator, mins, weight.mins.data(), k * sizeof(float));

  int idx = 0;
  kernel->setArg(idx++, *(cl::Buffer *)input.openclData_);
  kernel->setArg(idx++, *(cl::Buffer *) weight.openclData_);
  kernel->setArg(idx++, *(cl::Buffer*) output.openclData_);
  if (hasBias) {
    kernel->setArg(idx++, *(cl::Buffer *) bias.openclData_);
  }
  kernel->setArg(idx++, *scales);
  kernel->setArg(idx++, *mins);
  kernel->setArg(idx++, k);
  kernel->setArg(idx++, m);

  std::vector<int> gws{k >> 2, 1};
#ifdef __aarch64__
  // arm GPU 不能使用非均匀划分的 workgroup
  std::vector<int> lws{(int) runtime->GetKernelMaxWorkGroupSize(*kernel), 1};
#else
  std::vector<int> lws{16, 4};
#endif

  cl::Event event;
  runtime->command_queue().enqueueNDRangeKernel(*kernel, cl::NullRange,
                                                cl::NDRange(gws[0], gws[1]),
                                                cl::NDRange(lws[0], lws[1]),
                                                nullptr, &event);
  event.wait();
  runtime->command_queue().finish();

  allocator->Delete(scales);
  allocator->Delete(mins);
}

OpenCLLinearOp::OpenCLLinearOp() {
  OpenCLRuntime *runtime = OpenCLRuntime::GetGlobalOpenCLRuntime();

  kernel_ = std::make_shared<cl::Kernel>();
  kernelNoBias_ = std::make_shared<cl::Kernel>();
  runtime->BuildKernel("gemv", {"-DOP=Linear", "-DHAS_BIAS"},
                       "GemvConv1x1Impl", kernel_.get());
  runtime->BuildKernel("gemv", {"-DOP=Linear"}, "GemvConv1x1Impl",
                       kernelNoBias_.get());
}

void OpenCLLinearOp::Reshape(const std::string &opType, const DataDict &datas,
                             const FloatDict &floatParams,
                             const IntDict &intParams) {
  Data &input = *(datas.find("input")->second);
  Data &output = *(datas.find("output")->second);
  Data &weight = *(datas.find("weight")->second);

  AssertInFastLLM(weight.dims.size() == 2,
                  "Linear's weight's shape's size should be 2.\n");
  AssertInFastLLM(input.dims.back() == weight.dims[1],
                  "Linear's weight's shape error.\n");

  weight.weightType = WeightType::LINEAR;
  std::vector<int> dims = input.dims;
  dims.back() = weight.dims[0];

  output.dataType = DataType::FLOAT32;
  output.dataDevice = DataDevice::OPENCL;
  output.Resize(dims);
}

bool OpenCLLinearOp::CanRun(const std::string &opType, const DataDict &datas,
                            const FloatDict &floatParams,
                            const IntDict &intParams) {
  return true;
}

void OpenCLLinearOp::Run(const std::string &opType, const DataDict &datas,
                         const FloatDict &floatParams,
                         const IntDict &intParams) {
  Data &input = *(datas.find("input")->second);
  Data &output = *(datas.find("output")->second);
  Data &weight = *(datas.find("weight")->second);
  Data &bias = *(datas.find("bias")->second);

  output.Allocate();
  int n = input.Count(0) / input.dims.back();
  int m = input.dims.back();
  int k = output.dims.back();

  if (input.dataType == DataType::FLOAT32 &&
      weight.dataType == DataType::INT4_NOZERO) {
    bool hasBias = bias.dims.size() > 0;
    cl::Kernel *kernel = hasBias ? kernel_.get() : kernelNoBias_.get();
    FastllmOpenCLMatVecMulFloatInt4NoZero(kernel, input, weight, bias, output,
                                          n, m, k, hasBias);
  } else {
    ErrorInFastLLM("OpenCLLinearOp error: data type error.\n");
  }
}

} // namespace fastllm