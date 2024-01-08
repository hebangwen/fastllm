
#ifndef FASTLLM_OPENCLDEVICE_H
#define FASTLLM_OPENCLDEVICE_H

#include "device.h"
#include "devices/opencl/opencl_allocator.h"
#include <memory>

namespace fastllm {
    class OpenCLDevice : BaseDevice {
    public:
        OpenCLDevice();

        bool Malloc(void **ret, size_t size);
        bool Malloc (void **ret, Data &data); 
        bool Free(void *ret);

        bool CopyDataToCPU(void *dst, void *src, size_t size);
        bool CopyDataToCPU(Data &data);
        bool CopyDataFromCPU(void *dst, void *src, size_t size);
        bool CopyDataFromCPU(Data &data);
    
    private:
        OpenCLAllocator *oclAllocator_ = nullptr;
    };


    class OpenCLLinearOp : BaseOperator {
    public:
        OpenCLLinearOp();
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    
    private:
        std::shared_ptr<cl::Kernel> kernel_;
        std::shared_ptr<cl::Kernel> kernelNoBias_;
    };
}

#endif
