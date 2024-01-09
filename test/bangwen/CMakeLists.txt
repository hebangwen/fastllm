add_executable(gemv_bench gemv_benchmark.cpp)
target_link_libraries(gemv_bench PUBLIC fastllm)

if(USE_OPENCL)
  add_executable(gemv_opencl gemv_opencl.cpp)
  target_link_libraries(gemv_opencl PUBLIC fastllm OpenCL)
endif()
