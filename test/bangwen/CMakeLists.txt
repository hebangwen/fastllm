add_executable(gemv_bench gemv_benchmark.cpp)
target_link_libraries(gemv_bench PUBLIC fastllm)

add_executable(gemv_opencl gemv_opencl.cpp)
target_link_libraries(gemv_opencl PUBLIC fastllm OpenCL mace)

target_include_directories(gemv_opencl PRIVATE 
  ${PROJECT_SOURCE_DIR}/third_party/mace 
  ${PROJECT_SOURCE_DIR}/third_party/mace/include 
  ${PROJECT_BINARY_DIR}/third_party/mace
  ${PROJECT_BINARY_DIR}/third_party/mace/third_party/install/protobuf/include
  ${HALF_INCLUDE_DIR})

if (MACE_ENABLE_OPENCL)
  target_compile_options(gemv_opencl PRIVATE "-DMACE_ENABLE_OPENCL")
endif()
