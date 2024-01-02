add_executable(gemv_bench gemv_benchmark.cpp)
target_link_libraries(gemv_bench PUBLIC fastllm mace)

add_executable(gemv_opencl gemv_opencl.cpp)
target_link_libraries(gemv_opencl PUBLIC fastllm mace)

set(MACE_INCLUDE_DIR
  ${PROJECT_SOURCE_DIR}/third_party/mace 
  ${PROJECT_SOURCE_DIR}/third_party/mace/include 
  ${PROJECT_BINARY_DIR}/third_party/mace
  ${PROJECT_BINARY_DIR}/third_party/mace/third_party/install/protobuf/include
  ${OPENCL_CLHPP_INCLUDE_DIR}
  ${OPENCL_HEADERS_INCLUDE_DIR}
  ${HALF_INCLUDE_DIR})

target_include_directories(gemv_opencl PRIVATE ${MACE_INCLUDE_DIR})
target_include_directories(gemv_bench PRIVATE ${MACE_INCLUDE_DIR})

if (MACE_ENABLE_OPENCL)
  target_compile_options(gemv_opencl PRIVATE "-DMACE_ENABLE_OPENCL")
  target_compile_options(gemv_bench PRIVATE "-DMACE_ENABLE_OPENCL")
endif()
