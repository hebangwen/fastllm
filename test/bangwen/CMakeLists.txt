add_executable(gemv_bench gemv_benchmark.cpp)
target_link_libraries(gemv_bench PUBLIC fastllm)
