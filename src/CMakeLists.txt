cmake_minimum_required(VERSION 3.18)
project(CudaImageProcessor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda_processor src/main.cu)

set_target_properties(cuda_processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
