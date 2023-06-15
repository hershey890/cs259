cmake_minimum_required(VERSION 3.22)

set(Boost_USE_STATIC_LIBS ON)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.1/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)

project(cs259 CXX CUDA)

find_package(Boost 1.53 REQUIRED COMPONENTS thread)
find_package(CUDAToolkit)

include_directories(.)
include_directories(finalproject)

add_executable(ransac
    finalproject/ransac.cu
)

target_link_libraries(ransac CUDA::cublas)
