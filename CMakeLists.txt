cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.1/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)

project(cs259 CXX CUDA)

include_directories(.)
include_directories(finalproject)

add_executable(cs259
        finalproject/cusolver_utils.h
        finalproject/ransac.cu
        finalproject/ransac_img.cu
        finalproject/test.cu
        project1/classifier.cu
        project1/conv1.cu
        project1/convolution.cu
        project2/_convolution.cu
        project2/gpu_specs.cu
        cuda_common.h)
