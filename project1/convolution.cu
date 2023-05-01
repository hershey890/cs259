/*
 * For benchmarking against cuDNN:
 * http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
 * 
 * TODO: work on depthwise component of convolution
 */
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cassert>
#include "cuda_profiler_api.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudnn.h>
#include "cublas_v2.h"
#include "../cuda_common.h"


#define CONV1
#ifdef CONV1
    const int Nx = 224;
    const int Ny = 224;
    const int Kx = 3;
    const int Ky = 3;
    const int Ni = 64;
    const int Nn = 64;
#else
    const int Nx = 14;
    const int Ny = 14;
    const int Kx = 3;
    const int Ky = 3;
    const int Ni = 12;
    const int Nn = 12;
#endif
const int outNx = Nx - (Kx-1); // assuming no padding for now
const int outNy = Ny - (Ky-1); 

// const int nIters = 1; // # of times to average time calculation over
// const int nBlocks = 500; // Titan V has 640 cores and 80 SM
// const int nThreads = 1024; // divisible by 32, max 1024


/*
 * Run convolution using cuDNN
 * @param input: input tensor NCHW (# outputs, # input channels, height, width)
 * @param kernels: kernel tensor NCHW (# outputs, # input channels, height, width)
 * @param output: output tensor NCHW (# outputs, # input channels, height, width). 
 *  Output is assumed to be preallocated and is overwritten.
 */
void runCUDNNConv(float *input, float *kernels, float *output)
{
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnConvolutionDescriptor_t convDesc;

    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnCreateFilterDescriptor(&kernelDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, Ni, Ny, Nx);
    cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, Nn, Ni, Ky, Kx);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, Nn, outNy, outNx);
    cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    float *cuInput, *cuKernels, *cuOutput;
    cudaMalloc(&cuInput, Ni*Nx*Ny*sizeof(float));
    cudaMalloc(&cuKernels, Nn*Ni*Kx*Ky*sizeof(float));
    cudaMalloc(&cuOutput, Nn*outNx*outNy*sizeof(float));
    cudaMemcpy(cuInput, input, Ni*Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuKernels, kernels, Nn*Ni*Kx*Ky*sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, inputDesc, cuInput, kernelDesc, cuKernels, convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, NULL, 0, &beta, outputDesc, cuOutput);

    cudaMemcpy(output, cuOutput, Nn*outNx*outNy*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(cuInput);
    cudaFree(cuKernels);
    cudaFree(cuOutput);
    
}

const int nThreads = 1024;
dim3 gridDims(outNx/2, outNy/2, 1); // 222, 222, 64
dim3 blockDims(2, 2, Nn); // 64, 16, 1
__global__ void Conv2dGpu(float *input, float *kernels, float *output)
{
    int oX = 2*blockIdx.x + threadIdx.x;
    int oY = 2*blockIdx.y + threadIdx.y;
    int oZ = threadIdx.z;
    float sum = 0;
    for(int i=0; i<Ni; i++) {
        for(int y=0; y<Ky; y++) {
            for(int x=0; x<Kx; x++) {
                sum += kernels[oZ*Ni*Kx*Ky + i*Kx*Ky + y*Kx + x]
                    * input[i*Nx*Ny + (oY+y)*Nx + (oX+x)];
            }
        }
    }
    output[oZ*outNx*outNy + oY*outNx + oX] = sum;
}


int main(void)
{
    // Create Image, Kernel, and Output
    float *input = (float*)malloc(Ni*Nx*Ny*sizeof(float));
    float *kernels   = (float*)malloc(Nn*Ni*Kx*Ky*sizeof(float));
    float *output  = (float*)malloc(Nn*outNx*outNy*sizeof(float));
    float *validationOutput = (float*)malloc(Nn*outNx*outNy*sizeof(float));
    for(int i=0; i<Nx*Ny; i++)
        input[i] = (float)rand() / (float)RAND_MAX;
    for(int i=0; i<Nn*Kx*Ky; i++)
        kernels[i] = (float)rand() / (float)RAND_MAX;

    // GPU Implementation
    float *cuInput, *cuKernels, *cuOutput;
    cudaMalloc(&cuInput, Ni*Nx*Ny*sizeof(float));
    cudaMalloc(&cuKernels, Nn*Ni*Kx*Ky*sizeof(float));
    cudaMalloc(&cuOutput, Nn*outNx*outNy*sizeof(float));
    cudaMemcpy(cuInput, input, Ni*Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(cudaMemcpy(cuKernels, kernels, Nn*Ni*Kx*Ky*sizeof(float), cudaMemcpyHostToDevice));
    Conv2dGpu<<<gridDims, blockDims>>>(cuInput, cuKernels, cuOutput);
    CHECK_CUDA_ERROR(cudaMemcpy(output, cuOutput, Nn*outNx*outNy*sizeof(float), cudaMemcpyDeviceToHost));
    
    // CUDNN Benchmark
    runCUDNNConv(input, kernels, validationOutput);
    assert(is_gpu_cpu_arr_equal(output, validationOutput, Nn*outNx*outNy));

    // Free Memory
    CHECK_CUDA_ERROR(cudaFree(cuInput));
    cudaFree(cuKernels);
    cudaFree(cuOutput);
    free(output);
    free(kernels);
    free(input);

    CHECK_LAST_CUDA_ERROR();
    cudaDeviceReset();

    return 0;
}