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


// #define CONV1
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
    const int Ni = 512;
    const int Nn = 512;
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

#ifdef CONV1
    const int blockXDim = 222;
    const int blockYDim = 1;
    const int blockZDim = 4;
#else
    const int blockXDim = outNx;
    const int blockYDim = outNy;
    const int blockZDim = 4;
#endif

dim3 gridDims(outNx/blockXDim, outNy/blockYDim, Nn/blockZDim);
dim3 blockDims(blockXDim, blockYDim, blockZDim);
__global__ void Conv2dGpu(float *input, float *kernels, float *output)
{
    const int nElem = Kx*Ky*Ni;
    __shared__ float inputShared[nElem];
    // int tid = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    // int elemPerThread = nElem/nThreads;
    // if(tid*elemPerThread < nElem) {
    //     for(int i=elemPerThread*tid; i<elemPerThread*(tid+1)-5; i++) {
    //         inputShared[i] = input[i];
    //     }
    // }
    __syncthreads();

    int oX = blockXDim*blockIdx.x + threadIdx.x;
    int oY = blockYDim*blockIdx.y + threadIdx.y;
    int oZ = blockZDim*blockIdx.z + threadIdx.z;

    if (oX < outNx && oY < outNy && oZ < Nn) {
        float sum = 0;
        for(int i=0; i<Ni; i++) {
            const int kernelIdxPrefix = oZ*Ni*Kx*Ky + i*Kx*Ky;
            const int inputIdxPrefix = i*Nx*Ny;
            sum += kernels[kernelIdxPrefix + 0] * input[inputIdxPrefix + (oY+0)*Nx + (oX+0)]
                +  kernels[kernelIdxPrefix + 1] * input[inputIdxPrefix + (oY+0)*Nx + (oX+1)]
                +  kernels[kernelIdxPrefix + 2] * input[inputIdxPrefix + (oY+0)*Nx + (oX+2)]
                +  kernels[kernelIdxPrefix + 3] * input[inputIdxPrefix + (oY+1)*Nx + (oX+0)]
                +  kernels[kernelIdxPrefix + 4] * input[inputIdxPrefix + (oY+1)*Nx + (oX+1)]
                +  kernels[kernelIdxPrefix + 5] * input[inputIdxPrefix + (oY+1)*Nx + (oX+2)]
                +  kernels[kernelIdxPrefix + 6] * input[inputIdxPrefix + (oY+2)*Nx + (oX+0)]
                +  kernels[kernelIdxPrefix + 7] * input[inputIdxPrefix + (oY+2)*Nx + (oX+1)]
                +  kernels[kernelIdxPrefix + 8] * input[inputIdxPrefix + (oY+2)*Nx + (oX+2)];
        }
        output[oZ*outNx*outNy + oY*outNx + oX] = sum;
    }
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
    // assert(is_gpu_cpu_arr_equal(output, validationOutput, Nn*outNx*outNy));

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