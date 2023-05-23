/*
 * For benchmarking against cuDNN:
 * http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
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

#define Nx 12
#define Ny 12
#define Kx 3
#define Ky 3
#define Ni 16
#define Nn 16
#define blockXDim 12
#define blockYDim 12
#define blockZDim 4

const int outNx = Nx - (Kx-1); // assuming no padding for now
const int outNy = Ny - (Ky-1);

/**
 * Returns true if we need to enforce correctness on the inputs given (Conv1 or Conv2 need to be correct).
 */
bool isCorrectnessEnforced() {
    return (Nx==224 && Ny==224 && Kx==3 && Ky==3 && Ni==64 && Nn==64)
      || (Nx==14 && Ny==14 && Kx==3 && Ky==3 && Ni==512 && Nn==512);
}

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

dim3 gridDims(outNx/blockXDim, outNy/blockYDim, Nn/blockZDim);
dim3 blockDims(blockXDim, blockYDim, blockZDim);
__global__ void Conv2dGpu(float *input, float *kernels, float *output)
{
    int oX = blockXDim*blockIdx.x + threadIdx.x;
    int oY = blockYDim*blockIdx.y + threadIdx.y;
    int oZ = blockZDim*blockIdx.z + threadIdx.z;

    if (oX < outNx && oY < outNy && oZ < Nn) {
        float sum = 0;
        for(int i=0; i<Ni; i++) {
            const int kernelIdxPrefix = oZ*Ni*Kx*Ky + i*Kx*Ky;
            const int inputIdxPrefix = i*Nx*Ny;
            for (int ky=0; ky<Ky; ky++) {
                for (int kx=0; kx<Kx; kx++) {
                    sum += kernels[kernelIdxPrefix + ky*Ky + kx] * input[inputIdxPrefix + (oY+ky)*Nx + (oX+kx)];
                }
            }
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
    if (isCorrectnessEnforced()) {
        runCUDNNConv(input, kernels, validationOutput);
        assert(is_gpu_cpu_arr_equal(output, validationOutput, Nn * outNx * outNy));
    }

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