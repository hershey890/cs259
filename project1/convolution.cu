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

const int nIters = 1; // # of times to average time calculation over
const int nBlocks = 500; // Titan V has 640 cores and 80 SM
const int nThreads = 1024; // divisible by 32, max 1024


void conv2d_cpu(float *input, float *kernels, float *output)
{
    const int startX = Kx / 2;
    const int startY = Ky / 2;
    const int endX = Nx - Kx / 2;
    const int endY = Ny - Ky / 2;

    for(int i=0; i<Nn; i++) {
        for(int j=0; j<Ni; j++) {
            for(int k=startY; k<endY; k++) {
                for(int l=startX; l<endX; l++) {
                    
                    float sum = 0;
                    for(int y=-Ky/2; y<=Ky/2; y++) {
                        for(int x=-Kx/2; x<=Kx/2; x++)
                            sum += kernels[i*Ni*Kx*Ky + j*Kx*Ky + y*Kx + x]*input[j*Nx*Ny + (k+y)*Nx + (l+x)];
                    }
                    output[i*outNx*outNy + (k-startY)*outNx + (l-startX)] = sum;

                }
            }
        }
    }
}


// __global __ void conv2d(float *input, float *kernel, float *output)
// {
//     const int startX = Kx / 2;
//     const int startY = Ky / 2;
//     const int endX = Nx - Kx / 2;
//     const int endY = Ny - Ky / 2;

//     const int i = blockIdx.x * blockDim.x + threadIdx.x;
//     const int j = blockIdx.y * blockDim.y + threadIdx.y;
//     const int k = blockIdx.z * blockDim.z + threadIdx.z;

//     if(i < Ni && j < outNy && k < outNx) {
//         float sum = 0;
//         for(int y=-Ky/2; y<=Ky/2; y++) {
//             for(int x=-Kx/2; x<=Kx/2; x++)
//                 sum += kernel[y*Kx + x]*input[i*Nx*Ny + (j+y)*Nx + (k+x)];
//         }
//         output[j*outNx + k] = sum;
//     }
// }
__global__ void conv2d_gpu(float *input, float *kernel, float *output)
{
    const int startX = Kx / 2;
    const int startY = Ky / 2;
    const int endX = Nx - Kx / 2;
    const int endY = Ny - Ky / 2;

    for(int i=0; i<Ni; i++) {
        for(int j=startY; j<endY; j++) {
            for(int k=startX; k<endX; k++) {
                float sum = 0;
                for(int y=-Ky/2; y<=Ky/2; y++) {
                    for(int x=-Kx/2; x<=Kx/2; x++)
                        sum += kernel[y*Kx + x]*input[i*Nx*Ny + (j+y)*Nx + (k+x)];
                }
                output[(j-startY)*outNx + (k-startX)] = sum;
            }
        }
    }
}


int main(void)
{
    // Create Image, Kernel, and Output
    float *input = (float*)malloc(Ni*Nx*Ny*sizeof(float));
    float *kernels   = (float*)malloc(Nn*Ni*Kx*Ky*sizeof(float));
    float *output  = (float*)malloc(Nn*outNx*outNy*sizeof(float));
    for(int i=0; i<Nx*Ny; i++)
        input[i] = (float)rand() / (float)RAND_MAX;
    for(int i=0; i<Nn*Kx*Ky; i++)
        kernels[i] = (float)rand() / (float)RAND_MAX;

    // CUDNN Benchmark
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
    
    // cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, kernelDesc, &Nn, &outNx, &outNy);
    cudnnConvolutionFwdAlgoPerf_t perfResults[1];
    int returnedAlgoCount = 1;
    cudaProfilerStart();
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn, inputDesc, kernelDesc, convDesc, outputDesc, 1, &returnedAlgoCount, perfResults));
    cudaProfilerStop();


    // std::cout << perfResults[0] << std::endl;
    // float *cuInput, *cuKernels, *cuOutput;
    // cudaMalloc(&cuInput, Ni*Nx*Ny*sizeof(float));
    // cudaMalloc(&cuKernels, Nn*Ni*Kx*Ky*sizeof(float));
    // cudaMalloc(&cuOutput, Nn*outNx*outNy*sizeof(float));
    // cudaMemcpy(cuInput, input, Ni*Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(cuKernels, kernels, Nn*Ni*Kx*Ky*sizeof(float), cudaMemcpyHostToDevice);
    // float alpha = 1.0f, beta = 0.0f;
    // cudnnConvolutionForward(cudnn, &alpha, inputDesc, cuInput, kernelDesc, cuKernels, convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, NULL, 0, &beta, outputDesc, cuOutput);
    // cudaMemcpy(output, cuOutput, Nn*outNx*outNy*sizeof(float), cudaMemcpyDeviceToHost);

    // // Naive CPU Implementation
    // double elapsedTime = 0;
    // for(int i=0; i<nIters; i++) {
    //     auto time0 = std::chrono::steady_clock::now();
    //     conv2d_cpu(input, kernels, output);
    //     auto time1 = std::chrono::steady_clock::now();
    //     std::chrono::duration<double> elapsedSeconds = time1 - time0;
    //     elapsedTime += elapsedSeconds.count();
    // }
    // std::cout << "CPU Time: " << elapsedTime/nIters << std::endl;
    // arrayToFile<float>(output, Nn*outNx*outNy, "output_cpu.txt");

    // GPU Implementation
    float *cuInput, *cuKernels, *cuOutput;
    cudaMalloc(&cuInput, Ni*Nx*Ny*sizeof(float));
    cudaMalloc(&cuKernels, Nn*Ni*Kx*Ky*sizeof(float));
    cudaMalloc(&cuOutput, Nn*outNx*outNy*sizeof(float));
    cudaMemcpy(cuInput, input, Ni*Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuKernels, kernels, Nn*Ni*Kx*Ky*sizeof(float), cudaMemcpyHostToDevice);

    elapsedTime = 0;
    for(int i=0; i<nIters; i++) {
        auto time0 = std::chrono::steady_clock::now();
        conv2d<<<nBlocks, nThreads>>>(cuInput, cuKernel, cuOutput);
        cudaDeviceSynchronize();
        auto time1 = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedSeconds = time1 - time0;
        elapsedTime += elapsedSeconds.count();
    }

    // Free Memory
    free(output);
    free(kernels);
    free(input);

    CHECK_LAST_CUDA_ERROR();
    cudaDeviceReset();

    return 0;
}