/*
 * Compile with: nvcc classifier.cu -o classifier -std=c++11 -lcublas
 * Class 1 Performance (4096x25088)
 * --------------------------------
 * Time(%)      Time     Calls       Avg       Min       Max  Name
 *   1.87%  3.3943ms         5  678.85us  672.35us  700.70us  mat_mult_gpu(float*, float*, float*)
 *   0.37%  673.63us         1  673.63us  673.63us  673.63us  void gemv2T_kernel_val<int, int, float, float, float, float, int=128, int=16, int=4, int=4, bool=0, bool=0, cublasGemvParam
 *
 * Class 2 Performance (1024x4096)
 * -------------------------------
 * Time(%)      Time     Calls       Avg       Min       Max  Name
 *   4.78%  406.94us         5  81.388us  75.647us  102.40us  mat_mult_gpu(float*, float*, float*)
 *   0.46%  39.328us         1  39.328us  39.328us  39.328us  void gemv2T_kernel_val<int, int, float, float, float, float, int=128, int=16, int=4, int=4, bool=0, bool=0, cublasGemvParams<cublasGemvTensorStridedBat
 * 
 * Resources
 * ---------
 * https://siboehm.com/articles/22/CUDA-MMM
 * https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
 */
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cassert>
#include "cuda_profiler_api.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "cublas_v2.h"
#include "../cuda_common.h"


// #define CLASS1
#ifdef CLASS1
    const int Ni = 25088;
    const int Nn = 4096;
#else
    const int Ni = 4096;
    const int Nn = 1024;
#endif


const int nIters = 5; // # of times to average time calculation over
const int nBlocks = 500; // Titan V has 640 cores and 80 SM
const int nThreads = 1024; // divisible by 32, max 1024


bool is_gpu_cpu_arr_equal(float *output, float *cuOutput, float outputLen) {
    for(int i=0; i<outputLen; i++) {
        float diff = abs(output[i] - cuOutput[i])/(abs(cuOutput[i]) + 0.0001);
        if(diff > 0.05) {
            std::cout << output[i] << " " << cuOutput[i] << " " << diff << std::endl;
            return false;
        }
    }
    return true;
}


void mat_mult_cpu(float *input, float *weights, float *output)
{
    for(int i=0; i<Nn; i++) {
        output[i] = 0;
        for(int j=0; j<Ni; j++)
            output[i] += weights[Ni*i + j]*input[j];
    }
}


// 111ms. for 32 blocks 1 thread, only works for 1 thread
__global__
void mat_mult_gpu_naive(int *input, int *weights, int *output) 
{
    int rowsPerBlock = Nn / gridDim.x;
    int iStart = blockIdx.x*rowsPerBlock;
    int jStart = threadIdx.x;
    int jStride = blockDim.x;
    for(int i=iStart; i<iStart+rowsPerBlock; i++) { //4096/32=128
        int sum = 0;
        for(int j=jStart; j<Ni; j += jStride) { // 25088/32=784
            sum += weights[Ni*i + j]*input[j];
        }
        output[i] += sum;
    }
}


// 650 us for 500 blocks and 1024 threads
__global__
void mat_mult_gpu(float *input, float *weights, float *output)
{    
    __shared__ float outputReduce[nThreads];

    int rowsPerBlock = (Nn + nBlocks - 1) / nBlocks;
    int iStart = blockIdx.x*rowsPerBlock;
    int tid = threadIdx.x;

    for(int i=iStart; i<iStart+rowsPerBlock && i<Nn; i++) {
        float sum = 0;
        for(int j=tid; j<Ni; j += nThreads)
            sum += weights[Ni*i + j]*input[j];
        outputReduce[tid] = sum;

        // Reduction - deals with 
        // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
        __syncthreads();
        if(tid == 0) {
            sum = 0;
            for(int j=0; j<nThreads; j++)
                sum += outputReduce[j];
            output[i] = sum;
        }
    }
}


int main()
{
    // Create Weights, Inputs, and Outputs
    float *weights = (float*)malloc(Ni*Nn*sizeof(float));
    float *input   = (float*)malloc(Ni*sizeof(float));
    float *output  = (float*)malloc(Nn*sizeof(float));
    for(int i=0; i<Ni*Nn; i++)
        weights[i] = (float)rand() / (float)RAND_MAX;
    for(int i=0; i<Ni; i++)
        input[i] = (float)rand() / (float)RAND_MAX;


    // Naive CPU Implementation
    double elapsedTime = 0;
    for(int i=0; i<nIters; i++) {
        auto time0 = std::chrono::steady_clock::now();
        mat_mult_cpu(input, weights, output);
        auto time1 = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedSeconds = time1 - time0;
        elapsedTime += elapsedSeconds.count();
    }
    std::cout << "CPU Time: " << elapsedTime/nIters << std::endl;


    // GPU Setup
    float *cuWeights, *cuInput, *cuOutput;
    cudaMalloc(&cuWeights, Ni*Nn*sizeof(float));
    cudaMalloc(&cuInput,   Ni*sizeof(float));
    cudaMalloc(&cuOutput,  Nn*sizeof(float));
    cudaMemcpy(cuWeights, weights, Ni*Nn*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuInput, input, Ni*sizeof(float), cudaMemcpyHostToDevice);

    // GPU Implementation
    for(int i=0; i<nIters; i++) {
        auto time0 = std::chrono::steady_clock::now();
        mat_mult_gpu<<<nBlocks, nThreads>>>(cuInput, cuWeights, cuOutput);
        cudaDeviceSynchronize();
        auto time1 = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedSeconds = time1 - time0;
        elapsedTime += elapsedSeconds.count();
    }
    float *validationOutput = (float*)malloc(Nn*sizeof(float));
    cudaMemcpy(validationOutput, cuOutput, Nn*sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "GPU Time: " << elapsedTime/nIters << std::endl;
    assert(is_gpu_cpu_arr_equal(output, validationOutput, Nn));

    // Free Memory
    cudaFree(cuOutput);
    cudaFree(cuInput);
    cudaFree(cuWeights);


    /* CUBLAS Benchmark
     * Compares our kernel vs. CUBLAS performance
     * https://github.com/deeperlearning/professional-cuda-c-programming/blob/master/solutions/chapter08/cublas-matrix-matrix-async.cu
     */
    // Create the cuBLAS handle
    cublasHandle_t handle = 0;
    cublasCreate(&handle);

    // // Allocate device memory
    float *cublasInput, *cublasMatrix, *cublasOutput, *y;
    cudaMalloc(&cublasInput,  sizeof(float) * Ni);
    cudaMalloc(&cublasMatrix, sizeof(float) * Ni * Nn);
    cudaMalloc(&cublasOutput, sizeof(float) * Nn);
    cudaMalloc(&y, 1*sizeof(float));

    // // Transfer inputs to the device
    cublasSetMatrix(Nn, Ni, sizeof(float), weights, Nn, cublasMatrix, Nn);
    cublasSetVector(Ni, sizeof(float), input, Ni, cublasInput, Ni);

    // Execute Matrix Vector-Multiplication
    const float alpha = 1.0f;
    const float beta = 0;
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, Ni, Nn, &alpha, cublasMatrix, Ni, cublasInput, 1, &beta, cublasOutput, 1));
    CHECK_CUBLAS(cublasGetVector(Nn, sizeof(float), cublasOutput, 1, validationOutput, 1));
    cublasDestroy(handle);
    
    // Free Memory
    cudaFree(cublasOutput);
    cudaFree(cublasMatrix);
    cudaFree(cublasInput);
    free(output);
    free(input);
    free(weights);

    CHECK_LAST_CUDA_ERROR();
    cudaDeviceReset();

    return 0;
}