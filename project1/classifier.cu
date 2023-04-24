/*
 * https://siboehm.com/articles/22/CUDA-MMM
 */
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cassert>
#include "cuda_profiler_api.h"


const int Ni = 25088;
const int Nn = 4096;
const int nIters = 5; // # of times to average time calculation over

const int nBlocks = 500; // Titan V has 640 cores and 80 SM
const int nThreads = 1024; // divisible by 32, max 1024


bool is_gpu_cpu_arr_equal(int *output, int *cuOutput, int outputLen) {
    for(int i=0; i<outputLen; i++) {
        if(output[i] != cuOutput[i])
            return false;
    }
    return true;
}


void mat_mult_cpu(int *input, int *weights, int *output)
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
void mat_mult_gpu(int *input, int *weights, int *output)
{    
    __shared__ int outputReduce[nThreads];

    int rowsPerBlock = (Nn + nBlocks - 1) / nBlocks;
    int iStart = blockIdx.x*rowsPerBlock;
    int tid = threadIdx.x;

    for(int i=iStart; i<iStart+rowsPerBlock && i<Nn; i++) {
        int sum = 0;
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
    int *weights = (int*)malloc(Ni*Nn*sizeof(int));
    int *input   = (int*)malloc(Ni*sizeof(int));
    int *output  = (int*)malloc(Nn*sizeof(int));
    for(int i=0; i<Ni*Nn; i++)
        weights[i] = rand() % 10;
    for(int i=0; i<Ni; i++)
        input[i] = rand() % 10;

    // Naive CPU Implementation
    double elapsedTime = 0;
    for(int i=0; i<nIters; i++) {
        auto time0 = std::chrono::steady_clock::now();
        mat_mult_cpu(input, weights, output);
        auto time1 = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedSeconds = time1 - time0;
        elapsedTime += elapsedSeconds.count();
    }
    std::cout << "CPU Time:       " << elapsedTime/nIters << std::endl;

    // GPU Setup
    int *cuWeights, *cuInput, *cuOutput;
    cudaMalloc(&cuWeights, Ni*Nn*sizeof(int));
    cudaMalloc(&cuInput, Ni*sizeof(int));
    cudaMalloc(&cuOutput, Nn*sizeof(int));
    cudaMemcpy(cuWeights, weights, Ni*Nn*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuInput, input, Ni*sizeof(int), cudaMemcpyHostToDevice);

    // Naive GPU Implementation
    for(int i=0; i<nIters; i++) {
        // cudaMemset(cuOutput, 0, Nn*sizeof(int)); // only needed for the naive example
        auto time0 = std::chrono::steady_clock::now();
        mat_mult_gpu<<<nBlocks, nThreads>>>(cuInput, cuWeights, cuOutput);
        cudaDeviceSynchronize();
        auto time1 = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedSeconds = time1 - time0;
        elapsedTime += elapsedSeconds.count();
    }
    int *validationOutput = (int*)malloc(Nn*sizeof(int));
    cudaMemcpy(validationOutput, cuOutput, Nn*sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Naive GPU Time: " << elapsedTime/nIters << std::endl;
    assert(is_gpu_cpu_arr_equal(output, validationOutput, Nn));

    // Free Memory
    cudaFree(cuOutput);
    cudaFree(cuInput);
    cudaFree(cuWeights);
    free(output);
    free(input);
    free(weights);

    // Cuda Error Checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    cudaProfilerStop();
}