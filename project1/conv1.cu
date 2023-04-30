#include <string>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cassert>
#include "cuda_profiler_api.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "cublas_v2.h"
#include "../cuda_common.h"


using namespace std;


#ifdef CONV
  #define CONV_Ny 224
  #define CONV_Nx 224
  #define CONV_Ni 64
  #define CONV_Nn 64
#else
  #define CONV_Ny 14
  #define CONV_Nx 14
  #define CONV_Ni 512
  #define CONV_Nn 512
#endif

#define CONV_Ky 3
#define CONV_Kx 3
#define CONV_Sy 1
#define CONV_Sx 1

//Tiling Sizes
#define CONV_Tnn 32
#define CONV_Tn  16
#define CONV_Ti  16

#define CONV_Ty  8
#define CONV_Tx  8

#define CONV_NYPAD (CONV_Ny) // #define CONV_NYPAD (CONV_Ny+CONV_Ky)
#define CONV_NXPAD (CONV_Nx) // #define CONV_NXPAD (CONV_Nx+CONV_Kx)

#define CONV_NYSCL ((CONV_Ny - CONV_Ky + 1)/CONV_Sy) // #define CONV_NYSCL (CONV_Ny/CONV_Sy)
#define CONV_NXSCL ((CONV_Nx - CONV_Kx + 1)/CONV_Sx) // #define CONV_NXSCL (CONV_Nx/CONV_Sx)
                                                                                                                                   
#define CONV_FILTER_SIZE (CONV_Ky*CONV_Kx*CONV_Nn*CONV_Ni)
#define CONV_INPUT_SIZE (CONV_NYPAD*CONV_NXPAD*CONV_Ni)
#define CONV_OUTPUT_SIZE (CONV_NYSCL*CONV_NXSCL*CONV_Nn)

#define CONV_THREADS 1024
#define CONV_BLOCKS 500

using VTYPE = float;

#define OUTPUT_ADDR(ny, nx, nn) (CONV_Nx*CONV_Nn*(ny) + CONV_Nn*(nx) + (nn))
#define INPUT_ADDR(ni, ny, nx) (CONV_Ny*CONV_Nx*(ni) + CONV_Nx*(ny) + (nx))
#define KERNEL_ADDR(ni, nn, ky, kx) (CONV_Nn * CONV_Ky*CONV_Kx*(ni) + CONV_Ky*CONV_Kx*(nn) + CONV_Kx*(ky) + (kx))

bool is_gpu_cpu_arr_equal(VTYPE *output, VTYPE *cuOutput, int outputLen) {
    for(int i=0; i<outputLen; i++) {
        float diff = abs(output[i] - cuOutput[i])/(abs(cuOutput[i]) + 0.0001);
        if(diff > 0.05) {
            std::cout << output[i] << " " << cuOutput[i] << " " << diff << std::endl;
            return false;
        }
    }
    return true;
}

// Base CPU Version (no optimizations)
void convolution_layer_base(VTYPE *kernel, VTYPE *input, VTYPE *output) {
    for (int ny = 0; ny + CONV_Ky < CONV_Ny; ny += CONV_Sy) {
        for (int nx = 0; nx + CONV_Kx < CONV_Nx; nx += CONV_Sx) {
            for (int ky = 0; ky < CONV_Ky; ky++) {
                for (int kx = 0; kx < CONV_Kx; kx++) {
                    for (int ni = 0; ni < CONV_Ni; ni++) {
                        for (int nn = 0; nn < CONV_Nn; nn++) {
                          // Assumes output has already been pre-zero'd out.
                          output[OUTPUT_ADDR(ny, nx, nn)] += input[INPUT_ADDR(ni, ny + ky, nx + kx)] * kernel[KERNEL_ADDR(ni, nn, ky, kx)];
                        }
                    }
                }
            }
        }
    }
}

// Conv Layer DianNao implementation.
void convolution_layer(VTYPE *kernel, VTYPE *input, VTYPE *output) {
  VTYPE sum[CONV_Nn]={0};

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int y = 0; y + CONV_Ky < CONV_Ny; y += CONV_Sy) { // tiling for y;
    int xout = 0;
    for (int x = 0; x + CONV_Kx < CONV_Nx; x += CONV_Sx) { // tiling for x;
      for (int nn = 0; nn < CONV_Nn; nn += CONV_Tn) {
        for (int n = nn; n < nn + CONV_Tn; n++) {
          sum[n]=0;
        }

        // sliding window;
        for (int ky = 0; ky < CONV_Ky; ky++)
          for (int kx = 0; kx < CONV_Kx; kx++)
            for (int n = nn; n < nn + CONV_Tn; n++)
              for (int i = 0; i < CONV_Ni; i++) {
                VTYPE sv = kernel[KERNEL_ADDR(i, n, ky, kx)]; // VTYPE sv = synapse[ky][kx][n][i];
                VTYPE nv = input[INPUT_ADDR(i, ky + y, kx + x)]; // neuron_i[ky + y][kx + x][i];
                sum[n]+=sv*nv;
              }
        for (int n = nn; n < nn + CONV_Tn; n++) {
            output[OUTPUT_ADDR(yout, xout, n)] = sum[n]; //   output[yout][xout][n] = sum[n];
        }
      }
      xout++; 
    }
    yout++;
  }
}

__global__ void convolution_layer_parallelized_gpu(VTYPE *synapse_2, VTYPE *neuron_i_2, VTYPE *neuron_n) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    VTYPE sum = 0;

    for (int ni = 0; ni < CONV_Ni; ni++) {
        sum +=           neuron_i_2[INPUT_ADDR(ni, y+0, x+0)] * synapse_2[KERNEL_ADDR(ni, z, 0, 0)]
                        + neuron_i_2[INPUT_ADDR(ni, y+0, x+1)] * synapse_2[KERNEL_ADDR(ni, z, 0, 1)]
                        + neuron_i_2[INPUT_ADDR(ni, y+0, x+2)] * synapse_2[KERNEL_ADDR(ni, z, 0, 2)]
                        + neuron_i_2[INPUT_ADDR(ni, y+1, x+0)] * synapse_2[KERNEL_ADDR(ni, z, 1, 0)]
                        + neuron_i_2[INPUT_ADDR(ni, y+1, x+1)] * synapse_2[KERNEL_ADDR(ni, z, 1, 1)]
                        + neuron_i_2[INPUT_ADDR(ni, y+1, x+2)] * synapse_2[KERNEL_ADDR(ni, z, 1, 2)]
                        + neuron_i_2[INPUT_ADDR(ni, y+2, x+0)] * synapse_2[KERNEL_ADDR(ni, z, 2, 0)]
                        + neuron_i_2[INPUT_ADDR(ni, y+2, x+1)] * synapse_2[KERNEL_ADDR(ni, z, 2, 1)]
                        + neuron_i_2[INPUT_ADDR(ni, y+2, x+2)] * synapse_2[KERNEL_ADDR(ni, z, 2, 2)];

        neuron_n[OUTPUT_ADDR(x, y, z)] = sum;
    }
}

__global__ void convolution_layer_tiled_gpu(VTYPE *kernel, VTYPE *input, VTYPE *output) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    const int blockDim_x = blockDim.x;;
    const int blockDim_y = blockDim.y;
    const int blockDim_z = blockDim.z;
    int iStride = blockDim.z;
    int tidz = threadIdx.z;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    VTYPE sum = 0;

    __shared__ VTYPE cache[34 * 10 * 4];
    
    for (int nni = 0; nni < CONV_Ni; nni+=iStride) {
        cache[tidz * blockDim_x * blockDim_y + tidy * blockDim_x + tidx] = input[INPUT_ADDR(nni + tidz, y, x)];
        if (tidy >= blockDim_y - 2) {
            cache[tidz * blockDim_x * blockDim_y + (tidy + 2) * blockDim_x + tidx] = input[INPUT_ADDR(nni + tidz, y + 2, x)];
        }
        if (tidx >= blockDim_x - 2) {
            cache[tidz * blockDim_x * blockDim_y + tidy * blockDim_x + tidx + 2] = input[INPUT_ADDR(nni + tidz, y, x + 2)];
        }
        if (tidx >= blockDim_x - 2 && tidy >= blockDim_y - 2) {
            cache[tidz * blockDim_x * blockDim_y + (tidy + 2) * blockDim_x + tidx + 2] = input[INPUT_ADDR(nni + tidz, y + 2, x + 2)];
        }
        __syncthreads();

        for (int ni = 0; ni < iStride; ni++) {
            sum += cache[ni * blockDim_x * blockDim_y + tidy * blockDim_x + tidx] * kernel[KERNEL_ADDR(nni + ni, z, 0, 0)]
                + cache[ni * blockDim_x * blockDim_y + tidy * blockDim_x + tidx + 1] * kernel[KERNEL_ADDR(nni + ni, z, 0, 1)]
                + cache[ni * blockDim_x * blockDim_y + tidy * blockDim_x + tidx + 2] * kernel[KERNEL_ADDR(nni + ni, z, 0, 2)]
                + cache[ni * blockDim_x * blockDim_y + (tidy + 1) * blockDim_x + tidx] * kernel[KERNEL_ADDR(nni + ni, z, 1, 0)]
                + cache[ni * blockDim_x * blockDim_y + (tidy + 1) * blockDim_x + tidx + 1] * kernel[KERNEL_ADDR(nni + ni, z, 1, 1)]
                + cache[ni * blockDim_x * blockDim_y + (tidy + 1) * blockDim_x + tidx + 2] * kernel[KERNEL_ADDR(nni + ni, z, 1, 2)]
                + cache[ni * blockDim_x * blockDim_y + (tidy + 2) * blockDim_x + tidx] * kernel[KERNEL_ADDR(nni + ni, z, 2, 0)]
                + cache[ni * blockDim_x * blockDim_y + (tidy + 2) * blockDim_x + tidx + 1] * kernel[KERNEL_ADDR(nni + ni, z, 2, 1)]
                + cache[ni * blockDim_x * blockDim_y + (tidy + 2) * blockDim_x + tidx + 2] * kernel[KERNEL_ADDR(nni + ni, z, 2, 2)];
        }
        __syncthreads();
    }
    output[OUTPUT_ADDR(x, y, z)] = sum;
}

int main(const int argc, const char** argv) {
    VTYPE *kernel =  (VTYPE*)malloc(sizeof(VTYPE) * CONV_FILTER_SIZE);
    VTYPE *input = (VTYPE*)malloc(sizeof(VTYPE) * CONV_INPUT_SIZE);
    VTYPE *output =   (VTYPE*)malloc(sizeof(VTYPE) * CONV_OUTPUT_SIZE);
    VTYPE *output_validation = (VTYPE*)malloc(sizeof(VTYPE) * CONV_OUTPUT_SIZE);
    for(int i=0; i<CONV_FILTER_SIZE; i++)
        kernel[i] = (float)rand() / (float)RAND_MAX;
    for(int i=0; i<CONV_INPUT_SIZE; i++)
        input[i] = (float)rand() / (float)RAND_MAX;
    for(int i=0; i<CONV_OUTPUT_SIZE; i++) {
        output[i] = 0.0f;
        output_validation[i] = 0.0f;
    }
    const int X_DIM = 1;
    const int Y_DIM = 1;
    const int Z_DIM = 1024;
    dim3 gridDim(CONV_NXSCL/X_DIM, CONV_NYSCL/Y_DIM, CONV_Nn/Z_DIM);
    dim3 blockDim(X_DIM, Y_DIM, Z_DIM);

    VTYPE *cuInput, *cuKernels, *cuOutput;
    CHECK_CUDA_ERROR(cudaMalloc(&cuInput, CONV_INPUT_SIZE*sizeof(VTYPE)));
    CHECK_CUDA_ERROR(cudaMalloc(&cuKernels, CONV_FILTER_SIZE*sizeof(VTYPE)));
    CHECK_CUDA_ERROR(cudaMalloc(&cuOutput, CONV_OUTPUT_SIZE*sizeof(VTYPE)));
    CHECK_CUDA_ERROR(cudaMemcpy(cuInput, input, CONV_INPUT_SIZE*sizeof(VTYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(cuKernels, kernel, CONV_FILTER_SIZE*sizeof(VTYPE), cudaMemcpyHostToDevice));

    convolution_layer_parallelized_gpu<<<gridDim, blockDim>>>(cuKernels, cuInput, cuOutput);

    CHECK_CUDA_ERROR(cudaMemcpy(output, cuOutput, CONV_OUTPUT_SIZE*sizeof(VTYPE), cudaMemcpyDeviceToHost));
    
    convolution_layer_base(kernel, input, output_validation);
    assert(is_gpu_cpu_arr_equal(output, output_validation, CONV_OUTPUT_SIZE));

    cudaFree(cuOutput);
    cudaFree(cuInput);
    cudaFree(cuKernels);

    // free(kernel);
    // free(input);
    // free(output);
    // free(output_validation);

    CHECK_LAST_CUDA_ERROR();
    cudaDeviceReset();
}


