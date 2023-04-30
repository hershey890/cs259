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

#define CONV1_Ny 224
#define CONV1_Nx 224
#define CONV1_Ky 3
#define CONV1_Kx 3
#define CONV1_Ni 64
#define CONV1_Nn 64

#define CONV1_Sy 1
#define CONV1_Sx 1

//Tiling Sizes
#define CONV1_Tnn 32
#define CONV1_Tn  16
#define CONV1_Ti  16

#define CONV1_Ty  8
#define CONV1_Tx  8

#define CONV1_NYPAD (CONV1_Ny) // #define CONV1_NYPAD (CONV1_Ny+CONV1_Ky)
#define CONV1_NXPAD (CONV1_Nx) // #define CONV1_NXPAD (CONV1_Nx+CONV1_Kx)

#define CONV1_NYSCL (CONV1_Ny - CONV1_Ky + 1) // #define CONV1_NYSCL (CONV1_Ny/CONV1_Sy)
#define CONV1_NXSCL (CONV1_Nx - CONV1_Kx + 1) // #define CONV1_NXSCL (CONV1_Nx/CONV1_Sx)
                                                                                                                                   
#define CONV1_SYNAPSE_SIZE (CONV1_Ky*CONV1_Kx*CONV1_Nn*CONV1_Ni)
#define CONV1_NEURON_INPUT_SIZE (CONV1_NYPAD*CONV1_NXPAD*CONV1_Ni)
#define CONV1_NEURON_OUTPUT_SIZE (CONV1_NYSCL*CONV1_NXSCL*CONV1_Nn)

#define CONV1_THREADS 1024
#define CONV1_BLOCKS 500

using VTYPE = float;

#define NEUR_OUT_ADDR(ny, nx, nn) (CONV1_Nx*CONV1_Nn*(ny) + CONV1_Nn*(nx) + (nn))
#define NEUR_IN_ADDR(ni, ny, nx) (CONV1_Ny*CONV1_Nx*(ni) + CONV1_Nx*(ny) + (nx))
#define SYNAPSE_ADDR(ni, nn, ky, kx) (CONV1_Nn * CONV1_Ky*CONV1_Kx*(ni) + CONV1_Ky*CONV1_Kx*(nn) + CONV1_Kx*(ky) + (kx))

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

// Conv Layer DianNao implementation.
void  convolution_layer(VTYPE *synapse_2, VTYPE *neuron_i_2, VTYPE *neuron_n) {
  VTYPE sum[CONV1_Nn]={0};

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int y = 0; y + CONV1_Ky < CONV1_Ny; y += CONV1_Sy) { // tiling for y;
    int xout = 0;
    for (int x = 0; x + CONV1_Kx < CONV1_Nx; x += CONV1_Sx) { // tiling for x;
      for (int nn = 0; nn < CONV1_Nn; nn += CONV1_Tn) {
        for (int n = nn; n < nn + CONV1_Tn; n++) {
          sum[n]=0;
        }

        // sliding window;
        for (int ky = 0; ky < CONV1_Ky; ky++)
          for (int kx = 0; kx < CONV1_Kx; kx++)
            for (int n = nn; n < nn + CONV1_Tn; n++)
              for (int i = 0; i < CONV1_Ni; i++) {
                VTYPE sv = synapse_2[SYNAPSE_ADDR(i, n, ky, kx)]; // VTYPE sv = synapse[ky][kx][n][i];
                VTYPE nv = neuron_i_2[NEUR_IN_ADDR(i, ky + y, kx + x)]; // neuron_i[ky + y][kx + x][i];
                sum[n]+=sv*nv;
              }
        for (int n = nn; n < nn + CONV1_Tn; n++) {
            neuron_n[NEUR_OUT_ADDR(yout, xout, n)] = sum[n]; //   neuron_n[yout][xout][n] = sum[n];
        }
      }
      xout++; 
    }
    yout++;
  }
}

__global__ void convolution_layer_tiled_gpu(VTYPE *synapse_2, VTYPE *neuron_i_2, VTYPE *neuron_n) {

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

    __shared__ VTYPE input[34 * 10 * 4];
    
    for (int nni = 0; nni < CONV1_Ni; nni+=iStride) {
        input[tidz * blockDim_x * blockDim_y + tidy * blockDim_x + tidx] = neuron_i_2[NEUR_IN_ADDR(nni + tidz, y, x)];
        if (tidy >= blockDim_y - 2) {
            input[tidz * blockDim_x * blockDim_y + (tidy + 2) * blockDim_x + tidx] = neuron_i_2[NEUR_IN_ADDR(nni + tidz, y + 2, x)];
        }
        if (tidx >= blockDim_x - 2) {
            input[tidz * blockDim_x * blockDim_y + tidy * blockDim_x + tidx + 2] = neuron_i_2[NEUR_IN_ADDR(nni + tidz, y, x + 2)];
        }
        if (tidx >= blockDim_x - 2 && tidy >= blockDim_y - 2) {
            input[tidz * blockDim_x * blockDim_y + (tidy + 2) * blockDim_x + tidx + 2] = neuron_i_2[NEUR_IN_ADDR(nni + tidz, y + 2, x + 2)];
        }
        __syncthreads();

        for (int ni = 0; ni < iStride; ni++) {
            sum += input[ni * blockDim_x * blockDim_y + tidy * blockDim_x + tidx] * synapse_2[SYNAPSE_ADDR(nni + ni, z, 0, 0)]
                + input[ni * blockDim_x * blockDim_y + tidy * blockDim_x + tidx + 1] * synapse_2[SYNAPSE_ADDR(nni + ni, z, 0, 1)]
                + input[ni * blockDim_x * blockDim_y + tidy * blockDim_x + tidx + 2] * synapse_2[SYNAPSE_ADDR(nni + ni, z, 0, 2)]
                + input[ni * blockDim_x * blockDim_y + (tidy + 1) * blockDim_x + tidx] * synapse_2[SYNAPSE_ADDR(nni + ni, z, 1, 0)]
                + input[ni * blockDim_x * blockDim_y + (tidy + 1) * blockDim_x + tidx + 1] * synapse_2[SYNAPSE_ADDR(nni + ni, z, 1, 1)]
                + input[ni * blockDim_x * blockDim_y + (tidy + 1) * blockDim_x + tidx + 2] * synapse_2[SYNAPSE_ADDR(nni + ni, z, 1, 2)]
                + input[ni * blockDim_x * blockDim_y + (tidy + 2) * blockDim_x + tidx] * synapse_2[SYNAPSE_ADDR(nni + ni, z, 2, 0)]
                + input[ni * blockDim_x * blockDim_y + (tidy + 2) * blockDim_x + tidx + 1] * synapse_2[SYNAPSE_ADDR(nni + ni, z, 2, 1)]
                + input[ni * blockDim_x * blockDim_y + (tidy + 2) * blockDim_x + tidx + 2] * synapse_2[SYNAPSE_ADDR(nni + ni, z, 2, 2)];
                
                // + neuron_i_2[NEUR_IN_ADDR(ni, y+0, x+1)] * synapse_2[SYNAPSE_ADDR(ni, z, 0, 1)]
                // + neuron_i_2[NEUR_IN_ADDR(ni, y+0, x+2)] * synapse_2[SYNAPSE_ADDR(ni, z, 0, 2)]
                // + neuron_i_2[NEUR_IN_ADDR(ni, y+1, x+0)] * synapse_2[SYNAPSE_ADDR(ni, z, 1, 0)]
                // + neuron_i_2[NEUR_IN_ADDR(ni, y+1, x+1)] * synapse_2[SYNAPSE_ADDR(ni, z, 1, 1)]
                // + neuron_i_2[NEUR_IN_ADDR(ni, y+1, x+2)] * synapse_2[SYNAPSE_ADDR(ni, z, 1, 2)]
                // + neuron_i_2[NEUR_IN_ADDR(ni, y+2, x+0)] * synapse_2[SYNAPSE_ADDR(ni, z, 2, 0)]
                // + neuron_i_2[NEUR_IN_ADDR(ni, y+2, x+1)] * synapse_2[SYNAPSE_ADDR(ni, z, 2, 1)]
                // + neuron_i_2[NEUR_IN_ADDR(ni, y+2, x+2)] * synapse_2[SYNAPSE_ADDR(ni, z, 2, 2)];
        }
        __syncthreads();
    }
    neuron_n[NEUR_OUT_ADDR(x, y, z)] = sum;
        // int x = blockIdx.x * blockDim.x + threadIdx.x;

    // const int start_row = x * CONV1_ROWS_PROCESSED;
    // const int end_row = min(CONV1_Ky, start_row + CONV1_ROWS_PROCESSED);
    // for (int ny = start_row; ny < end_row; ny += CONV1_Sy) {
    //     for (int nx = 0; nx + CONV1_Kx < CONV1_Nx; nx += CONV1_Sx) {
    //         for (int ni = 0; ni < CONV1_Ni; ni++) {
    //             for (int nn = 0; nn < CONV1_Nn; nn++) {
    //                 neuron_n[NEUR_OUT_ADDR(ny, nx, nn)] = 
    //                       neuron_i_2[NEUR_IN_ADDR(ni, ny+0, nx+0)] * synapse_2[SYNAPSE_ADDR(ni, nn, 0, 0)]
    //                     + neuron_i_2[NEUR_IN_ADDR(ni, ny+0, nx+1)] * synapse_2[SYNAPSE_ADDR(ni, nn, 0, 1)]
    //                     + neuron_i_2[NEUR_IN_ADDR(ni, ny+0, nx+2)] * synapse_2[SYNAPSE_ADDR(ni, nn, 0, 2)]
    //                     + neuron_i_2[NEUR_IN_ADDR(ni, ny+1, nx+0)] * synapse_2[SYNAPSE_ADDR(ni, nn, 1, 0)]
    //                     + neuron_i_2[NEUR_IN_ADDR(ni, ny+1, nx+1)] * synapse_2[SYNAPSE_ADDR(ni, nn, 1, 1)]
    //                     + neuron_i_2[NEUR_IN_ADDR(ni, ny+1, nx+2)] * synapse_2[SYNAPSE_ADDR(ni, nn, 1, 2)]
    //                     + neuron_i_2[NEUR_IN_ADDR(ni, ny+2, nx+0)] * synapse_2[SYNAPSE_ADDR(ni, nn, 2, 0)]
    //                     + neuron_i_2[NEUR_IN_ADDR(ni, ny+2, nx+1)] * synapse_2[SYNAPSE_ADDR(ni, nn, 2, 1)]
    //                     + neuron_i_2[NEUR_IN_ADDR(ni, ny+2, nx+2)] * synapse_2[SYNAPSE_ADDR(ni, nn, 2, 2)];

    //                 // neuron_n[ny][nx][nn] = (
    //                 //           neuron_i_2[ni][ny+0][nx+0] * synapse_2[ni][nn][0][0]
    //                 //         + neuron_i_2[ni][ny+0][nx+1] * synapse_2[ni][nn][0][1]
    //                 //         + neuron_i_2[ni][ny+0][nx+2] * synapse_2[ni][nn][0][2]
    //                 //         + neuron_i_2[ni][ny+1][nx+0] * synapse_2[ni][nn][1][0]
    //                 //         + neuron_i_2[ni][ny+1][nx+1] * synapse_2[ni][nn][1][1]
    //                 //         + neuron_i_2[ni][ny+1][nx+2] * synapse_2[ni][nn][1][2]
    //                 //         + neuron_i_2[ni][ny+2][nx+0] * synapse_2[ni][nn][2][0]
    //                 //         + neuron_i_2[ni][ny+2][nx+1] * synapse_2[ni][nn][2][1]
    //                 //         + neuron_i_2[ni][ny+2][nx+2] * synapse_2[ni][nn][2][2]
    //                 // );
    //             }
    //         }
    //     }
    // }
}

int main(const int argc, const char** argv) {
    VTYPE *synapse_2 =  (VTYPE*)malloc(sizeof(VTYPE) * CONV1_SYNAPSE_SIZE);
    VTYPE *neuron_i_2 = (VTYPE*)malloc(sizeof(VTYPE) * CONV1_NEURON_INPUT_SIZE);
    VTYPE *neuron_n =   (VTYPE*)malloc(sizeof(VTYPE) * CONV1_NEURON_OUTPUT_SIZE);
    VTYPE *neuron_n_validation = (VTYPE*)malloc(sizeof(VTYPE) * CONV1_NEURON_OUTPUT_SIZE);
    for(int i=0; i<CONV1_SYNAPSE_SIZE; i++)
        synapse_2[i] = (float)rand() / (float)RAND_MAX;
    for(int i=0; i<CONV1_NEURON_INPUT_SIZE; i++)
        neuron_i_2[i] = (float)rand() / (float)RAND_MAX;
    
    const int X_DIM = 1;
    const int Y_DIM = 1;
    const int Z_DIM = 1024;
    dim3 gridDim(CONV1_NXSCL/X_DIM, CONV1_NYSCL/Y_DIM, CONV1_Nn/Z_DIM);
    dim3 blockDim(X_DIM, Y_DIM, Z_DIM);

    VTYPE *cuInput, *cuKernels, *cuOutput;
    CHECK_CUDA_ERROR(cudaMalloc(&cuInput, CONV1_NEURON_INPUT_SIZE*sizeof(VTYPE)));
    CHECK_CUDA_ERROR(cudaMalloc(&cuKernels, CONV1_SYNAPSE_SIZE*sizeof(VTYPE)));
    CHECK_CUDA_ERROR(cudaMalloc(&cuOutput, CONV1_NEURON_OUTPUT_SIZE*sizeof(VTYPE)));
    CHECK_CUDA_ERROR(cudaMemcpy(cuInput, neuron_i_2, CONV1_NEURON_INPUT_SIZE*sizeof(VTYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(cuKernels, synapse_2, CONV1_SYNAPSE_SIZE*sizeof(VTYPE), cudaMemcpyHostToDevice));

    convolution_layer_tiled_gpu<<<gridDim, blockDim>>>(cuKernels, cuInput, cuOutput);

    CHECK_CUDA_ERROR(cudaMemcpy(neuron_n, cuOutput, CONV1_NEURON_OUTPUT_SIZE*sizeof(VTYPE), cudaMemcpyDeviceToHost));
    
    convolution_layer(synapse_2, neuron_i_2, neuron_n_validation);
    // assert(is_gpu_cpu_arr_equal(neuron_n, neuron_n_validation, CONV1_NEURON_OUTPUT_SIZE));

    cudaFree(cuOutput);
    cudaFree(cuInput);
    cudaFree(cuKernels);

    // free(synapse_2);
    // free(neuron_i_2);
    // free(neuron_n);
    // free(neuron_n_validation);

    CHECK_LAST_CUDA_ERROR();
    cudaDeviceReset();
}


