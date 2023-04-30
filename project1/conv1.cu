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

#define CONV1_ROWS_PROCESSED ((CONV1_Ny-CONV1_Ky)/(CONV1_THREADS*CONV1_BLOCKS))
// #define CONV1_COLS_PROCESSED ((CONV1_NXSCL)/(CONV1_THREADS*CONV1_BLOCKS))

using VTYPE = float;

VTYPE (*synapse)[CONV1_Ky][CONV1_Kx][CONV1_Nn][CONV1_Ni]; // Kernel
VTYPE (*synapse_2)[CONV1_Ni][CONV1_Nn][CONV1_Ky][CONV1_Kx]; // Kernel re-shaped

VTYPE  (*neuron_i)[CONV1_NYPAD][CONV1_NXPAD][CONV1_Ni]; // Input Channel
VTYPE  (*neuron_i_2)[CONV1_Ni][CONV1_NYPAD][CONV1_NXPAD]; // Input Channel re-shaped
VTYPE  (*neuron_n)[CONV1_NYSCL][CONV1_NXSCL][CONV1_Nn]; // Output Channel 1
VTYPE  (*neuron_n2)[CONV1_NYSCL][CONV1_NXSCL][CONV1_Nn]; // Output Channel 2

// Base CPU Version (no optimizations)
void convolution_layer_base(VTYPE (&synapse)[CONV1_Ky][CONV1_Kx][CONV1_Nn][CONV1_Ni],
                              VTYPE (&neuron_i)[CONV1_NYPAD][CONV1_NXPAD][CONV1_Ni],
                              VTYPE (&neuron_n)[CONV1_NYSCL][CONV1_NXSCL][CONV1_Nn]) {
    for (int ny = 0; ny + CONV1_Ky < CONV1_Ny; ny += CONV1_Sy) {
        for (int nx = 0; nx + CONV1_Kx < CONV1_Nx; nx += CONV1_Sx) {
            for (int ky = 0; ky < CONV1_Ky; ky++) {
                for (int kx = 0; kx < CONV1_Kx; kx++) {
                    for (int ni = 0; ni < CONV1_Ni; ni++) {
                        for (int nn = 0; nn < CONV1_Nn; nn++) {
                            neuron_n[ny][nx][nn] += neuron_i[ny + ky][nx + kx][ni] * synapse[ky][kx][nn][ni];
                        }
                    }
                }
            }
        }
    }
}

// Base GPU Version (no tiling, 1-d parallelized)
__global__ void convolution_layer_base_gpu(VTYPE (&synapse)[CONV1_Ky][CONV1_Kx][CONV1_Nn][CONV1_Ni],
                                           VTYPE (&neuron_i)[CONV1_NYPAD][CONV1_NXPAD][CONV1_Ni],
                                           VTYPE (&neuron_n)[CONV1_NYSCL][CONV1_NXSCL][CONV1_Nn]) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    const int start_row = x * CONV1_ROWS_PROCESSED;
    const int end_row = min(CONV1_Ky, start_row + CONV1_ROWS_PROCESSED);
    for (int ny = start_row; ny < end_row; ny += CONV1_Sy) {
        for (int nx = 0; nx + CONV1_Kx < CONV1_Nx; nx += CONV1_Sx) {
            for (int ky = 0; ky < CONV1_Ky; ky++) {
                for (int kx = 0; kx < CONV1_Kx; kx++) {
                    for (int ni = 0; ni < CONV1_Ni; ni++) {
                        for (int nn = 0; nn < CONV1_Nn; nn++) {
                            neuron_n[ny][nx][nn] += neuron_i[ny + ky][nx + kx][ni] * synapse[ky][kx][nn][ni];
                        }
                    }
                }
            }
        }
    }
}


#define NEUR_OUT_ADDR(ny, nx, nn) (CONV1_Nx*CONV1_Nn*(ny) + CONV1_Nn*(nx) + (nn))
#define NEUR_IN_ADDR(ni, ny, nx) (CONV1_Ny*CONV1_Nx*(ni) + CONV1_Nx*(ny) + (nx))
#define SYNAPSE_ADDR(ni, nn, ky, kx) (CONV1_Nn * CONV1_Ky*CONV1_Kx*(ni) + CONV1_Ky*CONV1_Kx*(nn) + CONV1_Kx*(ky) + (kx))


// GPU Version (unrolled kernel + re-shaped data layouts for input and synapse)
// __global__ void convolution_layer_tiled_gpu(VTYPE synapse_2[CONV1_Ni][CONV1_Nn][CONV1_Ky][CONV1_Kx],
//                                            VTYPE neuron_i_2[CONV1_Ni][CONV1_NYPAD][CONV1_NXPAD],
//                                            VTYPE neuron_n[CONV1_NYSCL][CONV1_NXSCL][CONV1_Nn]) {
__global__ void convolution_layer_tiled_gpu(VTYPE *synapse_2, VTYPE *neuron_i_2, VTYPE *neuron_n) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    VTYPE sum = 0;
    // neuron_n[NEUR_OUT_ADDR(x, y, z)] = 0;

    for (int ni = 0; ni < CONV1_Ni; ni++) {
        //  neuron_n[NEUR_OUT_ADDR(x, y, z)] += 
        sum +=           neuron_i_2[NEUR_IN_ADDR(ni, y+0, x+0)] * synapse_2[SYNAPSE_ADDR(ni, z, 0, 0)]
                        + neuron_i_2[NEUR_IN_ADDR(ni, y+0, x+1)] * synapse_2[SYNAPSE_ADDR(ni, z, 0, 1)]
                        + neuron_i_2[NEUR_IN_ADDR(ni, y+0, x+2)] * synapse_2[SYNAPSE_ADDR(ni, z, 0, 2)]
                        + neuron_i_2[NEUR_IN_ADDR(ni, y+1, x+0)] * synapse_2[SYNAPSE_ADDR(ni, z, 1, 0)]
                        + neuron_i_2[NEUR_IN_ADDR(ni, y+1, x+1)] * synapse_2[SYNAPSE_ADDR(ni, z, 1, 1)]
                        + neuron_i_2[NEUR_IN_ADDR(ni, y+1, x+2)] * synapse_2[SYNAPSE_ADDR(ni, z, 1, 2)]
                        + neuron_i_2[NEUR_IN_ADDR(ni, y+2, x+0)] * synapse_2[SYNAPSE_ADDR(ni, z, 2, 0)]
                        + neuron_i_2[NEUR_IN_ADDR(ni, y+2, x+1)] * synapse_2[SYNAPSE_ADDR(ni, z, 2, 1)]
                        + neuron_i_2[NEUR_IN_ADDR(ni, y+2, x+2)] * synapse_2[SYNAPSE_ADDR(ni, z, 2, 2)];

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
    neuron_n[NEUR_OUT_ADDR(x, y, z)] = sum;
}

int main(const int argc, const char** argv) {
    VTYPE *synapse_2 =  (VTYPE*)malloc(sizeof(VTYPE) * CONV1_SYNAPSE_SIZE);
    VTYPE *neuron_i_2 = (VTYPE*)malloc(sizeof(VTYPE) * CONV1_NEURON_INPUT_SIZE);
    VTYPE *neuron_n =   (VTYPE*)malloc(sizeof(VTYPE) * CONV1_NEURON_OUTPUT_SIZE);

//     synapse   = (VTYPE (*)[CONV1_Ky][CONV1_Kx][CONV1_Nn][CONV1_Ni])  malloc(CONV1_SYNAPSE_SIZE*sizeof(VTYPE));
//     neuron_i  = (VTYPE (*)[CONV1_NYPAD][CONV1_NXPAD][CONV1_Ni])malloc(CONV1_NYPAD*CONV1_NXPAD*CONV1_Ni*sizeof(VTYPE))
    // synapse_2   = (VTYPE (*)[CONV1_Ni][CONV1_Nn][CONV1_Ky][CONV1_Kx])  malloc(CONV1_SYNAPSE_SIZE*sizeof(VTYPE));
    // neuron_i_2  = (VTYPE (*)[CONV1_Ni][CONV1_NYPAD][CONV1_NXPAD])malloc(CONV1_NYPAD*CONV1_NXPAD*CONV1_Ni*sizeof(VTYPE));
    // neuron_n  = (VTYPE (*)[CONV1_NYSCL][CONV1_NXSCL][CONV1_Nn])malloc(CONV1_NYSCL*CONV1_NXSCL*CONV1_Nn*sizeof(VTYPE));
    // neuron_n2 = (VTYPE (*)[CONV1_NYSCL][CONV1_NXSCL][CONV1_Nn])aligned_malloc(64,CONV1_NYSCL*CONV1_NXSCL*CONV1_Nn*sizeof(VTYPE));
    const int X_DIM = 32;
    const int Y_DIM = 8;
    const int Z_DIM = 4;
    dim3 gridDim(CONV1_NXSCL/X_DIM, CONV1_NYSCL/Y_DIM, CONV1_Nn/Z_DIM);
    dim3 blockDim(X_DIM, Y_DIM, Z_DIM);

    float *cuInput, *cuKernels, *cuOutput;
    CHECK_CUDA_ERROR(cudaMalloc(&cuInput, CONV1_NEURON_INPUT_SIZE*sizeof(VTYPE)));
    CHECK_CUDA_ERROR(cudaMalloc(&cuKernels, CONV1_SYNAPSE_SIZE*sizeof(VTYPE)));
    CHECK_CUDA_ERROR(cudaMalloc(&cuOutput, CONV1_NEURON_OUTPUT_SIZE*sizeof(VTYPE)));
    CHECK_CUDA_ERROR(cudaMemcpy(cuInput, neuron_i_2, CONV1_NEURON_INPUT_SIZE*sizeof(VTYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(cuKernels, synapse_2, CONV1_SYNAPSE_SIZE*sizeof(VTYPE), cudaMemcpyHostToDevice));

    convolution_layer_tiled_gpu<<<gridDim, blockDim>>>(cuKernels, cuInput, cuOutput);

    CHECK_CUDA_ERROR(cudaMemcpy(neuron_n, cuOutput, CONV1_NEURON_OUTPUT_SIZE*sizeof(VTYPE), cudaMemcpyDeviceToHost));

    cudaFree(cuOutput);
    cudaFree(cuInput);
    cudaFree(cuKernels);

    free(synapse_2);
    free(neuron_i_2);
    free(neuron_n);

    CHECK_LAST_CUDA_ERROR();
    cudaDeviceReset();
}


