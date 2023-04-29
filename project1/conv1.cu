#include <iostream>
#include <string>

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

#define CONV1_NYSCL ((CONV1_Ny - CONV1_Ky + 1)/CONV1_Sy) // #define CONV1_NYSCL (CONV1_Ny/CONV1_Sy)
#define CONV1_NXSCL ((CONV1_Nx - CONV1_Kx + 1)/CONV1_Sx) // #define CONV1_NXSCL (CONV1_Nx/CONV1_Sx)

#define CONV1_SYNAPSE_SIZE (CONV1_Ky*CONV1_Kx*CONV1_Nn*CONV1_Ni)
#define CONV1_NEURON_INPUT_SIZE (CONV1_NYPAD*CONV1_NXPAD*CONV1_Ni)
#define CONV1_NEURON_OUTPUT_SIZE (CONV1_NYSCL*CONV1_NXSCL*CONV1_Nn)

#define CONV1_THREADS 1024
#define CONV1_BLOCKS 500

#define CONV1_ROWS_PROCESSED ((CONV1_NYSCL)/(CONV1_THREADS*CONV1_BLOCKS*CONV1_Sy))
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
    const int end_row = min(CONV1_NYSCL, start_row + CONV1_ROWS_PROCESSED);
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

// GPU Version (unrolled kernel + re-shaped data layouts for input and synapse)
__global__ void convolution_layer_tiled_gpu(VTYPE (&synapse_2)[CONV1_Ni][CONV1_Nn][CONV1_Ky][CONV1_Kx],
                                           VTYPE (&neuron_i_2)[CONV1_Ni][CONV1_NYPAD][CONV1_NXPAD],
                                           VTYPE (&neuron_n)[CONV1_NYSCL][CONV1_NXSCL][CONV1_Nn]) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    const int start_row = x * CONV1_ROWS_PROCESSED;
    const int end_row = min(CONV1_NYSCL, start_row + CONV1_ROWS_PROCESSED);
    for (int ny = start_row; ny < end_row; ny += CONV1_Sy) {
        for (int nx = 0; nx + CONV1_Kx < CONV1_Nx; nx += CONV1_Sx) {
            for (int ni = 0; ni < CONV1_Ni; ni++) {
                for (int nn = 0; nn < CONV1_Nn; nn++) {
                    neuron_n[ny][nx][nn] = (
                            neuron_i_2[ni][ny+0][nx+0] * synapse_2[ni][nn][0][0]
                            + neuron_i_2[ni][ny+0][nx+1] * synapse_2[ni][nn][0][1]
                            + neuron_i_2[ni][ny+0][nx+2] * synapse_2[ni][nn][0][2]
                            + neuron_i_2[ni][ny+1][nx+0] * synapse_2[ni][nn][1][0]
                            + neuron_i_2[ni][ny+1][nx+1] * synapse_2[ni][nn][1][1]
                            + neuron_i_2[ni][ny+1][nx+2] * synapse_2[ni][nn][1][2]
                            + neuron_i_2[ni][ny+2][nx+0] * synapse_2[ni][nn][2][0]
                            + neuron_i_2[ni][ny+2][nx+1] * synapse_2[ni][nn][2][1]
                            + neuron_i_2[ni][ny+2][nx+2] * synapse_2[ni][nn][2][2]
                    );
                }
            }
        }
    }
}

int main(const int argc, const char** argv) {
//     synapse   = (VTYPE (*)[CONV1_Ky][CONV1_Kx][CONV1_Nn][CONV1_Ni])  malloc(CONV1_SYNAPSE_SIZE*sizeof(VTYPE));
//     neuron_i  = (VTYPE (*)[CONV1_NYPAD][CONV1_NXPAD][CONV1_Ni])malloc(CONV1_NYPAD*CONV1_NXPAD*CONV1_Ni*sizeof(VTYPE))
    synapse_2   = (VTYPE (*)[CONV1_Ni][CONV1_Nn][CONV1_Ky][CONV1_Kx])  malloc(CONV1_SYNAPSE_SIZE*sizeof(VTYPE));
    neuron_i_2  = (VTYPE (*)[CONV1_Ni][CONV1_NYPAD][CONV1_NXPAD])malloc(CONV1_NYPAD*CONV1_NXPAD*CONV1_Ni*sizeof(VTYPE));
    neuron_n  = (VTYPE (*)[CONV1_NYSCL][CONV1_NXSCL][CONV1_Nn])malloc(CONV1_NYSCL*CONV1_NXSCL*CONV1_Nn*sizeof(VTYPE));
    // neuron_n2 = (VTYPE (*)[CONV1_NYSCL][CONV1_NXSCL][CONV1_Nn])aligned_malloc(64,CONV1_NYSCL*CONV1_NXSCL*CONV1_Nn*sizeof(VTYPE));
    float *cuInput, *cuKernels, *cuOutput;
    cudaMalloc(&cuInput, CONV1_NEURON_INPUT_SIZE*sizeof(VTYPE));
    cudaMalloc(&cuKernels, CONV1_SYNAPSE_SIZE*sizeof(VTYPE));
    cudaMalloc(&cuOutput, CONV1_NEURON_OUTPUT_SIZE*sizeof(VTYPE));
    cudaMemcpy(cuInput, neuron_i, CONV1_NEURON_INPUT_SIZE*sizeof(VTYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(cuKernels, synapse, CONV1_SYNAPSE_SIZE*sizeof(VTYPE), cudaMemcpyHostToDevice);

    convolution_layer_tiled_gpu<<<CONV1_BLOCKS, CONV1_THREADS>>>(*synapse_2, *neuron_i_2, *neuron_n);
}


