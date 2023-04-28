#include <iostream>
#include <string>
#include "dnn.hpp"

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

#define CONV1_SYNAPSE_SIZE (1L*CONV1_Ky*CONV1_Kx*CONV1_Nn*CONV1_Ni)

VTYPE (*synapse)[Ky][Kx][Nn][Ni]; // Kernel

VTYPE  (*neuron_i)[NYPAD][NXPAD][Ni]; // Input Channel
VTYPE  (*neuron_n)[NYSCL][NXSCL][Nn]; // Output Channel 1
VTYPE (*neuron_n2)[NYSCL][NXSCL][Nn]; // Output Channel 2

void convolution_layer_base(VTYPE (&synapse)[CONV1_Ky][CONV1_Kx][CONV1_Nn][CONV1_Ni],
                              VTYPE (&neuron_i)[CONV1_NYPAD][CONV1_NXPAD][CONV1_Ni],
                              VTYPE (&neuron_n)[CONV1_NYSCL][CONV1_NXSCL][CONV1_Nn]) {
    for (int ny = 0; ny + CONV1_Ky < CONV1_Ny; ny += CONV1_Sy) {
        for (int nx = 0; nx CONV1_Kx < CONV1_Nx; nx += CONV1_Sx) {
            for (int ky = 0; ky < CONV1_Ky; ky++) {
                for (int kx = 0; kx < CONV1_Kx; kx++) {
                    for (int ni = 0; ni < Ni; ni++) {
                        for (int nn = 0; nn < Nn; nn++) {
                            neuron_n[ny][nx][nn] += neuron_i[ny + ky][nx + kx][ni] * synapse[ky][kx][nn][ni];
                        }
                    }
                }
            }
        }
    }
}

// TODO - Implement Base GPU Version (no tiling, parallelized)
__global__ void convolution_layer_base_gpu(VTYPE (&synapse)[CONV1_Ky][CONV1_Kx][CONV1_Nn][CONV1_Ni],
                                           VTYPE (&neuron_i)[CONV1_NYPAD][CONV1_NXPAD][CONV1_Ni],
                                           VTYPE (&neuron_n)[CONV1_NYSCL][CONV1_NXSCL][CONV1_Nn]) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    
}

// TODO - Implement GPU Version (tiled)

int main(const int argc, const char** argv) {
    synapse   = (VTYPE (*)[CONV1_Ky][CONV1_Kx][CONV1_Nn][CONV1_Ni])  aligned_malloc(64,  CONV1_SYNAPSE_SIZE*sizeof(VTYPE));
    neuron_i  = (VTYPE (*)[CONV1_NYPAD][CONV1_NXPAD][CONV1_Ni])aligned_malloc(64,CONV1_NYPAD*CONV1_NXPAD*CONV1_Ni*sizeof(VTYPE));
    neuron_n  = (VTYPE (*)[CONV1_NYSCL][CONV1_NXSCL][CONV1_Nn])aligned_malloc(64,CONV1_NYSCL*CONV1_NXSCL*CONV1_Nn*sizeof(VTYPE));
    neuron_n2 = (VTYPE (*)[CONV1_NYSCL][CONV1_NXSCL][CONV1_Nn])aligned_malloc(64,CONV1_NYSCL*CONV1_NXSCL*CONV1_Nn*sizeof(VTYPE));
}


