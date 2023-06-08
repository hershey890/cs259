// #include <cstdio>
// #include <cstdlib>
// #include <vector>

// #include <cublas_v2.h>
// #include <cuda_runtime.h>
// #include <cusolverDn.h>

// #include "cusolver_utils.h"

// int main(int argc, char *argv[]) {
//     cusolverDnHandle_t cusolverH = NULL;
//     cublasHandle_t cublasH = NULL;
//     cudaStream_t stream = NULL;

//     const int m = 3;   /* 1 <= m <= 32 */
//     const int n = 2;   /* 1 <= n <= 32 */
//     const int lda = m; /* lda >= m */

//     /*
//      *       | 1 2 |
//      *   A = | 4 5 |
//      *       | 2 1 |
//      */

//     const std::vector<double> A = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
//     std::vector<double> U(lda * m, 0);  /* m-by-m unitary matrix, left singular vectors  */
//     std::vector<double> VT(lda * n, 0); /* n-by-n unitary matrix, right singular vectors */
//     std::vector<double> S(n, 0);        /* numerical singular value */
//     std::vector<double> S_exact = {7.065283497082729,
//                                    1.040081297712078}; /* exact singular values */
//     int info_gpu = 0;                                  /* host copy of error info */

//     double *d_A = nullptr;
//     double *d_S = nullptr;  /* singular values */
//     double *d_U = nullptr;  /* left singular vectors */
//     double *d_VT = nullptr; /* right singular vectors */
//     double *d_W = nullptr;  /* W = S*VT */

//     int *devInfo = nullptr;

//     int lwork = 0; /* size of workspace */
//     double *d_work = nullptr;
//     double *d_rwork = nullptr;

//     const double h_one = 1;
//     const double h_minus_one = -1;

//     std::printf("A = (matlab base-1)\n");
//     print_matrix(m, n, A.data(), lda);
//     std::printf("=====\n");

//     /* step 1: create cusolver handle, bind a stream */
//     CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
//     CUBLAS_CHECK(cublasCreate(&cublasH));

//     CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
//     CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

//     /* step 2: copy A to device */
//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(double) * S.size()));
//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(double) * U.size()));
//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_VT), sizeof(double) * VT.size()));
//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(double) * lda * n));
//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));

//     CUDA_CHECK(
//         cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));

//     /* step 3: query working space of SVD */
//     CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork));

//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

//     /* step 4: compute SVD*/
//     signed char jobu = 'A';  // all m columns of U
//     signed char jobvt = 'A'; // all n columns of VT
//     CUSOLVER_CHECK(cusolverDnDgesvd(cusolverH, jobu, jobvt, m, n, d_A, lda, d_S, d_U,
//                                     lda, // ldu
//                                     d_VT,
//                                     lda, // ldvt,
//                                     d_work, lwork, d_rwork, devInfo));

//     CUDA_CHECK(
//         cudaMemcpyAsync(U.data(), d_U, sizeof(double) * U.size(), cudaMemcpyDeviceToHost, stream));
//     CUDA_CHECK(cudaMemcpyAsync(VT.data(), d_VT, sizeof(double) * VT.size(), cudaMemcpyDeviceToHost,
//                                stream));
//     CUDA_CHECK(
//         cudaMemcpyAsync(S.data(), d_S, sizeof(double) * S.size(), cudaMemcpyDeviceToHost, stream));
//     CUDA_CHECK(cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));

//     CUDA_CHECK(cudaStreamSynchronize(stream));

//     std::printf("after gesvd: info_gpu = %d\n", info_gpu);
//     if (0 == info_gpu) {
//         std::printf("gesvd converges \n");
//     } else if (0 > info_gpu) {
//         std::printf("%d-th parameter is wrong \n", -info_gpu);
//         exit(1);
//     } else {
//         std::printf("WARNING: info = %d : gesvd does not converge \n", info_gpu);
//     }

//     std::printf("S = singular values (matlab base-1)\n");
//     print_matrix(n, 1, S.data(), n);
//     std::printf("=====\n");

//     std::printf("U = left singular vectors (matlab base-1)\n");
//     print_matrix(m, m, U.data(), lda);
//     std::printf("=====\n");

//     std::printf("VT = right singular vectors (matlab base-1)\n");
//     print_matrix(n, n, VT.data(), lda);
//     std::printf("=====\n");

//     // step 5: measure error of singular value
//     double ds_sup = 0;
//     for (int j = 0; j < n; j++) {
//         double err = fabs(S[j] - S_exact[j]);
//         ds_sup = (ds_sup > err) ? ds_sup : err;
//     }
//     std::printf("|S - S_exact| = %E \n", ds_sup);

//     CUBLAS_CHECK(cublasDdgmm(cublasH, CUBLAS_SIDE_LEFT, n, n, d_VT, lda, d_S, 1, d_W, lda));

//     CUDA_CHECK(
//         cudaMemcpyAsync(d_A, A.data(), sizeof(double) * lda * n, cudaMemcpyHostToDevice, stream));

//     CUBLAS_CHECK(cublasDgemm(cublasH,
//                              CUBLAS_OP_N,  // U
//                              CUBLAS_OP_N,  // W
//                              m,            // number of rows of A
//                              n,            // number of columns of A
//                              n,            // number of columns of U
//                              &h_minus_one, /* host pointer */
//                              d_U,          // U
//                              lda,
//                              d_W,         // W
//                              lda, &h_one, /* hostpointer */
//                              d_A, lda));

//     double dR_fro = 0.0;
//     CUBLAS_CHECK(cublasDnrm2(cublasH, lda * n, d_A, 1, &dR_fro));

//     std::printf("|A - U*S*VT| = %E \n", dR_fro);

//     /* free resources */
//     CUDA_CHECK(cudaFree(d_A));
//     CUDA_CHECK(cudaFree(d_U));
//     CUDA_CHECK(cudaFree(d_VT));
//     CUDA_CHECK(cudaFree(d_S));
//     CUDA_CHECK(cudaFree(d_W));
//     CUDA_CHECK(cudaFree(devInfo));
//     CUDA_CHECK(cudaFree(d_work));
//     CUDA_CHECK(cudaFree(d_rwork));

//     CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
//     CUBLAS_CHECK(cublasDestroy(cublasH));

//     CUDA_CHECK(cudaStreamDestroy(stream));

//     CUDA_CHECK(cudaDeviceReset());

//     return EXIT_SUCCESS;
// }











#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
// #include "cuda_common.h"
 
void printMatrix(float* A, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << A[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}
 
int main() {
    // Define input matrix A
    float A[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int m = 3;  // Number of rows
    int n = 2;  // Number of columns
 
    // Allocate memory on the GPU
    float* d_A;
    cudaMalloc((void**)&d_A, m * n * sizeof(float));
    cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
 
    // Create cuSOLVER handle
    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);
 
    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
 
    // Compute the size required for the workspace
    int lwork;
    cusolverDnSgesvd_bufferSize(cusolver_handle, m, n, &lwork);
 
    // Allocate workspace memory on the GPU
    float* d_work;
    cudaMalloc((void**)&d_work, lwork * sizeof(float));
 
    // Allocate memory for singular values on the GPU
    float* d_S;
    cudaMalloc((void**)&d_S, std::min(m, n) * sizeof(float));
 
    // Allocate memory for left singular vectors on the GPU
    float* d_U;
    cudaMalloc((void**)&d_U, m * m * sizeof(float));
 
    // Allocate memory for right singular vectors on the GPU
    float* d_VT;
    cudaMalloc((void**)&d_VT, n * n * sizeof(float));
 
    // Perform SVD
    int* dev_info;
    cudaMalloc((void**)&dev_info, sizeof(int));
    cusolverDnSgesvd(cusolver_handle, 'A', 'A', m, n, d_A, m, d_S, d_U, m, d_VT, n, d_work, lwork, nullptr, dev_info);
    int dev_info_host;
    cudaMemcpy(&dev_info_host, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
    if(dev_info_host != 0) {
        std::cout << "Unsuccessful SVD execution.\n" << std::endl;
        std::cout << "Error code " << dev_info_host << ".\n" << std::endl;
    }
 
    // Copy the results back to the host
    float S[std::min(m, n)];
    float U[m * m];
    float VT[n * n];
    cudaMemcpy(S, d_S, std::min(m, n) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, m * m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(VT, d_VT, n * n * sizeof(float), cudaMemcpyDeviceToHost);
 
    // Print the singular values
    std::cout << "Singular Values:" << std::endl;
    for (int i = 0; i < std::min(m, n); ++i) {
        std::cout << S[i] << " ";
    }
    std::cout << std::endl;
 
    // Print the left singular vectors
    std::cout << "Left Singular Vectors:" << std::endl;
}