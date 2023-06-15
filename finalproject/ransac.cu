#include <iostream>
#include <fstream>
#include <cublas_v2.h>

void readPtsFile(std::string filename, float** src, float** dst, uint32_t* n_bytes) {
    std::ifstream stream(filename, std::ios::in | std::ios::binary);

    stream.read(reinterpret_cast<char *>(n_bytes), sizeof(uint32_t));

    *src = new (std::nothrow) float[*n_bytes / sizeof(float)];
    *dst = new (std::nothrow) float[*n_bytes / sizeof(float)];
    stream.read(reinterpret_cast<char *>(*src), *n_bytes);
    stream.read(reinterpret_cast<char *>(*dst), *n_bytes);

    stream.close();
}

// Source: https://github.com/pradyotsn/Matrix-Inverse-in-CUDA/blob/master/mat_inv.cu
void invert(float** src, float** dst, int n, int batchSize)
{
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    int *P, *INFO;

    cudaMalloc(&P, n * batchSize * sizeof(int));
    cudaMalloc(&INFO,  batchSize * sizeof(int));

    int lda = n;

    float **A = (float **)malloc(batchSize*sizeof(float *));
    float **A_d, *A_dflat;

    cudaMalloc(&A_d,batchSize*sizeof(float *));
    cudaMalloc(&A_dflat, n*n*batchSize*sizeof(float));

    A[0] = A_dflat;
    for (int i = 1; i < batchSize; i++)
        A[i] = A[i-1]+(n*n);

    cudaMemcpy(A_d,A,batchSize*sizeof(float *),cudaMemcpyHostToDevice);

    for (int i = 0; i < batchSize; i++)
        cudaMemcpy(A_dflat+(i*n*n), src[i], n*n*sizeof(float), cudaMemcpyHostToDevice);


    cublasSgetrfBatched(handle,n,A_d,lda,P,INFO,batchSize);


    int INFOh[batchSize];
    cudaMemcpy(INFOh,INFO,batchSize*sizeof(int),cudaMemcpyDeviceToHost);

    for (int i = 0; i < batchSize; i++)
        if(INFOh[i]  != 0)
        {
            fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }

    float **C = (float **)malloc(batchSize*sizeof(float *));
    float **C_d, *C_dflat;

    cudaMalloc(&C_d,batchSize*sizeof(float *));
    cudaMalloc(&C_dflat, n*n*batchSize*sizeof(float));
    C[0] = C_dflat;
    for (int i = 1; i < batchSize; i++)
        C[i] = C[i-1] + (n*n);
    cudaMemcpy(C_d,C,batchSize*sizeof(float *),cudaMemcpyHostToDevice);
    cublasSgetriBatched(handle,n,(const float **)A_d,lda,P,C_d,lda,INFO,batchSize);

    cudaMemcpy(INFOh,INFO,batchSize*sizeof(int),cudaMemcpyDeviceToHost);

    for (int i = 0; i < batchSize; i++)
        if(INFOh[i] != 0)
        {
            fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", i);
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
    for (int i = 0; i < batchSize; i++)
        cudaMemcpy(dst[i], C_dflat + (i*n*n), n*n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_d); cudaFree(A_dflat); free(A);
    cudaFree(C_d); cudaFree(C_dflat); free(C);
    cudaFree(P); cudaFree(INFO); cublasDestroy_v2(handle);
}

void linearRegressorFit(float* X, float* y, float* params, uint32_t N) {
    // Pad X on the top with ones
    float *cublasXPadded;
    float *ones = new float[N];
    for (int i = 0; i < N; i++)
        ones[i] = 1.0f;
    cudaMalloc(&cublasXPadded, sizeof(float) * N * 2);
    cudaMemcpy(cublasXPadded, ones, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cublasXPadded + N, X, sizeof(float) * N, cudaMemcpyHostToDevice);
    delete[] ones;

//    std::cout << "y:";
//    for (int i = 0; i < N; i++) {
//        std::cout << ' ' << y[i];
//    }
//    std::cout << '\n';

//    float *padded = new float[N * 2];
//    cudaMemcpy(padded, cublasXPadded, sizeof(float) * N * 2, cudaMemcpyDeviceToHost);
//    std::cout << "Padded:";
//    for (int i = 0; i < N * 2; i++) {
//        std::cout << ' ' << padded[i];
//    }
//    std::cout << '\n';

    // Perform outer product of cublasMatMul1 = cublasXPadded.T x cublasXPadded
    float *cublasMatMul1;
    const float alpha = 1.0f; const float beta = 1.0f;
    cudaMalloc(&cublasMatMul1, sizeof(float) * 4);
    cudaMemset(cublasMatMul1, 0, sizeof(float) * 4);
    cublasHandle_t handle = 0;
    cublasCreate(&handle);
    cublasSgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            2, 2, N,
            &alpha,
            cublasXPadded, N,
            cublasXPadded, N,
            &beta,
            cublasMatMul1, 2
    );

//    float* matMul1 = new float[4];
//    cudaMemcpy(matMul1, cublasMatMul1, sizeof(float) * 4, cudaMemcpyDeviceToHost);
//    std::cout << "matMul1:";
//    for (int i = 0; i < 4; i++) {
//        std::cout << ' ' << matMul1[i];
//    }
//    std::cout << '\n';

    // Invert X.T x X
    float **srcMatrix, **dstMatrix;
    srcMatrix = new float*[1];
    dstMatrix = new float*[1];
    srcMatrix[0] = new float[4];
    dstMatrix[0] = new float[4];
    cudaMemcpy(srcMatrix[0], cublasMatMul1, sizeof(float) * 4, cudaMemcpyDeviceToHost);
    invert(srcMatrix, dstMatrix, 2, 1);

//    std::cout << "Inverted matrix:";
//    for (int i = 0; i < 4; i++) {
//        std::cout << ' ' << dstMatrix[0][i];
//    }
//    std::cout << '\n';

    // Multiply inverse(X.T x X) by X.T
    float *cublasInverseMat, *cublasMatMul2;
    cudaMalloc(&cublasInverseMat, sizeof(float) * 4);
    cudaMalloc(&cublasMatMul2, sizeof(float) * N * 2);
    cudaMemcpy(cublasInverseMat, dstMatrix[0], sizeof(float) * 4, cudaMemcpyHostToDevice);
    cudaMemset(cublasMatMul2, 0, sizeof(float) * N * 2);
    cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            2, N, 2,
            &alpha,
            cublasInverseMat, 2,
            cublasXPadded, N,
            &beta,
            cublasMatMul2, 2
    ); // Gets the transpose of the answer?

//    float* matMul2 = new float[N * 2];
//    cudaMemcpy(matMul2, cublasMatMul2, sizeof(float) * N * 2, cudaMemcpyDeviceToHost);
//    std::cout << "MatMul2:";
//    for (int i = 0; i < N * 2; i++) {
//        std::cout << ' ' << matMul2[i];
//    }
//    std::cout << '\n';

    // Multiply (inverse(X.T x X) by X.T) by y
    float *cublasY, *cublasMatMul3;
    cudaMalloc(&cublasY, sizeof(float) * N);
    cudaMalloc(&cublasMatMul3, sizeof(float) * 2);
    cublasSetVector(N, sizeof(float), y, 1, cublasY, 1);
    cudaMemset(cublasMatMul3, 0, sizeof(float) * 2);
    cublasSgemv(
            handle, CUBLAS_OP_N,
            2, N,
            &alpha,
            cublasMatMul2, 2,
            cublasY, 1,
            &beta,
            cublasMatMul3, 1
    );

//    float* matMul3 = new float[2];
//    cudaMemcpy(matMul3, cublasMatMul3, sizeof(float) * 2, cudaMemcpyDeviceToHost);
//    std::cout << "MatMul3:";
//    for (int i = 0; i < 2; i++) {
//        std::cout << ' ' << matMul3[i];
//    }
//    std::cout << '\n';

    cudaMemcpy(params, cublasMatMul3, sizeof(float) * 2, cudaMemcpyDeviceToHost);

    // Deletes
    delete srcMatrix[0];
    delete dstMatrix[0];
    delete srcMatrix;
    delete dstMatrix;

    // Cuda Frees
    cudaFree(cublasXPadded);
    cudaFree(cublasMatMul1);
    cudaFree(cublasMatMul2);
    cudaFree(cublasMatMul3);
    cudaFree(cublasY);
    cudaFree(cublasInverseMat);

    // Cublas Destroy Handle
    cublasDestroy(handle);
}

void linearRegressorPredict(float* X, float* params, uint32_t N) {
    // Pad X on the top with ones
    float *cublasXPadded;
    float *ones = new float[N];
    for (int i = 0; i < N; i++)
        ones[i] = 1.0f;
    cudaMalloc(&cublasXPadded, sizeof(float) * N * 2);
    cudaMemcpy(cublasXPadded, ones, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cublasXPadded + N, X, sizeof(float) * N, cudaMemcpyHostToDevice);
    delete[] ones;

//    float *padded = new float[N * 2];
//    cudaMemcpy(padded, cublasXPadded, sizeof(float) * N * 2, cudaMemcpyDeviceToHost);
//    std::cout << "Padded:";
//    for (int i = 0; i < N * 2; i++) {
//        std::cout << ' ' << padded[i];
//    }
//    std::cout << '\n';
//
//    std::cout << "Params:";
//    for (int i = 0; i < 2; i++) {
//        std::cout << ' ' << params[i];
//    }
//    std::cout << '\n';

    // Perform the prediction calculation
    float *cublasParams, *cublasMatMul;
    const float alpha = 1.0f; const float beta = 1.0f;
    cudaMalloc(&cublasParams, sizeof(float) * 2);
    cudaMalloc(&cublasMatMul, sizeof(float) * N);
    cudaMemcpy(cublasParams, params, sizeof(float) * 2, cudaMemcpyHostToDevice);
    cudaMemset(cublasMatMul, 0, sizeof(float) * N);
    cublasHandle_t handle = 0;
    cublasCreate(&handle);
    cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            1, N, 2,
            &alpha,
            cublasParams, 1,
            cublasXPadded, N,
            &beta,
            cublasMatMul, 1
    );

//    float *matMul = new float[N];
//    cudaMemcpy(matMul, cublasMatMul, sizeof(float) * N, cudaMemcpyDeviceToHost);
//    std::cout << "MatMul:";
//    for (int i = 0; i < N; i++) {
//        std::cout << ' ' << matMul[i];
//    }
//    std::cout << '\n';

    // Cuda Frees
    cudaFree(cublasXPadded);
    cudaFree(cublasParams);
    cudaFree(cublasMatMul);

    // Cublas Destroy Handle
    cublasDestroy(handle);
}

int main()
{
    float *src, *dst;
    uint32_t n_bytes;
    readPtsFile("./data/src_dst_pts.bin", &src, &dst, &n_bytes);

//    float **inv_src, **inv_dst;
//    inv_src = new float*[1];
//    inv_dst = new float*[1];
//    inv_src[0] = new float[9];
//    inv_dst[0] = new float[9];
//    inv_src[0][0] = inv_src[0][4] = inv_src[0][8] = 2.0f;
//    invert(inv_src, inv_dst, 3, 1);

    float *X, *y, *params;
    X = new float[3];
    y = new float[3];
    params = new float[2];
    X[0] = 1; X[1] = 2; X[2] = 3;
    y[0] = 7; y[1] = 8; y[2] = 9;
    linearRegressorFit(X, y, params, 3);
    linearRegressorPredict(X, params, 3);
}
