#include <boost/asio.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/assert.hpp>
#include <boost/bind/bind.hpp>
#include <boost/iterator/permutation_iterator.hpp>
#include <cmath>
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <numeric>
#include <random>

#define LIN_REG_PARAMS_DIM 2
#define THREAD_COUNT 1

struct RansacFitResult {
    float error;
    float params[LIN_REG_PARAMS_DIM];
};

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
            std::cout << "Matrix:";
            for (int r = 0; r < n; r++) {
                for (int c = 0; c < n; c++) {
                    std::cout << ' ' << src[i][r * n + c];
                }
            }
            std::cout << '\n';
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
//    delete matMul2;

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
//    delete matMaul3;

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

void linearRegressorPredict(float* X, float* params, float* result, uint32_t N) {
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
//    delete padded;

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
//    delete matMul;

    cudaMemcpy(result, cublasMatMul, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Cuda Frees
    cudaFree(cublasXPadded);
    cudaFree(cublasParams);
    cudaFree(cublasMatMul);

    // Cublas Destroy Handle
    cublasDestroy(handle);
}

// Assumes that LinReg Fit+Predict is used.
// Assumes that metric is (y_hat - y)^2.
// TODO: Tan - maybe we can parallelize the for loop body?
void ransacKernel(const float* const X, const float* const y, const uint n, const float t, const uint d, const uint N, RansacFitResult* const fitResult) {
    // Generate random indices to use - Boost uses std under the hood.
    std::vector<unsigned int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);


    // Set the X and y according to randomly sampled indices
    float* XLin = new float[n];
    float* yLin = new float[n];
    for (int i = 0; i < n; i++) {
        XLin[i] = X[indices[i]];
        yLin[i] = y[indices[i]];
    }
    float *initialParams = new float[LIN_REG_PARAMS_DIM];

    // Fit the linear regressor to our model
    linearRegressorFit(XLin, yLin, initialParams, n);

    // Perform prediction
    float* yLinPred = new float[n];
    linearRegressorPredict(XLin, initialParams, yLinPred, n);

    // Delete array
    delete initialParams;

    // Get thresholded indices
    std::vector<float> XLinThreshold;
    std::vector<float> yLinThreshold;
    for (int i = 0; i < n; i++) {
        // Check to see if square error loss is below the set threshold, (if so, include as inlier point)
        if (pow(yLin[i] - yLinPred[i], 2) < t) {
            XLinThreshold.push_back(XLin[i]);
            yLinThreshold.push_back(yLin[i]);
        }
    }

    // Delete arrays
    delete XLin;
    delete yLin;
    delete yLinPred;

    // Sanity check assertion that dimensions match between X and y inliers and get the count
    BOOST_ASSERT(XLinThreshold.size() == yLinThreshold.size());
    const uint numInliers = XLinThreshold.size();

    // Number of inlier points is too low so we discard this from consideration
    if (numInliers <= d) {
        fitResult->error = std::numeric_limits<float>::max();
        return;
    }

    // Create a model on the inliers & set the params
    linearRegressorFit(&XLinThreshold[0], &yLinThreshold[0], fitResult->params, numInliers);

    // Form a prediction from the inliers
    float* yLinThresholdPred = new float[numInliers];
    linearRegressorPredict(&XLinThreshold[0], fitResult->params, yLinThresholdPred, numInliers);

    // Compute the mean squared error
    float squareErrorAccumulator = 0;
    for (int i = 0; i < numInliers; i++) {
        squareErrorAccumulator += pow(yLinThresholdPred[i] - yLinThreshold[i], 2);
    }

    // Set the error
    fitResult->error = squareErrorAccumulator / numInliers;

    // Delete array
    delete yLinThresholdPred;
}

/**
 * Perform ransac fit using Linear Regression Fit and Predict as model and square difference as metric.
 * @param X 1-d X array
 * @param y 1-d y array
 * @param n Minimum number of data points to estimate parameters
 * @param k Maximum iterations allowed
 * @param t Threshold value to determine if points are fit well
 * @param d Number of close data points required to assert model fits well
 * @param N Length of X & y
 * @param bestFitResult the best fit result (measured by lowest loss) returned by RANSAC
 */
void ransacFit(float* X, float* y, uint n, uint k, float t, uint d, uint N, RansacFitResult* bestFitResult) {
    boost::asio::thread_pool pool(THREAD_COUNT);
    RansacFitResult* fitResults = new RansacFitResult[k];
    for (int i = 0; i < k; i++) {
        boost::asio::post(
            pool,
            boost::bind(
                ransacKernel,
                X, y, n, t, d, N, fitResults + i
            )
        );
    }

    pool.join();

    bestFitResult->error = std::numeric_limits<float>::max();
    for (int i = 0; i < k; i++) {
        if (fitResults[i].error < bestFitResult->error) {
            bestFitResult->error = fitResults[i].error;
            for (int j = 0; j < LIN_REG_PARAMS_DIM; j++) {
                bestFitResult->params[j] = fitResults[i].params[j];
            }
        }
    }

    delete fitResults;
}

int main()
{
    float *X, *y;
    uint32_t n_bytes;
    readPtsFile("./data/src_dst_pts.bin", &X, &y, &n_bytes);

//    float **inv_src, **inv_dst;
//    inv_src = new float*[1];
//    inv_dst = new float*[1];
//    inv_src[0] = new float[9];
//    inv_dst[0] = new float[9];
//    inv_src[0][0] = inv_src[0][4] = inv_src[0][8] = 2.0f;
//    invert(inv_src, inv_dst, 3, 1);

//    float *params, *result;
//    params = new float[LIN_REG_PARAMS_DIM];
//    result = new float[N];
//    linearRegressorFit(X, y, params, N);
//    linearRegressorPredict(X, params, result, N);
//
//    std::cout << "X:";
//    for (int i = 0; i < N; i++)
//        std::cout << ' ' << X[i];
//    std::cout << '\n';
//
//    std::cout << "y:";
//    for (int i = 0; i < N; i++)
//        std::cout << ' ' << y[i];
//    std::cout << '\n';
//
//    std::cout << "params:";
//    for (int i = 0; i < 2; i++)
//        std::cout << ' ' << params[i];
//    std::cout << '\n';
//
//    std::cout << "result:";
//    for (int i = 0; i < N; i++)
//        std::cout << ' ' << result[i];
//    std::cout << '\n';
//
//    delete params;
//    delete result;

    const int N = (int) (n_bytes / sizeof(float)); // Number of X/y data points
    const int n = 100; // Minimum number of data points to estimate parameters
    const int k = 10000; // Maximum number of iterations allowed
    const float t = 10.0; // Threshold value to determine if points are fit well
    const int d = 80; // Number of close data points required to assert model fits
    RansacFitResult bestFitResult;
    ransacFit(
        X, y, n, k, t, d, N, &bestFitResult
    );

    std::cout << "Params: (" << bestFitResult.params[0] << ',' << bestFitResult.params[1] << ")\n";
    std::cout << "MSE: " << bestFitResult.error << '\n';

    delete X;
    delete y;
}
