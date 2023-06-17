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
#define THREAD_COUNT 12

struct RansacFitResult {
    double error;
    double params[LIN_REG_PARAMS_DIM];
};

void readPtsFile(std::string filename, double** src, double** dst, uint32_t* n_bytes) {
    std::ifstream stream(filename, std::ios::in | std::ios::binary);

    stream.read(reinterpret_cast<char *>(n_bytes), sizeof(uint32_t));

    *src = new (std::nothrow) double[*n_bytes / sizeof(double)];
    *dst = new (std::nothrow) double[*n_bytes / sizeof(double)];
    stream.read(reinterpret_cast<char *>(*src), *n_bytes);
    stream.read(reinterpret_cast<char *>(*dst), *n_bytes);

    stream.close();
}

void linearRegressorFit(double* X, double* y, double* params, uint32_t N) {
    // Pad X on the top with ones
    double *cublasXPadded;
    // TODO: make ones global later
    double *ones = new double[N];
    for (int i = 0; i < N; i++)
        ones[i] = 1.0f;
    cudaMalloc(&cublasXPadded, sizeof(double) * N * 2);
    cudaMemcpy(cublasXPadded, ones, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cublasXPadded + N, X, sizeof(double) * N, cudaMemcpyHostToDevice);
    delete[] ones;

//    std::cout << "y:";
//    for (int i = 0; i < N; i++) {
//        std::cout << ' ' << y[i];
//    }
//    std::cout << '\n';

//    double *padded = new double[N * 2];
//    cudaMemcpy(padded, cublasXPadded, sizeof(double) * N * 2, cudaMemcpyDeviceToHost);
//    std::cout << "Padded:";
//    for (int i = 0; i < N * 2; i++) {
//        std::cout << ' ' << padded[i];
//    }
//    std::cout << '\n';

    // Perform outer product of cublasMatMul1 = cublasXPadded.T x cublasXPadded
    double *cublasMatMul1;
    const double alpha = 1.0f; const double beta = 0.0f;
    cudaMalloc(&cublasMatMul1, sizeof(double) * 4);
    cudaMemset(cublasMatMul1, 0, sizeof(double) * 4);
    cublasHandle_t handle = 0;
    cublasCreate(&handle);
    cublasDgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            2, 2, N,
            &alpha,
            cublasXPadded, N,
            cublasXPadded, N,
            &beta,
            cublasMatMul1, 2
    );
    cudaFree(cublasXPadded);

//    double* matMul1 = new double[4];
//    cudaMemcpy(matMul1, cublasMatMul1, sizeof(double) * 4, cudaMemcpyDeviceToHost);
//    std::cout << "matMul1:";
//    for (int i = 0; i < 4; i++) {
//        std::cout << ' ' << matMul1[i];
//    }
//    std::cout << '\n';

    // Invert X.T x X
    double matMul1[4];
    cudaMemcpy(matMul1, cublasMatMul1, sizeof(double) * 4, cudaMemcpyDeviceToHost);
    const double determinant = matMul1[0] * matMul1[3] - (matMul1[1] * matMul1[2]);
    if (abs(determinant) <= 0.0001) {
        std::cout << "matMul1:";
        for (int i = 0; i < 4; i++) {
            std::cout << ' ' << matMul1[i];
        }
        std::cout << '\n';
        fprintf(stderr, "Failed to invert matrix - determinant of 0 discovered.\n");
        exit(EXIT_FAILURE);
    }
    const double inverseMatrix[4] = {matMul1[3] / determinant, -matMul1[1] / determinant, -matMul1[2] / determinant, matMul1[0] / determinant};
    cudaFree(cublasMatMul1);

//    std::cout << "Inverted matrix:";
//    for (int i = 0; i < 4; i++) {
//        std::cout << ' ' << inverseMatrix[i];
//    }
//    std::cout << '\n';

    // Multiply inverse(X.T x X) by X.T
    double *cublasInverseMat, *cublasMatMul2;
    cudaMalloc(&cublasInverseMat, sizeof(double) * 4);
    cudaMalloc(&cublasMatMul2, sizeof(double) * N * 2);
    cudaMemcpy(cublasInverseMat, inverseMatrix, sizeof(double) * 4, cudaMemcpyHostToDevice);
    cudaMemset(cublasMatMul2, 0, sizeof(double) * N * 2);
    cublasDgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            2, N, 2,
            &alpha,
            cublasInverseMat, 2,
            cublasXPadded, N,
            &beta,
            cublasMatMul2, 2
    ); // Gets the transpose of the answer?
    cudaFree(cublasInverseMat);

//    double* matMul2 = new double[N * 2];
//    cudaMemcpy(matMul2, cublasMatMul2, sizeof(double) * N * 2, cudaMemcpyDeviceToHost);
//    std::cout << "MatMul2:";
//    for (int i = 0; i < N * 2; i++) {
//        std::cout << ' ' << matMul2[i];
//    }
//    std::cout << '\n';
//    delete matMul2;

    // Multiply (inverse(X.T x X) by X.T) by y
    double *cublasY, *cublasMatMul3;
    cudaMalloc(&cublasY, sizeof(double) * N);
    cudaMalloc(&cublasMatMul3, sizeof(double) * 2);
    cublasSetVector(N, sizeof(double), y, 1, cublasY, 1);
    cudaMemset(cublasMatMul3, 0, sizeof(double) * 2);
    cublasDgemv(
            handle, CUBLAS_OP_N,
            2, N,
            &alpha,
            cublasMatMul2, 2,
            cublasY, 1,
            &beta,
            cublasMatMul3, 1
    );
    cudaFree(cublasMatMul2);

//    double* matMul3 = new double[2];
//    cudaMemcpy(matMul3, cublasMatMul3, sizeof(double) * 2, cudaMemcpyDeviceToHost);
//    std::cout << "MatMul3:";
//    for (int i = 0; i < 2; i++) {
//        std::cout << ' ' << matMul3[i];
//    }
//    std::cout << '\n';
//    delete matMaul3;

    cudaMemcpy(params, cublasMatMul3, sizeof(double) * 2, cudaMemcpyDeviceToHost);

    // Cuda Frees
    cudaFree(cublasMatMul3);
    cudaFree(cublasY);

    // Cublas Destroy Handle
    cublasDestroy(handle);
}

void linearRegressorPredict(double* X, double* params, double* result, uint32_t N) {
    // Pad X on the top with ones
    double *cublasXPadded;
    double *ones = new double[N];
    for (int i = 0; i < N; i++)
        ones[i] = 1.0f;
    cudaMalloc(&cublasXPadded, sizeof(double) * N * 2);
    cudaMemcpy(cublasXPadded, ones, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cublasXPadded + N, X, sizeof(double) * N, cudaMemcpyHostToDevice);
    delete[] ones;

//    double *padded = new double[N * 2];
//    cudaMemcpy(padded, cublasXPadded, sizeof(double) * N * 2, cudaMemcpyDeviceToHost);
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
    double *cublasParams, *cublasMatMul;
    const double alpha = 1.0f; const double beta = 1.0f;
    cudaMalloc(&cublasParams, sizeof(double) * 2);
    cudaMalloc(&cublasMatMul, sizeof(double) * N);
    cudaMemcpy(cublasParams, params, sizeof(double) * 2, cudaMemcpyHostToDevice);
    cudaMemset(cublasMatMul, 0, sizeof(double) * N);
    cublasHandle_t handle = 0;
    cublasCreate(&handle);
    cublasDgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            1, N, 2,
            &alpha,
            cublasParams, 1,
            cublasXPadded, N,
            &beta,
            cublasMatMul, 1
    );

//    double *matMul = new double[N];
//    cudaMemcpy(matMul, cublasMatMul, sizeof(double) * N, cudaMemcpyDeviceToHost);
//    std::cout << "MatMul:";
//    for (int i = 0; i < N; i++) {
//        std::cout << ' ' << matMul[i];
//    }
//    std::cout << '\n';
//    delete matMul;

    cudaMemcpy(result, cublasMatMul, sizeof(double) * N, cudaMemcpyDeviceToHost);

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
void ransacKernel(const double* const X, const double* const y, const uint n, const double t, const uint d, const uint N, RansacFitResult* const fitResult) {
    // Generate random indices to use - Boost uses std under the hood.
    std::vector<unsigned int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);


    // Set the X and y according to randomly sampled indices
    double* XLin = new double[n];
    double* yLin = new double[n];
    for (int i = 0; i < n; i++) {
        XLin[i] = X[indices[i]];
        yLin[i] = y[indices[i]];
    }
    double *initialParams = new double[LIN_REG_PARAMS_DIM];

    // Fit the linear regressor to our model
    linearRegressorFit(XLin, yLin, initialParams, n);

    // Perform prediction
    double* yLinPred = new double[n];
    linearRegressorPredict(XLin, initialParams, yLinPred, n);

    // Delete array
    delete initialParams;

    // Get thresholded indices
    // mean square loss vs mean square error
    std::vector<double> XLinThreshold;
    std::vector<double> yLinThreshold;
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
    const int numInliers = XLinThreshold.size();

    // Number of inlier points is too low so we discard this from consideration
    if (numInliers < d) {
        fitResult->error = std::numeric_limits<double>::max();
        return;
    }

    // Create a model on the inliers & set the params
    linearRegressorFit(&XLinThreshold[0], &yLinThreshold[0], fitResult->params, numInliers);

    // Form a prediction from the inliers
    double* yLinThresholdPred = new double[numInliers];
    linearRegressorPredict(&XLinThreshold[0], fitResult->params, yLinThresholdPred, numInliers);

    // Compute the mean squared error
    double squareErrorAccumulator = 0;
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
void ransacFit(double* X, double* y, uint n, uint k, double t, uint d, uint N, RansacFitResult* bestFitResult) {
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

    bestFitResult->error = std::numeric_limits<double>::max();
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

int main() {
    /*
    X = np.array([-0.848,-0.800,-0.704,-0.632,-0.488,-0.472,-0.368,-0.336,-0.280,-0.200,-0.00800,-0.0840,0.0240,0.100,0.124,0.148,0.232,0.236,0.324,0.356,0.368,0.440,0.512,0.548,0.660,0.640,0.712,0.752,0.776,0.880,0.920,0.944,-0.108,-0.168,-0.720,-0.784,-0.224,-0.604,-0.740,-0.0440,0.388,-0.0200,0.752,0.416,-0.0800,-0.348,0.988,0.776,0.680,0.880,-0.816,-0.424,-0.932,0.272,-0.556,-0.568,-0.600,-0.716,-0.796,-0.880,-0.972,-0.916,0.816,0.892,0.956,0.980,0.988,0.992,0.00400]).reshape(-1,1)
    y = np.array([-0.917,-0.833,-0.801,-0.665,-0.605,-0.545,-0.509,-0.433,-0.397,-0.281,-0.205,-0.169,-0.0531,-0.0651,0.0349,0.0829,0.0589,0.175,0.179,0.191,0.259,0.287,0.359,0.395,0.483,0.539,0.543,0.603,0.667,0.679,0.751,0.803,-0.265,-0.341,0.111,-0.113,0.547,0.791,0.551,0.347,0.975,0.943,-0.249,-0.769,-0.625,-0.861,-0.749,-0.945,-0.493,0.163,-0.469,0.0669,0.891,0.623,-0.609,-0.677,-0.721,-0.745,-0.885,-0.897,-0.969,-0.949,0.707,0.783,0.859,0.979,0.811,0.891,-0.137]).reshape(-1,1)

    
    */
    // double *X, *y;
    // uint32_t n_bytes;
    // readPtsFile("./data/src_dst_pts.bin", &X, &y, &n_bytes);

    const uint32_t n_elements = 69;
    double X[n_elements] = {
        -0.848,-0.800,-0.704,-0.632,-0.488,-0.472,-0.368,-0.336,-0.280,-0.200,-0.00800,-0.0840,0.0240,0.100,0.124,0.148,0.232,0.236,0.324,0.356,0.368,0.440,0.512,0.548,0.660,0.640,0.712,0.752,0.776,0.880,0.920,0.944,-0.108,-0.168,-0.720,-0.784,-0.224,-0.604,-0.740,-0.0440,0.388,-0.0200,0.752,0.416,-0.0800,-0.348,0.988,0.776,0.680,0.880,-0.816,-0.424,-0.932,0.272,-0.556,-0.568,-0.600,-0.716,-0.796,-0.880,-0.972,-0.916,0.816,0.892,0.956,0.980,0.988,0.992,0.00400
    };
    double y[n_elements] = {
        -0.917,-0.833,-0.801,-0.665,-0.605,-0.545,-0.509,-0.433,-0.397,-0.281,-0.205,-0.169,-0.0531,-0.0651,0.0349,0.0829,0.0589,0.175,0.179,0.191,0.259,0.287,0.359,0.395,0.483,0.539,0.543,0.603,0.667,0.679,0.751,0.803,-0.265,-0.341,0.111,-0.113,0.547,0.791,0.551,0.347,0.975,0.943,-0.249,-0.769,-0.625,-0.861,-0.749,-0.945,-0.493,0.163,-0.469,0.0669,0.891,0.623,-0.609,-0.677,-0.721,-0.745,-0.885,-0.897,-0.969,-0.949,0.707,0.783,0.859,0.979,0.811,0.891,-0.137
    };

    const int n = 10; // Minimum number of data points to estimate parameters
    const int k = 1000; // Maximum number of iterations allowed
    const double t = 0.05; // Threshold value to determine if points are fit well
    const int d = 10; // Number of close data points required to assert model fits
    RansacFitResult bestFitResult;
    ransacFit(
        X, y, n, k, t, d, n_elements, &bestFitResult
    );

    std::cout << "Params: (" << bestFitResult.params[0] << ',' << bestFitResult.params[1] << ")\n";
    std::cout << "MSE: " << bestFitResult.error << '\n';

    // delete X;
    // delete y;
}
