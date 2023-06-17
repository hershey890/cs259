/*

Resources
---------
* https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/gesvd/cusolver_gesvd_example.cu
* https://docs.nvidia.com/cuda/cusolver/index.html
* http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdfs
 */
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <vector>
#include <limits.h>
#include <numeric>
#include <random>
#include <cmath>
#include <algorithm>

class Points {
public:
    float* source; // n/2 rows x 2 cols
    float* destination; // n/2 rows x 2 cols
    int n_points;
};


Points* readPtsFile(std::string filename) {
    std::ifstream stream(filename, std::ios::in | std::ios::binary);

    uint32_t n_bytes;
    stream.read(reinterpret_cast<char *>(&n_bytes), sizeof(uint32_t));

    Points *points = new(std::nothrow) Points;
    points->source = new(std::nothrow) float[n_bytes / sizeof(float)];
    points->destination = new(std::nothrow) float[n_bytes / sizeof(float)];
    points->n_points = n_bytes / (2*sizeof(float));
    stream.read(reinterpret_cast<char *>(points->source), n_bytes);
    stream.read(reinterpret_cast<char *>(points->destination), n_bytes);

    stream.close();

    return points;
}

__global__ void threshold(
    double* temp, 
    bool* result, 
    double* dst_pts, 
    int N, 
    double val
){
    int thread = blockDim.x * blockIdx.x + threadIdx.x;
    double norm = temp[thread * 3];

    double val1 = temp[thread*3]/norm - dst_pts[thread*3], 
           val2 = temp[thread*3+1]/norm - dst_pts[thread*3+1], 
           val3 = temp[thread*3+2]/norm - dst_pts[thread*3+2];
           
    val1 *= val1; val2 *= val2; val3 *= val3;
    double sum = (val1 + val2 + val3) / N;
    result[thread] = sum < val;
}

__global__ void calc_error(
    double* temp, 
    double* error, 
    double* dst_pts, 
    const int N
){
    __shared__ double errs[1132 * 3];

    int thread = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = thread; i < N; i+= blockDim.x){
        errs[i] = temp[i] / temp[i + (2 - (i % 3))] - dst_pts[i];
        errs[i] *= errs[i];
    }

    __syncthreads();
    int sum = 0;
    if (thread == 0){
        for (int i = 0; i < N*3; i++){
            sum += errs[i];
        }
        *error = sum / N;
    }

}

void ransac_fit(
    double* src_points, 
    double* dst_points, 
    double* M, 
    int N, 
    int k, 
    double tval,
    std::vector<bool>& best_inlier_mask,
    double& best_error
){
    const int n_data_pts = 100, n_valid_data_pts = 80;
    std::vector<double> ones(N, 1);

    // send data to gpu
    double* cudaMul;
    double* cudaModel;
    double* cudaX;
    double* cudaY;
    double* cudaResult;
    double* cudaError;
    bool* cudaMask;

    cudaMalloc(&cudaModel, sizeof(double) * 9);
    cudaMalloc(&cudaResult, sizeof(double) * N * 3);
    cudaMalloc(&cudaMul, sizeof(double) * N * 3);
    cudaMalloc(&cudaX, sizeof(double) * N * 3);
    cudaMalloc(&cudaY, sizeof(double) * N * 3);
    cudaMalloc(&cudaMask, sizeof(bool) * N);
    cudaMalloc(&cudaError, sizeof(double));

    cudaMemcpy(cudaModel, M, sizeof(double) * 9, cudaMemcpyHostToDevice);

    std::vector<bool> inlier_mask(N, false);
    best_error = 1000000;
    //std::vector<bool> best_inlier_mask;

    cublasHandle_t handle = 0;
    cublasCreate(&handle);

    //coarse grain parallelize this
    for (int i = 0; i < k; i++){

        // sample src points
        std::vector<unsigned int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);


        // Set the X and y according to randomly sampled indices
        double* sampledX = new double[n_data_pts*3];
        double* sampledY = new double[n_data_pts*3];
        for (int i = 0; i < n_data_pts; i++) {
            sampledX[i*3] = src_points[indices[i]*3];
            sampledX[i*3 + 1] = src_points[indices[i]*3 + 1];
            sampledX[i*3 + 2] = src_points[indices[i]*3 + 2];
            sampledY[i*3] = dst_points[indices[i]*3];
            sampledY[i*3 + 1] = dst_points[indices[i]*3 + 1];
            sampledY[i*3 + 2] = dst_points[indices[i]*3 + 2];
        }

        
        // perform threshold check
        // double* cudaX;
        // double* cudaY;
        // double* cudaMul;
        // bool* cudaMask;

        const double alpha = 1.0f; const double beta = 0.0f;
        // cudaMalloc(&cudaX, sizeof(double) * n_data_pts * 3);
        // cudaMalloc(&cudaY, sizeof(double) * n_data_pts * 3);
        // cudaMalloc(&cudaMul, sizeof(double) * n_data_pts * 3);
        // cudaMalloc(&cudaMask, sizeof(bool) * n_data_pts);

        cudaMemcpy(cudaX, sampledX, sizeof(double) * n_data_pts * 3, cudaMemcpyHostToDevice);
        cudaMemcpy(cudaY, sampledY, sizeof(double) * n_data_pts * 3, cudaMemcpyHostToDevice);

        //do matmul

        cublasDgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            n_data_pts, 3, 3,
            &alpha,
            cudaX, n_data_pts,
            cudaModel, 3,
            &beta,
            cudaMul, n_data_pts
        );

        double* mul = new double[3*n_data_pts];
        cudaMemcpy(mul, cudaMul, sizeof(double) * n_data_pts * 3, cudaMemcpyDeviceToHost);
        // for (int i = 0; i < n_data_pts * 3; i++){
        //     std::cout << mul[i] << std::endl;
        // }

        threshold<<<4,25>>>(cudaMul, cudaMask, cudaY, n_data_pts, tval);

        bool* temp_inlier_mask = new bool[n_data_pts];
        cudaMemcpy(temp_inlier_mask, cudaMask, sizeof(bool) * n_data_pts, cudaMemcpyDeviceToHost);

        // for (int i = 0; i < n_data_pts; i++){
        //     if (temp_inlier_mask[i])
        //         std::cout << temp_inlier_mask[i] << std::endl;
        // }

        int total_pts = 0;
        for(int i = 0; i<n_data_pts; i++){
            total_pts += temp_inlier_mask[i];
        }

        if (total_pts >= n_valid_data_pts) {
            
            std::vector<double> src, dst;
            for(int i = 0; i < n_data_pts; i++){
                inlier_mask[indices[i]] = inlier_mask[indices[i]] || temp_inlier_mask[i];
            }
            
            for(int i = 0; i < N; i++){
                if (inlier_mask[i]){
                    src.push_back(src_points[i*3]);
                    src.push_back(src_points[i*3+1]);
                    src.push_back(src_points[i*3+2]);

                    dst.push_back(dst_points[i*3]);
                    dst.push_back(dst_points[i*3+1]);
                    dst.push_back(dst_points[i*3+2]);
                }
            }
            
            // cudaMalloc(&cudaX, sizeof(double) * src.size());
            // cudaMalloc(&cudaY, sizeof(double) * src.size());
            int inlier_size = src.size() / 3;
            cudaMemcpy(cudaX, src.data(), sizeof(double) * src.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(cudaY, src.data(), sizeof(double) * src.size(), cudaMemcpyHostToDevice);

            cublasDgemm(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                inlier_size, 3, 3,
                &alpha,
                cudaX, inlier_size,
                cudaModel, 3,
                &beta,
                cudaMul, inlier_size
            );
            
            calc_error<<<1, 1024>>>(cudaMul, cudaError, cudaY, inlier_size);

            double err = 10000000000000;
            cudaMemcpy(&err, cudaError, sizeof(double), cudaMemcpyDeviceToHost);
            std::cout << err << std::endl;
            if (err < best_error){
                best_error = err;
                best_inlier_mask = inlier_mask;
            }
        }
        //invoke threshold check kernel
    }
}


int main()
{
    float reproj_thresh = 10.0;
    double M[9]= {0.598693576, 0.0169989140, 514.018603, -0.136867155, 0.876117794, 41.4169688, -0.000334456444, -0.00000839683573, 1.0000};
    Points *pts = readPtsFile("./data/src_dst_pts.bin");

    //std::cout << pts->n_points << std::endl;
    std::vector<double> src, dst;
    const int max_iter = 10000, N = 1132;

    for (int i = 0 ; i < N; i++){
        src.push_back(pts->source[i*2]);
        src.push_back(pts->source[i*2+1]);
        src.push_back(1);
        dst.push_back(pts->destination[i*2]);
        dst.push_back(pts->destination[i*2+1]);
        dst.push_back(1);
    }

    std::vector<bool> best_inlier_mask;
    double best_error = 1000000;
    ransac_fit(src.data(), dst.data(), M , N, max_iter, 10000.0, best_inlier_mask, best_error);

    std::cout << best_error << std::endl;
    //RansacReturn *ret = ransac(pts, reproj_thresh, max_iter);
    
    //char* best_inlier_mask = ret->inlier_mask;
    //float* best_model = ret->h;

    /*std::cout << "Best model: " << std::endl;
    for(int i=0; i<9; i++)
        std::cout << best_model[i] << " ";
    std::cout << std::endl;*/

    //delete ret;

}