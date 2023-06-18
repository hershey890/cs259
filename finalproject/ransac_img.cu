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
    double norm = temp[2 * N + thread];

    double val1 = temp[thread]/norm - dst_pts[thread], 
           val2 = temp[1 * N + thread]/norm - dst_pts[1 * N + thread], 
           val3 = temp[2 * N + thread]/norm - dst_pts[2 * N + thread];
           
    val1 *= val1; val2 *= val2; val3 *= val3;
    double sum = (val1 + val2 + val3);
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
    if (thread < N) {
        
        double norm = temp[2 * N + thread];

        double val1 = temp[thread]/norm - dst_pts[thread], 
            val2 = temp[1 * N + thread]/norm - dst_pts[1 * N + thread], 
            val3 = temp[2 * N + thread]/norm - dst_pts[2 * N + thread];
            
        val1 *= val1; val2 *= val2; val3 *= val3;

        errs[thread] = val1;
        errs[1 * N + thread] = val2;
        errs[2 * N + thread] = val3;

        __syncthreads();
        if (thread == 0){
            double sum = 0;
            for (int i = 0; i<N*3; i ++){
                sum += errs[i];
            }
            *error = sum;
        }
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
    cudaMalloc(&cudaError, sizeof(double) * N * 3);

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

        // indices = {1126,  960,  191,  588,  164,  939,  443,   57,  242,   64,   61,
        // 698,  185,  149,  442,    7,  390,  943, 1128, 1025,  864,  346,
        // 929,  705,  771, 1031,  944,  879,   52,  116,  827,  375,  379,
        // 782,  834,  146,  493,  795,  272,  123,  707,  427,  545, 1036,
        // 320,  152,  196,   80,  211,  665,  296,  975,  148,  486,  174,
        // 495,  810,  524,  546,   56,  784, 1108,  595,  201,  184,  585,
        // 854,  959,  227,  531,   70, 1130,  663,  985,  971,  586,  373,
        // 161,  783,  692,  741,  186,  421,  607,  883,  312, 1062,  690,
        // 257,  681,  979,  899,  989,   92,  279,  353,  452, 1066, 1043,
        // 54};

        // Set the X and y according to randomly sampled indices
        double* sampledX = new double[n_data_pts*3];
        double* sampledY = new double[n_data_pts*3];
        for (int j = 0; j < 3; j++){
            for (int i = 0; i < n_data_pts; i++) {
                sampledX[j*n_data_pts + i] = src_points[j * N + indices[i]];
                sampledY[j*n_data_pts +i] = dst_points[j * N + indices[i]];
                // if (i < 3 || i > (n_data_pts-3)){
                //     std::cout << sampledX[j*n_data_pts + i] << std::endl;
                //     std::cout << sampledY[j*n_data_pts +i] << std::endl;
                // }
            }
        }
        
        // perform threshold check
        const double alpha = 1.0f; const double beta = 0.0f;

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

        // double* mul = new double[3*n_data_pts];
        // cudaMemcpy(mul, cudaMul, sizeof(double) * n_data_pts * 3, cudaMemcpyDeviceToHost);
        // for (int i = 0; i < 9; i++){
        //     std::cout << mul[i] << std::endl;
        // }

        // std::cout << "here" << std::endl;

        threshold<<<4, 25>>>(cudaMul, cudaMask, cudaY, n_data_pts, tval);

        bool* temp_inlier_mask = new bool[n_data_pts];
        cudaMemcpy(temp_inlier_mask, cudaMask, sizeof(bool) * n_data_pts, cudaMemcpyDeviceToHost);

        // for (int i = 0; i < 27; i++){
        //     std::cout << temp_inlier_mask[i] << std::endl;
        // }

        // std::cout << "here" << std::endl;
        
        int total_pts = 0;
        for(int i = 0; i<n_data_pts; i++){
            total_pts += temp_inlier_mask[i];
        }

        // std::cout << total_pts << std::endl;

        // std::cout << "here" << std::endl;

        if (total_pts >= n_valid_data_pts) {
            std::vector<double> src, dst;
            for(int i = 0; i < n_data_pts; i++){
                inlier_mask[indices[i]] = inlier_mask[indices[i]] || temp_inlier_mask[i];
            }

            for(int j = 0; j < 3; j++){
                for(int i = 0; i < N; i++){
                    if (inlier_mask[i]){
                        src.push_back(src_points[i+j*N]);
                        dst.push_back(dst_points[i+j*N]);
                    }
                }
            }

            // for (int i = 0; i < 9; i ++) {
            //     std::cout<< src[i] << std::endl;
            //     std::cout << dst[i] << std::endl;
            // }

            int inlier_size = src.size() / 3;
            cudaMemcpy(cudaX, src.data(), sizeof(double) * src.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(cudaY, dst.data(), sizeof(double) * src.size(), cudaMemcpyHostToDevice);

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
            
            // double* ml = new double[N*3];
            // cudaMemcpy(ml, cudaMul, sizeof(double) * src.size(), cudaMemcpyDeviceToHost);
            //cudaMemcpy(cudaMul,ml,sizeof(double) * src.size(),cudaMemcpyHostToDevice);
            // for (int i = inlier_size; i < inlier_size + 9; i++){
            //     std::cout << ml[i] << std::endl;
            //     //std::cout << dst[i] << std::endl;
            // }
            
            //std::cout << "here" << std::endl;

            calc_error<<<1, 1000>>>(cudaMul, cudaError, cudaY, inlier_size);

            double *err = new double[1];
            cudaMemcpy(err, cudaError, sizeof(double), cudaMemcpyDeviceToHost);
    
            double s = 0;
            *err = *err / inlier_size;

            // for (int i = 0; i < 9; i ++){

            //     //if (abs(err[i] - pydat[i]) > 0.00005){
            //         //std::cout << "miss: " << i << std::endl;
            //         std::cout << err[i] << std::endl;

            //         //std::cout << ml[inlier_size * 2 + i] << std::endl;
            //         //std::cout << pydat[i] << std::endl;
            //     //}
            //     //s += err[i];
            // }
            // std::cout << "here" << std::endl;
            // std::cout << *err << std::endl;
            if (*err < best_error){
                best_error = *err;
                best_inlier_mask = inlier_mask;
            }
        }
        //invoke threshold check kernel
    }

    cudaFree(cudaModel);
    cudaFree(cudaResult);
    cudaFree(cudaMul);
    cudaFree(cudaX);
    cudaFree(cudaY);
    cudaFree(cudaMask);
    cudaFree(cudaError);
}

void writeArrayToCSV(const std::vector<bool>& array, const std::string& filename) {
    std::ofstream outputFile(filename);

    if (outputFile.is_open()) {
        for (size_t i = 0; i < array.size(); ++i) {
            outputFile << array[i];
            if (i != array.size() - 1) {
                outputFile << ",";
            }
        }

        outputFile.close();
        std::cout << "Array successfully written to " << filename << std::endl;
    } else {
        std::cout << "Failed to open the file: " << filename << std::endl;
    }
}

int main()
{
    float reproj_thresh = 10.0;
    double M[9]= { 5.98693576e-01, -1.36867155e-01, -3.34456444e-04, 1.69989140e-02,  8.76117794e-01, -8.39683573e-06, 514.0186034790541, 41.416968762719435, 1.0};
    Points *pts = readPtsFile("./data/src_dst_pts.bin");

    //std::cout << pts->n_points << std::endl;
    std::vector<double> src, dst;
    const int max_iter = 10000, N = 1132;
    
    for(int n = 0; n < 3; n++){
        for (int i = 0 ; i < N; i++){
            if (n != 2) {
                src.push_back(pts->source[i*2+n]);
                dst.push_back(pts->destination[i*2+n]);
            }
            else {
                src.push_back(1);
                dst.push_back(1);
            }

            // if (i < 3 || i > (N-3)){
            //     std::cout << pts->source[i*2+n] << std::endl;
            //     std::cout << pts->destination[i*2+n] << std::endl;
            // }
        }
    }

    std::vector<bool> best_inlier_mask;
    double best_error = 1000000;
    ransac_fit(src.data(), dst.data(), M , N, max_iter, 10.0, best_inlier_mask, best_error);

    std::cout <<"best error: " << best_error << std::endl;
    std::cout <<"best mask:" <<std::endl;

    std::string filename = "new.csv";
    writeArrayToCSV(best_inlier_mask, filename);


}