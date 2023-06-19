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
#include <chrono>
#include <thread>
#include <pthread.h>

#define NUM_THREADS 8
#define DATA_SIZE 1132 * 3

const int N = 1132;
const int n_data_pts = 100, n_valid_data_pts = 80;

class Points {
public:
    float* source; // n/2 rows x 2 cols
    float* destination; // n/2 rows x 2 cols
    int n_points;
};

struct thread_data {
    double* src_points;
    double* dst_points;
    double* cudaModel;
    double* cudaX;
    double* cudaY;
    double* cudaMul;
    bool* cudaMask;
    double* cudaError;
    int N;
    int k;
    double tval;
    int i;
    double best_error;
    std::vector<bool>best_mask;
    cublasHandle_t handle;
    std::vector<unsigned int>* indices;
    std::vector<bool> *inlier_mask;
    double* sampledX;
    double* sampledY;
    bool* temp_inlier_mask;
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
    __shared__ double errs[283 * 3];
    int curt = threadIdx.x;
    int bldim = blockDim.x;
    int thread = blockDim.x * blockIdx.x + threadIdx.x;
    int bl = blockIdx.x;
    if (thread < N) {
        
        double norm = temp[2 * N + thread];

        double val1 = temp[thread]/norm - dst_pts[thread], 
            val2 = temp[1 * N + thread]/norm - dst_pts[1 * N + thread], 
            val3 = temp[2 * N + thread]/norm - dst_pts[2 * N + thread];
            
        val1 *= val1; val2 *= val2; val3 *= val3;

        errs[curt] = val1;
        errs[1 * bldim + curt] = val2;
        errs[2 * bldim + curt] = val3;

        __syncthreads();
        if (curt == 0){
            double sum = 0;
            for (int i = 0 ; i<283*3; i++){
                sum += errs[i];
            }
            error[bl] = sum; 
        }
    }


}


void* ransac_fit(
    void* arg
){

    thread_data *tdata=(thread_data *)arg;
    int loopIndex = tdata->i;
    double* src_points = tdata->src_points;
    double* dst_points = tdata->dst_points;
    double* cudaMul = tdata->cudaMul + loopIndex * DATA_SIZE;
    double* cudaX = tdata->cudaX + loopIndex * DATA_SIZE;
    double* cudaY = tdata->cudaY + loopIndex * DATA_SIZE;
    bool* cudaMask = tdata->cudaMask + loopIndex * DATA_SIZE/3;
    double* cudaError = tdata->cudaError+loopIndex * 4;
    double* cudaModel = tdata->cudaModel;
    cublasHandle_t handle = tdata->handle;
    std::vector<unsigned int>& indices = *(tdata->indices);
    std::vector<bool>& inlier_mask = *(tdata->inlier_mask);
    double* sampledX = tdata->sampledX;
    double* sampledY = tdata->sampledY;
    bool* temp_inlier_mask = tdata->temp_inlier_mask;
    // std::vector<double>& src = *(tdata->src);
    // std::vector<double>& dst = *(tdata->dst);

    int N = tdata->N;
    int k = tdata->k;
    double tval = tdata->tval;
    std::vector<bool>& best_inlier_mask = tdata->best_mask;
    double& best_error = tdata->best_error;

    best_error = 1000000;

    int start = loopIndex * (k / NUM_THREADS);
    int end = (loopIndex == NUM_THREADS - 1) ? k : (loopIndex + 1) * (k / NUM_THREADS);
            
    // int src_size = 0;
    // int src_size_prev = 0;

    //coarse grain parallelize this
    for (int i = start; i < end; i++){
        // sample src points
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        // Set the X and y according to randomly sampled indices
        // double* sampledX = new double[n_data_pts*3];
        // double* sampledY = new double[n_data_pts*3];
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
            for(int i=0; i<n_data_pts; i++)
                inlier_mask[indices[i]] = inlier_mask[indices[i]] || temp_inlier_mask[i];

            int n_pts_used = 0;
            std::vector<double> src, dst;
            for(int j = 0; j < 3; j++){
                for(int i = 0; i < N; i++){
                    if (inlier_mask[i]){
                        src.push_back(src_points[i+j*N]);
                        dst.push_back(dst_points[i+j*N]);
                    }
                }
            }

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

            calc_error<<<4, 283>>>(cudaMul, cudaError, cudaY, inlier_size);

            // double *err = new double[1];
            double errors[4];
            cudaMemcpy(&errors, cudaError, sizeof(double) * 4, cudaMemcpyDeviceToHost);
    
            // *err = *err / inlier_size;
            double err = 0;
             for (int i = 0 ; i < 4; i++){
                err += errors[i];
            }
            err = err / inlier_size;

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
            if (err < best_error){
                best_error = err;
                best_inlier_mask = inlier_mask;
            }
        }
        //invoke threshold check kernel
    }

    pthread_exit(NULL);
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
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;


    float reproj_thresh = 10.0;
    double M[9]= { 5.98693576e-01, -1.36867155e-01, -3.34456444e-04, 1.69989140e-02,  8.76117794e-01, -8.39683573e-06, 514.0186034790541, 41.416968762719435, 1.0};
    Points *pts = readPtsFile("./data/src_dst_pts.bin");

    //std::cout << pts->n_points << std::endl;
    std::vector<double> src, dst;
    const int max_iter = 10000;
    

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

    // send data to gpu
    double* cudaMul;
    double* cudaModel;
    double* cudaX;
    double* cudaY;
    double* cudaError;
    bool* cudaMask;

    cudaMalloc(&cudaModel, sizeof(double) * 9);
    cudaMalloc(&cudaMul, sizeof(double) * N * 3 * NUM_THREADS);
    cudaMalloc(&cudaX, sizeof(double) * N * 3 * NUM_THREADS);
    cudaMalloc(&cudaY, sizeof(double) * N * 3 * NUM_THREADS);
    cudaMalloc(&cudaMask, sizeof(bool) * N * NUM_THREADS);
    cudaMalloc(&cudaError, sizeof(double) * N);

    cudaMemcpy(cudaModel, M, sizeof(double) * 9, cudaMemcpyHostToDevice);
    std::vector<bool> best_inlier_mask;
    double best_error = 1000000;
    
    // Create an array of thread objects
    pthread_t threads[NUM_THREADS];

    // Create an array of loop indices
    thread_data data[NUM_THREADS];

    // Create threads and assign loop indices
    for (int i = 0; i < NUM_THREADS; i++) {
        data[i].src_points = src.data();
        data[i].dst_points = dst.data();
        data[i].cudaMul = cudaMul;
        data[i].cudaModel = cudaModel;
        data[i].cudaX = cudaX;
        data[i].cudaY = cudaY;
        data[i].cudaError = cudaError;
        data[i].cudaMask = cudaMask;
        data[i].N = N;
        data[i].k = max_iter;
        data[i].tval = reproj_thresh;
        data[i].i = i;
        data[i].best_error = best_error;
        cublasHandle_t handle = 0;
        data[i].handle = handle;
        cublasCreate(&(data[i].handle));
        data[i].indices = new std::vector<unsigned int>(N);
        data[i].inlier_mask = new std::vector<bool>(N, false);
        data[i].sampledX= new double[n_data_pts*3];
        data[i].sampledY= new double[n_data_pts*3];
        data[i].temp_inlier_mask = new bool[n_data_pts];
    }

    


    auto t1 = high_resolution_clock::now();
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, ransac_fit, &data[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    auto t2 = high_resolution_clock::now();

    int best_idx = 0;
    for (int i = 0; i < NUM_THREADS; i++){
        if (best_error > data[i].best_error){
            best_idx = i;
            best_error = data[i].best_error;
        }
    }
    // ransac_fit(src.data(), dst.data(), M , N, max_iter, reproj_thresh, best_inlier_mask, best_error);
    cudaFree(cudaModel);
    cudaFree(cudaMul);
    cudaFree(cudaX);
    cudaFree(cudaY);
    cudaFree(cudaMask);
    cudaFree(cudaError);

    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout <<"best error: " << best_error << std::endl;
    std::cout <<"best mask:" <<std::endl;

    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";
    return 0;
    std::string filename = "new.csv";
    writeArrayToCSV(data[best_idx].best_mask, filename);
}