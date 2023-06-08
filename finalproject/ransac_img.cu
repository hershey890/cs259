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


void set_random_indices(char* random_indices, int n_pts_to_fit_model, int n_points)
{
    std::unordered_set<char> random_indices_set;
    while (random_indices_set.size() < n_pts_to_fit_model) {
        random_indices_set.insert(rand() % n_points);
    }

    int i = 0;
    for(auto v: random_indices_set) {
        random_indices[i] = v;
        i++;
    }
}


void fit(cusolverDnHandle_t cusolver_handle, float* src, float* dst, char* random_indices, 
        int n_pts_to_fit_model,  float* h, float* d_h, float* A, float* d_A, float *d_S, 
        float *d_U, float *d_VT, float *d_work, int *dev_info, int lwork)
{
    // set A matrix
    for(int i=0; i<n_pts_to_fit_model; i++) {
        int idx = random_indices[i];
        float x1 = src[2*idx];
        float y1 = src[2*idx+1];
        float x2 = dst[2*idx];
        float y2 = dst[2*idx+1];

        A[2*i + 0] = x1;
        A[2*i + 1] = y1;
        A[2*i + 2] = 1;
        A[2*i + 3] = 0;
        A[2*i + 4] = 0;
        A[2*i + 5] = 0;
        A[2*i + 6] = -x2*x1;
        A[2*i + 7] = -x2*y1;
        A[2*i + 8] = -x2;

        A[2*i + 9] = 0;
        A[2*i + 10] = 0;
        A[2*i + 11] = 0;
        A[2*i + 12] = x1;
        A[2*i + 13] = y1;
        A[2*i + 14] = 1;
        A[2*i + 15] = -y2*x1;
        A[2*i + 16] = -y2*y1;
        A[2*i + 17] = -y2;
    }

    // copy A to GPU
    cudaMemcpy(d_A, A, 2*n_pts_to_fit_model*9*sizeof(float), cudaMemcpyHostToDevice);

    // compute SVD
    cusolverDnSgesvd(cusolver_handle, 'A', 'A', 2*n_pts_to_fit_model, 9, d_A, 2*n_pts_to_fit_model, d_S, d_U, 2*n_pts_to_fit_model, d_VT, 9, d_work, lwork, NULL, dev_info);

    // save the last row of d_VT to h
    cublasGetVector(9, sizeof(float), d_VT + 8, 8, h, 1);
    // cudaMemcpy(h, d_VT + 8, 9*sizeof(float), cudaMemcpyDeviceToHost);
}

/*
 * Compute the mean squared error
 */
float mse(float x1, float y1, float x2, float y2, float* h) {
    float x2_hat = h[0]*x1 + h[1]*y1 + h[2];
    float y2_hat = h[3]*x1 + h[4]*y1 + h[5];
    float w2_hat = h[6]*x1 + h[7]*y1 + h[8];

    float x2_hat_norm = x2_hat / w2_hat;
    float y2_hat_norm = y2_hat / w2_hat;

    return pow(x2_hat_norm - x2, 2) + pow(y2_hat_norm - y2, 2);
}


/*
 * Compute the threshold values for each point and set the inlier mask
 * @return: number of inliers
 */
int threshold_values(float* src, float* dst, float* h, char* inlier_mask_temp, int n_points, float reproj_thresh,
    char* random_indices, int n_pts_to_fit_model)
{
    int n_inliers = 0;
    memset(inlier_mask_temp, 0, n_points*sizeof(char));
    for(int i=0; i<n_pts_to_fit_model; i++) {
        int idx = random_indices[i];
        float dist = mse(src[2*idx], src[2*idx+1], dst[2*idx], dst[2*idx+1], h);

        if(dist < reproj_thresh) {
            inlier_mask_temp[idx] = 1;
            n_inliers++;
        }
    }
    return n_inliers;
}


class RansacReturn {
public:
    char* inlier_mask;
    float* h;
    // create destructor
    ~RansacReturn() {
        delete[] inlier_mask;
        delete[] h;
    }
};


RansacReturn* ransac(Points *pts, float reproj_thresh, int max_iter)
{
    const int n_pts_to_fit_model = 100; // min # pts needed to fit model
    const int min_num_inliers = 80; // min # pts needed to accept model

    float* src = pts->source;
    float* dst = pts->destination;
    int n_points = pts->n_points;

    char* inlier_mask_temp = new char[n_points];
    char* inlier_mask =      new char[n_points];
    char* best_inlier_mask = new char[n_points];
    float *best_model =      new float[9];
    char random_indices[n_pts_to_fit_model] = {0};

    float best_error = 1e9;

    // Allocate A and h on GPU
    float h[9];
    const int m = 2*n_pts_to_fit_model;
    const int n = 9;
    float A[m*n];
    float *d_A, *d_h;

    cudaMalloc(&d_A, m*n*sizeof(float));
    cudaMalloc(&d_h, 9*sizeof(float));

    // Create cuSolver and cuBlas handles
    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // Compte and allocate workspace memory
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

    for(int i=0; i<max_iter; i++) {
        set_random_indices(random_indices, n_pts_to_fit_model, n_points);
        fit(cusolver_handle, src, dst, random_indices, n_pts_to_fit_model, h, d_h, A, d_A, d_S, d_U, d_VT, d_work, dev_info, lwork);
        int n_inliers = threshold_values(src, dst, h, inlier_mask_temp, n_points, reproj_thresh, random_indices, n_pts_to_fit_model);

        if(n_inliers >= min_num_inliers) {
            for(int i=0; i<n_points; i++)
                inlier_mask[i] = inlier_mask[i] || inlier_mask_temp[i];
            float error = 0;
            for(int i=0; i<n_points; i++) {
                if(inlier_mask[i])
                    error += mse(src[2*i], src[2*i+1], dst[2*i], dst[2*i+1], h);
            }
            if(error < best_error) {
                best_error = error;
                memcpy(best_inlier_mask, inlier_mask, n_points*sizeof(char));
                memcpy(best_model, h, 9*sizeof(char));
            }
        }
    }

    // TODO: transfer stuff back and do cudaFree
    cudaFree(d_A);
    cudaFree(d_h);
    cudaFree(d_work);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(dev_info);
    delete[] inlier_mask_temp;
    delete[] inlier_mask;

    RansacReturn *ret = new(std::nothrow) RansacReturn;
    ret->inlier_mask = best_inlier_mask;
    ret->h = best_model;

    return ret;
}


int main()
{
    float reproj_thresh = 10.0;
    int max_iter = 10000;


    Points *pts = readPtsFile("./data/src_dst_pts.bin");
    
    RansacReturn *ret = ransac(pts, reproj_thresh, max_iter);
    char* best_inlier_mask = ret->inlier_mask;
    float* best_model = ret->h;

    std::cout << "Best model: " << std::endl;
    for(int i=0; i<9; i++)
        std::cout << best_model[i] << " ";
    std::cout << std::endl;

    delete ret;
}