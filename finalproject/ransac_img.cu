#include <iostream>
#include <fstream>
#include <unordered_set>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

struct Points {
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
    // np.random.choice(n_points, n_pts_to_fit_model, replace=False) in C++
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


void fit(float* src, float* dst, char* random_indices, int n_pts_to_fit_model, 
        float* h, float* A)
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
}


char* ransac(Points *pts, float reproj_thresh, int max_iter)
{
    const int n_pts_to_fit_model = 100; // min # pts needed to fit model
    const int n_required_valid_pts = 100; // min # pts needed to accept model

    float* src = pts->source;
    float* dst = pts->destination;
    int n_points = pts->n_points;

    char* inlier_mask_temp = new char[n_points];
    char* inlier_mask = new char[n_points];
    char* best_inlier_mask = new char[n_points];
    char random_indices[n_pts_to_fit_model] = {0};

    float best_error = 1e9;

    // Allocate A and h on GPU
    float h[9];
    // const int 
    float A[2*n_pts_to_fit_model*9];
    float *d_A, *d_h;
    cudaMalloc(&d_A, 2*n_pts_to_fit_model*9*sizeof(float));
    cudaMalloc(&d_h, 9*sizeof(float));

    cudaMemcpy(d_A, A, 2*n_pts_to_fit_model*9*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h, 9*sizeof(float), cudaMemcpyHostToDevice);

    // Create cuSolver and cuBlas handles
    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // Compte and allocate workspace memory
    int lwork;
    cusolverDnSgesvd_bufferSize(cusolver_handle, 9, 9, &lwork);


    for(int i=0; i<max_iter; i++) {
        fit(src, dst, random_indices, n_pts_to_fit_model, h, A);
    }

    delete[] inlier_mask_temp;
    delete[] inlier_mask;

    return best_inlier_mask;
}


int main()
{
    float reproj_thresh = 10.0;
    int max_iter = 10000;


    Points *pts = readPtsFile("./data/src_dst_pts.bin");
    
    char* best_inlier_mask = ransac(pts, reproj_thresh, max_iter);
}