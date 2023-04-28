#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudnn.h>
#include "cublas_v2.h"


/* https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
 */
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}


/* https://stackoverflow.com/questions/13041399/equivalent-of-cudageterrorstring-for-cublas
 */
#define CHECK_CUBLAS(val) check_cublas((val), #val, __FILE__, __LINE__)
template <typename T>
void check_cublas(T err, const char* const func, const char* const file,
                  const int line)
{
    if (err != CUBLAS_STATUS_SUCCESS) {
        std::string errorStr;
        switch(err)
        {
            case CUBLAS_STATUS_SUCCESS: errorStr = "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED: errorStr = "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED: errorStr = "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE: errorStr = "CUBLAS_STATUS_INVALID_VALUE"; 
            case CUBLAS_STATUS_ARCH_MISMATCH: errorStr = "CUBLAS_STATUS_ARCH_MISMATCH"; 
            case CUBLAS_STATUS_MAPPING_ERROR: errorStr = "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED: errorStr = "CUBLAS_STATUS_EXECUTION_FAILED"; 
            case CUBLAS_STATUS_INTERNAL_ERROR: errorStr = "CUBLAS_STATUS_INTERNAL_ERROR"; 
            // default: errorStr = "unknown error";
        }
        std::cerr << "CUBLAS Runtime Error at: " << file << ":" << line
                << std::endl;
        std::cerr << errorStr << " " << func << std::endl;

    }
}


#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
    err = cudaGetLastError(); // done twice intenstinoally
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}


#define CHECK_CUDNN checkCUDNN(__FILE__, __LINE__)
template <typename T>
void checkCUDNN(T err, const char* const file, const int line)
{
    cudnnStatus_t status(err);
    if(err != CUDNN_STATUS_SUCCESS) {
        std::cerr << "CUDNN Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudnnGetErrorString(err) << std::endl;
    }
}


/* Writes an array to a .txt file
 * Each value is on a new line with no comma
 * Previous comments of file are discarded
 * @param arr: array to write to file
 * @param size: size of array
 * @param filename: name of file to write to
 * @return: void
 */
template <typename T>
void arrayToFile(T *arr, int size, std::string filename)
{
    std::ofstream f(filename, std::ofstream::trunc);
    for (int i = 0; i < size; i++)
        f << arr[i] << "\n";
    f.close();
}