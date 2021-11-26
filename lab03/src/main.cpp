#include <iostream>
#include <vector>

#include <CL/cl.h>

#include <omp.h>

#include "array_utils.h"
#include "matrix_multiply.h"

constexpr int BLOCK_SIZE = 16;

int main(int argc, char *argv[])
{
    // Setup platform
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);
    auto *platform = new cl_platform_id[platformCount];
    clGetPlatformIDs(platformCount, platform, nullptr);

    // Setup devices
    cl_uint deviceCount = 0;

    cl_device_id gpuDeviceId = nullptr;
    char gpuDeviceName[128] = {0};
    clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 1, &gpuDeviceId, &deviceCount);
    clGetDeviceInfo(gpuDeviceId, CL_DEVICE_NAME, 128, gpuDeviceName, nullptr);
    std::cout << "GPU: " << gpuDeviceName << std::endl;

    // Matrix initialization
    constexpr int m = BLOCK_SIZE * 1;
    constexpr int n = BLOCK_SIZE * 1;
    constexpr int k = BLOCK_SIZE * 1;

    std::vector<float> a(m * n);
    std::vector<float> b(n * k);
    std::vector<float> cReference(m * k);
    ArrayUtils::fillWithRandomValues(a);
    ArrayUtils::fillWithRandomValues(b);

    {
        double begin = omp_get_wtime();
        MatrixUtils::matrixMultiply(a.data(), b.data(), cReference.data(), m, n, k);
        double end = omp_get_wtime();
        std::cout << "Sequential: " << (end - begin) << std::endl;
    }

    {
        std::vector<float> c(m * k);
        double begin = omp_get_wtime();
        MatrixUtils::matrixMultiplyOpenMp(a.data(), b.data(), c.data(), m, n, k);
        double end = omp_get_wtime();
        std::cout << "OpenMP: " << (end - begin) << std::endl;
        std::cout << "Equal to reference: " << ArrayUtils::checkEquality(c, cReference) << std::endl;
    }

    {
        std::vector<float> c(m * k);
        double computationTime;
        double begin = omp_get_wtime();
        MatrixUtils::matrixMultiplyOpenCl(a.data(), b.data(), c.data(), m, n, k, gpuDeviceId, &computationTime);
        double end = omp_get_wtime();
        std::cout << "OpenCL: " << (end - begin) << std::endl;
        std::cout << "OpenCL computation: " << computationTime << std::endl;
        std::cout << "Equal to reference: " << ArrayUtils::checkEquality(c, cReference) << std::endl;
    }

    {
        std::vector<float> c(m * k);
        double computationTime;
        double begin = omp_get_wtime();
        MatrixUtils::matrixMultiplyOpenClBlock(a.data(), b.data(), c.data(), m, n, k, gpuDeviceId, &computationTime);
        double end = omp_get_wtime();
        std::cout << "OpenCL block: " << (end - begin) << std::endl;
        std::cout << "OpenCL block computation: " << computationTime << std::endl;
        std::cout << "Equal to reference: " << ArrayUtils::checkEquality(c, cReference) << std::endl;
    }

    {
        std::vector<float> c(m * k);
        double computationTime;
        double begin = omp_get_wtime();
        MatrixUtils::matrixMultiplyOpenClImage(a.data(), b.data(), c.data(), m, n, k, gpuDeviceId, &computationTime);
        double end = omp_get_wtime();
        std::cout << "OpenCL image: " << (end - begin) << std::endl;
        std::cout << "OpenCL image computation: " << computationTime << std::endl;
        std::cout << "Equal to reference: " << ArrayUtils::checkEquality(c, cReference) << std::endl;
    }
}