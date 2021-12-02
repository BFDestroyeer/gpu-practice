#include <iostream>

#include <CL/cl.h>

#include "array_utils.h"
#include "jacobi_method.h"

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
    constexpr int n = 16;

    std::vector<float> a(n * n);
    std::vector<float> b(n);
    std::vector<float> x(n);
    ArrayUtils::fillWithRandomValues(a);
    ArrayUtils::makeDiagonallyDominant(a, n);
    ArrayUtils::fillWithRandomValues(b);

    {
        double begin = omp_get_wtime();
        JacobiMethod::jacobiMethodOpenCL(a.data(), b.data(), x.data(), n, 0.01, 10, gpuDeviceId, nullptr);
        double end = omp_get_wtime();
        std::cout << "OpenCL: " << (end - begin) << std::endl;
    }
}