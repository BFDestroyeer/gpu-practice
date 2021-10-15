#include <iostream>
#include <vector>

#include <omp.h>

#include "array_utils.h"
#include "blas.h"

int main(int argc, char *argv[])
{
    constexpr int n = 256 * 60000;
    constexpr int incx = 3;
    constexpr int incy = 2;
    constexpr size_t xSize = n * incx;
    constexpr size_t ySize = n * incy;

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

    {
        constexpr float a = 5;

        std::vector<float> xIntial(xSize);
        ArrayUtils::fillWithStep(xIntial, 7.f, incx);
        std::vector<float> yIntial(ySize);
        ArrayUtils::fillWithStep(yIntial, 3.f, incy);
        std::vector<float> yReference;

        std::cout << std::endl << "[float]" << std::endl;
        // Sequential
        {
            auto y = yIntial;
            double begin = omp_get_wtime();
            BLAS::saxpy(n, a, xIntial.data(), incx, y.data(), incy);
            double end = omp_get_wtime();
            std::cout << "Sequential: " << (end - begin) << std::endl;
            yReference = y;
        }

        // OpenMP
        {
            auto y = yIntial;
            double begin = omp_get_wtime();
            BLAS::saxpyOpenMP(n, a, xIntial.data(), incx, y.data(), incy);
            double end = omp_get_wtime();
            std::cout << "OpenMP: " << (end - begin) << std::endl;
            std::cout << "Equal to reference: " << (yReference == y) << std::endl;
        }

        // OpenCL GPU
        {
            auto y = yIntial;
            double computationTime;
            double begin = omp_get_wtime();
            BLAS::saxpyOpenCL(n, a, xIntial.data(), incx, y.data(), incy, gpuDeviceId, &computationTime);
            double end = omp_get_wtime();
            std::cout << "OpenCL total: " << (end - begin) << std::endl;
            std::cout << "OpenCL computation: " << computationTime << std::endl;
            std::cout << "Equal to reference: " << (yReference == y) << std::endl;
        }
    }

    {
        constexpr double a = 5;

        std::vector<double> xIntial(xSize);
        ArrayUtils::fillWithStep(xIntial, 7., incx);
        std::vector<double> yIntial(ySize);
        ArrayUtils::fillWithStep(yIntial, 3., incy);
        std::vector<double> yReference;

        std::cout << std::endl << "[double]" << std::endl;
        // Sequential
        {
            auto y = yIntial;
            double begin = omp_get_wtime();
            BLAS::daxpy(n, a, xIntial.data(), incx, y.data(), incy);
            double end = omp_get_wtime();
            std::cout << "Sequential: " << (end - begin) << std::endl;
            yReference = y;
        }

        // OpenMP
        {
            auto y = yIntial;
            double begin = omp_get_wtime();
            BLAS::daxpyOpenMP(n, a, xIntial.data(), incx, y.data(), incy);
            double end = omp_get_wtime();
            std::cout << "OpenMP: " << (end - begin) << std::endl;
            std::cout << "Equal to reference: " << (yReference == y) << std::endl;
        }

        // OpenCL GPU
        {
            auto y = yIntial;
            double computationTime;
            double begin = omp_get_wtime();
            BLAS::daxpyOpenCL(n, a, xIntial.data(), incx, y.data(), incy, gpuDeviceId, &computationTime);
            double end = omp_get_wtime();
            std::cout << "OpenCL total: " << (end - begin) << std::endl;
            std::cout << "OpenCL computation: " << computationTime << std::endl;
            std::cout << "Equal to reference: " << (yReference == y) << std::endl;
        }
    }
}