#include <iostream>

#include <CL/cl.h>

#include "kernel_utils.h"

#ifndef KERNELS_DIR
#define KERNELS_DIR
#endif

int main(int argc, char *argv[])
{
    // Base task
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);
    cl_platform_id *platform = new cl_platform_id[platformCount];
    clGetPlatformIDs(platformCount, platform, nullptr);
    for (cl_uint i = 0; i < platformCount; i++)
    {
        constexpr size_t maxLength = 128;
        char platformName[maxLength];
        clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, maxLength, platformName, nullptr);
        std::cout << platformName << std::endl;
    }

    // Basic initialization
    cl_device_id deviceId = nullptr;
    cl_uint deviceCount = 0;
    clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 1, &deviceId, &deviceCount);
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    // Task 1
    {
        // Setup program
        std::string source = KernelUtils::readKernelFile(KERNELS_DIR "hello.cl");
        const char *strings[] = {source.c_str()};
        cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
        clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
        cl_kernel kernel = clCreateKernel(program, "hello", nullptr);
        constexpr size_t globalWorkSize[] = {8};
        constexpr size_t localWorkSize[] = {4, 4};

        // Execute program
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);

        clReleaseKernel(kernel);
        clReleaseProgram(program);
    }

    // Task 2
    {
        constexpr size_t elementsCount = 8;
        cl_uint array[elementsCount];
        for (size_t i = 0; i < elementsCount; i++)
        {
            array[i] = 10;
        }

        // Setup program
        std::string source = KernelUtils::readKernelFile(KERNELS_DIR "arraySum.cl");
        const char *strings[] = {source.c_str()};
        cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
        clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
        cl_kernel kernel = clCreateKernel(program, "arraySum", nullptr);
        constexpr size_t globalWorkSize[] = {elementsCount};

        // Create buffer
        cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, elementsCount * sizeof(cl_int), nullptr, nullptr);
        clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, elementsCount * sizeof(cl_int), array, 0, nullptr, nullptr);
        clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffer);

        // Execute program
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
        clFinish(queue);
        clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, elementsCount * sizeof(cl_int), array, 0, nullptr, nullptr);

        // Print result
        for (size_t i = 0; i < elementsCount; i++)
        {
            std::cout << "array[" << i << "] = " << array[i] << std::endl;
        }

        clReleaseMemObject(buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
    }

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    delete[] platform;
}
