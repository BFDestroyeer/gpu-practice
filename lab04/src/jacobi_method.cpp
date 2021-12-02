#include "jacobi_method.h"

namespace JacobiMethod
{
bool isAccurate(const float *xCurrent, const float *xPrevious, size_t n, float epsilon)
{
    float accuracy = 0;
#pragma omp parallel for default(none), shared(xCurrent, xPrevious, n) reduction(+ : accuracy)
    for (size_t i = 0; i < n; i++)
    {
        accuracy += (xCurrent[i] - xPrevious[i]) * (xCurrent[i] - xPrevious[i]);
    }
    return accuracy < (epsilon * epsilon) / 4;
}

float deviation(const float *a, const float *b, const float *x, int n, float epsilon)
{
    float result = 0;
    for (size_t i = 0; i < n; i++)
    {
        float local_deviation = 0;
        for (size_t j = 0; j < n; j++)
        {
            local_deviation += a[j * n + i] * x[j];
        }
        local_deviation -= b[i];
        result += local_deviation * local_deviation;
    }
    return result;
}

void jacobiMethodOpenCL(float *a, float *b, float *x, int n, float epsilon, size_t iterationsCount,
                        cl_device_id deviceId, double *computationTime)
{
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    // Setup program
    std::string source = KernelUtils::readKernelFile(KERNELS_DIR "jacobi_method.cl");
    const char *strings[] = {source.c_str()};
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "jacobi_method", nullptr);

    // Create buffer
    auto aBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, n * n * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, aBuffer, CL_TRUE, 0, n * n * sizeof(float), a, 0, nullptr, nullptr);
    auto bBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, bBuffer, CL_TRUE, 0, n * sizeof(float), b, 0, nullptr, nullptr);
    auto xInputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(float), nullptr, nullptr);
    auto xOutputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, xOutputBuffer, CL_TRUE, 0, n * sizeof(float), x, 0, nullptr, nullptr);

    // Setup arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &xInputBuffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &xOutputBuffer);
    clSetKernelArg(kernel, 4, sizeof(int), &n);
    clSetKernelArg(kernel, 5, sizeof(float), &epsilon);

    size_t group;
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);

    const auto globalWorkSize = static_cast<size_t>(n);

    double begin = omp_get_wtime();

    size_t iteration = 0;
    std::vector<float> xPrevious(n, 0);
    std::vector<float> xCurrent(b, b + n);
    do
    {
        xPrevious = xCurrent;
        clEnqueueWriteBuffer(queue, xInputBuffer, CL_TRUE, 0, n * sizeof(float), xPrevious.data(), 0, nullptr, nullptr);

        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
        clFinish(queue);

        clEnqueueReadBuffer(queue, xOutputBuffer, CL_TRUE, 0, n * sizeof(float), xCurrent.data(), 0, nullptr, nullptr);
    } while (++iteration < iterationsCount && !isAccurate(xCurrent.data(), xPrevious.data(), n, epsilon));
    std::copy(xCurrent.begin(), xCurrent.end(), x);

    double end = omp_get_wtime();
    if (computationTime != nullptr)
    {
        *computationTime = end - begin;
    }

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(bBuffer);
    clReleaseMemObject(xInputBuffer);
    clReleaseMemObject(xOutputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
} // namespace JacobiMethod