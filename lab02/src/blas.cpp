#include "blas.h"

namespace BLAS
{
void saxpy(int n, float a, float *x, int incx, float *y, int incy)
{
    for (int i = 0; i < n; i++)
    {
        y[i * incy] = y[i * incy] + a * x[i * incx];
    }
}

void daxpy(int n, double a, double *x, int incx, double *y, int incy)
{
    for (int i = 0; i < n; i++)
    {
        y[i * incy] = y[i * incy] + a * x[i * incx];
    }
}

void saxpyOpenMP(int n, float a, float *x, int incx, float *y, int incy)
{
#pragma omp parallel for default(none) shared(n, y, incy, a, x, incx)
    for (int i = 0; i < n; i++)
    {
        y[i * incy] = y[i * incy] + a * x[i * incx];
    }
}

void daxpyOpenMP(int n, double a, double *x, int incx, double *y, int incy)
{
#pragma omp parallel for default(none) shared(n, y, incy, a, x, incx)
    for (int i = 0; i < n; i++)
    {
        y[i * incy] = y[i * incy] + a * x[i * incx];
    }
}

void saxpyOpenCL(int n, float a, float *x, int incx, float *y, int incy, cl_device_id deviceId, double *computationTime)
{
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    // Setup program
    std::string source = KernelUtils::readKernelFile(KERNELS_DIR "blas.cl");
    const char *strings[] = {source.c_str()};
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "saxpy", nullptr);

    // Create buffer
    auto xBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, n * incx * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, xBuffer, CL_TRUE, 0, n * incx * sizeof(float), x, 0, nullptr, nullptr);
    auto yBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, n * incy * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, yBuffer, CL_TRUE, 0, n * incy * sizeof(float), y, 0, nullptr, nullptr);

    // Setup arguments
    clSetKernelArg(kernel, 0, sizeof(int), &n);
    clSetKernelArg(kernel, 1, sizeof(float), &a);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &xBuffer);
    clSetKernelArg(kernel, 3, sizeof(int), &incx);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &yBuffer);
    clSetKernelArg(kernel, 5, sizeof(int), &incy);

    size_t group;
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);
    auto globalWorkSize = static_cast<size_t>(n);
    double begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    // clFinish(queue);
    double end = omp_get_wtime();
    if (computationTime != nullptr)
    {
        *computationTime = end - begin;
    }

    clEnqueueReadBuffer(queue, yBuffer, CL_TRUE, 0, n * incy * sizeof(float), y, 0, nullptr, nullptr);

    clReleaseMemObject(xBuffer);
    clReleaseMemObject(yBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void daxpyOpenCL(int n, double a, double *x, int incx, double *y, int incy, cl_device_id deviceId,
                 double *computationTime)
{
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    // Setup program
    std::string source = KernelUtils::readKernelFile(KERNELS_DIR "blas.cl");
    const char *strings[] = {source.c_str()};
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "daxpy", nullptr);

    // Create buffer
    auto xBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, n * incx * sizeof(double), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, xBuffer, CL_TRUE, 0, n * incx * sizeof(double), x, 0, nullptr, nullptr);
    auto yBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, n * incy * sizeof(double), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, yBuffer, CL_TRUE, 0, n * incy * sizeof(double), y, 0, nullptr, nullptr);

    // Setup arguments
    clSetKernelArg(kernel, 0, sizeof(int), &n);
    clSetKernelArg(kernel, 1, sizeof(double), &a);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &xBuffer);
    clSetKernelArg(kernel, 3, sizeof(int), &incx);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &yBuffer);
    clSetKernelArg(kernel, 5, sizeof(int), &incy);

    size_t group;
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);
    auto globalWorkSize = static_cast<size_t>(n);
    double begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    // clFinish(queue);
    double end = omp_get_wtime();
    if (computationTime != nullptr)
    {
        *computationTime = end - begin;
    }

    clEnqueueReadBuffer(queue, yBuffer, CL_TRUE, 0, n * incy * sizeof(double), y, 0, nullptr, nullptr);

    clReleaseMemObject(xBuffer);
    clReleaseMemObject(yBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

} // namespace BLAS