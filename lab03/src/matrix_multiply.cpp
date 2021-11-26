#include "matrix_multiply.h"

namespace MatrixUtils
{
void matrixMultiply(const float *a, const float *b, float *c, size_t m, size_t n, size_t k)
{
    for (size_t row = 0; row < m; row++)
    {
        for (size_t column = 0; column < k; column++)
        {
            float *localResult = c + k * row + column;
            *localResult = 0;
            for (size_t i = 0; i < n; i++)
            {
                *localResult += a[row * n + i] * b[column + k * i];
            }
        }
    }
}

void matrixMultiplyOpenMp(const float *a, const float *b, float *c, size_t m, size_t n, size_t k)
{
#pragma omp parallel for default(none), shared(a, b, c, m, n, k)
    for (size_t row = 0; row < m; row++)
    {
        for (size_t column = 0; column < k; column++)
        {
            float *localResult = c + k * row + column;
            *localResult = 0;
            for (size_t i = 0; i < n; i++)
            {
                *localResult += a[row * n + i] * b[column + k * i];
            }
        }
    }
}

void matrixMultiplyOpenCl(float *a, float *b, float *c, int m, int n, int k, cl_device_id deviceId,
                          double *computationTime)
{
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    // Setup program
    std::string source = KernelUtils::readKernelFile(KERNELS_DIR "matrix_multiply.cl");
    const char *strings[] = {source.c_str()};
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "matrix_multiply", nullptr);

    // Create buffer
    auto aBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, m * n * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, aBuffer, CL_TRUE, 0, m * n * sizeof(float), a, 0, nullptr, nullptr);
    auto bBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, n * k * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, bBuffer, CL_TRUE, 0, n * k * sizeof(float), b, 0, nullptr, nullptr);
    auto cBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, m * k * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, cBuffer, CL_TRUE, 0, m * k * sizeof(float), c, 0, nullptr, nullptr);

    // Setup arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cBuffer);
    clSetKernelArg(kernel, 3, sizeof(int), &m);
    clSetKernelArg(kernel, 4, sizeof(int), &n);
    clSetKernelArg(kernel, 5, sizeof(int), &k);

    size_t group;
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);

    const size_t globalWorkSize[] = {static_cast<size_t>(m), static_cast<size_t>(k)};
    const size_t localWorkSize[] = {16, 16};
    double begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    double end = omp_get_wtime();
    if (computationTime != nullptr)
    {
        *computationTime = end - begin;
    }

    clEnqueueReadBuffer(queue, cBuffer, CL_TRUE, 0, m * k * sizeof(float), c, 0, nullptr, nullptr);

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(bBuffer);
    clReleaseMemObject(cBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void matrixMultiplyOpenClBlock(float *a, float *b, float *c, int m, int n, int k, cl_device_id deviceId,
                               double *computationTime)
{
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    // Setup program
    std::string source = KernelUtils::readKernelFile(KERNELS_DIR "matrix_multiply_block.cl");
    const char *strings[] = {source.c_str()};
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "matrix_multiply_block", nullptr);

    // Create buffer
    auto aBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, m * n * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, aBuffer, CL_TRUE, 0, m * n * sizeof(float), a, 0, nullptr, nullptr);
    auto bBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, n * k * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, bBuffer, CL_TRUE, 0, n * k * sizeof(float), b, 0, nullptr, nullptr);
    auto cBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, m * k * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, cBuffer, CL_TRUE, 0, m * k * sizeof(float), c, 0, nullptr, nullptr);

    // Setup arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cBuffer);
    clSetKernelArg(kernel, 3, sizeof(int), &m);
    clSetKernelArg(kernel, 4, sizeof(int), &n);
    clSetKernelArg(kernel, 5, sizeof(int), &k);

    size_t group;
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);

    const size_t globalWorkSize[] = {static_cast<size_t>(m), static_cast<size_t>(k)};
    const size_t localWorkSize[] = {16, 16};
    double begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    double end = omp_get_wtime();
    if (computationTime != nullptr)
    {
        *computationTime = end - begin;
    }

    clEnqueueReadBuffer(queue, cBuffer, CL_TRUE, 0, m * k * sizeof(float), c, 0, nullptr, nullptr);

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(bBuffer);
    clReleaseMemObject(cBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void matrixMultiplyOpenClImage(float *a, float *b, float *c, int m, int n, int k, cl_device_id deviceId,
                               double *computationTime)
{
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    // Setup program
    std::string source = KernelUtils::readKernelFile(KERNELS_DIR "matrix_multiply_image.cl");
    const char *strings[] = {source.c_str()};
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "matrix_multiply_image", nullptr);

    // Create image
    cl_image_format imageFormat;
    imageFormat.image_channel_order = CL_R;
    imageFormat.image_channel_data_type = CL_FLOAT;

    auto aImage = clCreateImage2D(context, CL_MEM_READ_ONLY, &imageFormat, static_cast<size_t>(n),
                                  static_cast<size_t>(m), 0, nullptr, nullptr);
    auto bImage = clCreateImage2D(context, CL_MEM_READ_ONLY, &imageFormat, static_cast<size_t>(k),
                                  static_cast<size_t>(n), 0, nullptr, nullptr);
    auto cImage = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &imageFormat, static_cast<size_t>(m),
                                  static_cast<size_t>(k), 0, nullptr, nullptr);
    size_t origin[] = {0, 0, 0};
    {
        size_t region[] = {static_cast<size_t>(n), static_cast<size_t>(m), 1};
        clEnqueueWriteImage(queue, aImage, CL_TRUE, origin, region, 0, 0, a, 0, nullptr, nullptr);
    }
    {
        size_t region[] = {static_cast<size_t>(k), static_cast<size_t>(n), 1};
        clEnqueueWriteImage(queue, bImage, CL_TRUE, origin, region, 0, 0, b, 0, nullptr, nullptr);
    }

    // Setup arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aImage);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bImage);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cImage);
    clSetKernelArg(kernel, 3, sizeof(int), &m);
    clSetKernelArg(kernel, 4, sizeof(int), &n);
    clSetKernelArg(kernel, 5, sizeof(int), &k);

    size_t group;
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);

    const size_t globalWorkSize[] = {static_cast<size_t>(m), static_cast<size_t>(k)};
    const size_t localWorkSize[] = {16, 16};
    double begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    double end = omp_get_wtime();
    if (computationTime != nullptr)
    {
        *computationTime = end - begin;
    }

    {
        size_t region[] = {static_cast<size_t>(m), static_cast<size_t>(k), 1};
        clEnqueueReadImage(queue, cImage, CL_TRUE, origin, region, 0, 0, c, 0, nullptr, nullptr);
    }

    clReleaseMemObject(aImage);
    clReleaseMemObject(bImage);
    clReleaseMemObject(cImage);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
} // namespace MatrixUtils