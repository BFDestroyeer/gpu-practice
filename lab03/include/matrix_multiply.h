#pragma once

#include <CL/opencl.h>

#include <omp.h>

#include "kernel_utils.h"

namespace MatrixUtils
{
void matrixMultiply(const float *a, const float *b, float *c, size_t m, size_t n, size_t k);

void matrixMultiplyOpenMp(const float *a, const float *b, float *c, size_t m, size_t n, size_t k);

void matrixMultiplyOpenCl(float *a, float *b, float *c, int m, int n, int k, cl_device_id deviceId,
                          double *computationTime);
} // namespace MatrixUtils