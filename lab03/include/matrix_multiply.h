#pragma once

#include <CL/opencl.h>

#include <omp.h>

#include "kernel_utils.h"

namespace MatrixUtils
{
void matrixMultiply(const float *a, const float *b, float *c, size_t n);

void matrixMultiplyOpenMp(const float *a, const float *b, float *c, size_t n);

void matrixMultiplyOpenCl(float *a, float *b, float *c, int n, cl_device_id deviceId, double *computationTime);

void matrixMultiplyOpenClBlock(float *a, float *b, float *c, int n, cl_device_id deviceId, double *computationTime);

void matrixMultiplyOpenClImage(float *a, float *b, float *c, int n, cl_device_id deviceId, double *computationTime);
} // namespace MatrixUtils