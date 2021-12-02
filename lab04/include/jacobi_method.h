#pragma once

#include <string>
#include <vector>

#include <omp.h>

#include <CL/cl.h>

#include "kernel_utils.h"

namespace JacobiMethod
{
bool isAccurate(const float *xCurrent, const float *xPrevious, size_t n, float epsilon);

bool isAccurateSolution(float *a, float *b, float *x, int n, float epsilon);

void jacobiMethodOpenCL(float *a, float *b, float *x, int n, float epsilon, size_t iterationsCount,
                        cl_device_id deviceId, double *computationTime);
}; // namespace JacobiMethod