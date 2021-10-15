#pragma once

#include <CL/cl.h>
#include <omp.h>

#include "kernel_utils.h"

namespace BLAS
{
void saxpy(int n, float a, float *x, int incx, float *y, int incy);

void daxpy(int n, double a, double *x, int incx, double *y, int incy);

void saxpyOpenMP(int n, float a, float *x, int incx, float *y, int incy);

void daxpyOpenMP(int n, double a, double *x, int incx, double *y, int incy);

void saxpyOpenCL(int n, float a, float *x, int incx, float *y, int incy, cl_device_id deviceId,
                 double *computationTime);

void daxpyOpenCL(int n, double a, double *x, int incx, double *y, int incy, cl_device_id deviceId,
                 double *computationTime);
} // namespace BLAS