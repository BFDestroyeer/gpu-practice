__kernel void saxpy(int n, float a, __global float *x, int incx, __global float *y, int incy) {
    printf("saxpy\n");
    int global_id = get_global_id(0);
    y[global_id * incy] += a * x[global_id * incx];
}

__kernel void daxpy(int n, double a, __global double *x, int incx, __global double *y, int incy) {
    printf("daxpy\n");
    //int global_id = get_global_id(0);
    //y[global_id * incy] += a * x[global_id * incx];
}