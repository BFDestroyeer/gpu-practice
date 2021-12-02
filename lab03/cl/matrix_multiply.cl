__kernel void matrix_multiply(__global float *a, __global float *b, __global float *c, int n)
{
    int row = get_global_id(1);
    int column = get_global_id(0);
    float localResult = 0;
    int i;
    for (i = 0; i < n; i++) {
        localResult += a[row * n + i] * b[column + n * i];
    }
    c[n * row + column] = localResult;
}