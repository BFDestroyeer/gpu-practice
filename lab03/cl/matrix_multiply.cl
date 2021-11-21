__kernel void matrix_multiply(__global float *a, __global float *b, __global float *c, int m, int n, int k)
{
    int row = get_global_id(0);
    int column = get_global_id(1);
    float localResult = 0;
    int i;
    for (i = 0; i < n; i++)
        localResult += a[row * n + i] * b[column + k * i];
    c[k * row + column] = localResult;
}