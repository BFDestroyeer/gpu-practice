__kernel void jacobi_method(__global float *a, __global float *b, __global float *x_input, __global float *x_output, int n, float epsilon)
{
    int i = get_global_id(0);
    float localResult = 0;
    for (int j = 0; j < n; j++) {
        localResult += i != j ? a[j * n + i] * x_input[j] : 0;
    }
    x_output[i] = (b[i] - localResult) / a[i * n + i];
}