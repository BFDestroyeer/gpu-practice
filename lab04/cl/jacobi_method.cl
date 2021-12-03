__kernel void jacobi_method(__read_only image2d_t a, __global float *b, __global float *x_input, __global float *x_output, int n, float epsilon)
{
    int i = get_global_id(0);
    float localResult = 0;
    for (int j = 0; j < n; j++) {
        localResult += i != j ? read_imagef(a, (int2) (i, j)).x * x_input[j] : 0;
    }
    x_output[i] = (b[i] - localResult) / read_imagef(a, (int2) (i, i)).x;
}