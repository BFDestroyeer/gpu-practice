#define BLOCK_SIZE 16

__kernel void matrix_multiply_block(__global float *a, __global float *b, __global float *c, int m, int n, int k)
{
    __local float a_block[BLOCK_SIZE][BLOCK_SIZE];
    __local float b_block[BLOCK_SIZE][BLOCK_SIZE];
    int row = get_global_id(0);
    int column = get_global_id(1);
    int local_row = get_local_id(0);
    int local_column = get_local_id(1);
    int blocks_count = m / BLOCK_SIZE;
    float localResult = 0;
    for (int i = 0; i < blocks_count; i++)
    {
        a_block[local_column][local_row] = a[column * m + BLOCK_SIZE * i + local_row];
        b_block[local_column][local_row] = b[(BLOCK_SIZE * i + local_column) * n + row];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            localResult += a_block[local_column][j] * b_block[j][local_row];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[k * column + row] = localResult;
}