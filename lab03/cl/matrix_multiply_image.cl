#define BLOCK_SIZE 16

__kernel void matrix_multiply_image(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c, int m, int n, int k)
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
        a_block[local_column][local_row] = read_imagef(a, (int2)(BLOCK_SIZE * i + local_row, column)).x;
        b_block[local_column][local_row] = (b, (int2)(row, BLOCK_SIZE * i + local_column)).x;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            localResult += a_block[local_column][j] * b_block[j][local_row];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    write_imagef(c, (int2)(row, column), localResult);
}