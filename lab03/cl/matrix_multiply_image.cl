#define BLOCK_SIZE 16

__kernel void matrix_multiply_image(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c, int n)
{
    __local float a_block[BLOCK_SIZE][BLOCK_SIZE];
    __local float b_block[BLOCK_SIZE][BLOCK_SIZE];
    int row = get_global_id(1);
    int column = get_global_id(0);
    int local_row = get_local_id(1);
    int local_column = get_local_id(0);
    int blocks_count = n / BLOCK_SIZE;
    float localResult = 0;
    for (int i = 0; i < blocks_count; i++)
    {
        a_block[local_row][local_column] = read_imagef(a, (int2)(BLOCK_SIZE * i + local_column, row)).x;
        b_block[local_row][local_column] = read_imagef(b, (int2)(column, BLOCK_SIZE * i + local_row)).x;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            localResult += a_block[local_row][j] * b_block[j][local_column];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    write_imagef(c, (int2)(column, row), localResult);
}
