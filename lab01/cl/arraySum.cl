__kernel void arraySum(__global int *array)
{
	int global_id = get_global_id(0);
	array[global_id] += global_id;
}