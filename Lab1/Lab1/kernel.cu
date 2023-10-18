#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void kernel(int* _res, int*_arr)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    _res[index] = _arr[index] + index;
    printf("Hello, world! b %d , t %d , g %d \n", blockIdx.x, threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x);
}

void add(int* _c, int* _arrPtr, const int _size)
{
    int* devC = 0;
    int* devArr = 0;
    cudaMalloc((void**)&devC, _size * sizeof(int));
    cudaMalloc((void**)&devArr, _size * sizeof(int));
    cudaMemcpy(devArr, _arrPtr, _size * sizeof(int), cudaMemcpyHostToDevice);
    kernel <<<2, _size / 2 >>> (devC, devArr);
    cudaDeviceSynchronize();
    cudaMemcpy(_c, devC, _size * sizeof(int), cudaMemcpyDeviceToHost);

}

int main()
{
    const int size = 4;
    int arr[size] = { 1,2,3,4 };
    int res[size] = {0};
    add(res, arr, size);
    for (int index = 0; index < size; ++index)
    {
        printf(" %d ", res[index]);
    }
    printf("\n");
    return 0;
}
