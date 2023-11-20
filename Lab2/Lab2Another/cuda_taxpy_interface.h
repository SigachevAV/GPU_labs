#ifndef CUDA_TAXPY_INTERFACE_H
#define CUDA_TAXPY_INTERFACE_H
#include "cuda_taxpy_template.h"

template <typename T>
double cuda_t_axpy(int n, T* X, int Xinc, T* Y, int Yinc, T alpfa, int blocksPerGrid, int threadsPerBlock){return -1.0;}

template <>
double cuda_t_axpy<double>(int n, double* X, int Xinc, double* Y, int Yinc, double alpfa, int blocksPerGrid, int threadsPerBlock){
    return cuda_daxpy(n, X, Xinc, Y, Yinc, alpfa, blocksPerGrid, threadsPerBlock);
}

template <>
double cuda_t_axpy<float>(int n, float* X, int Xinc, float* Y, int Yinc, float alpfa, int blocksPerGrid, int threadsPerBlock){
    return cuda_faxpy(n, X, Xinc, Y, Yinc, alpfa, blocksPerGrid, threadsPerBlock);
}

#endif //CUDA_TAXPY_INTERFACE_H