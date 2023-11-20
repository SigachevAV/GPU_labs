#ifndef CUDA_TAXPY_TEMPLATE_H
#define CUDA_TAXPY_TEMPLATE_H


double cuda_daxpy(int n, double* X, int Xinc, double* Y, int Yinc, double alpfa, int blocksPerGrid, int threadsPerBlock);
double cuda_faxpy(int n, float* X, int Xinc, float* Y, int Yinc, float alpfa, int blocksPerGrid, int threadsPerBlock);


#endif //CUDA_TAXPY_TEMPLATE_H