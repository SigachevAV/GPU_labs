#ifndef TAXPY_INTERFACE__H
#define TAXPY_INTERFACE__H

#include <stdio.h>
#include <random>
#include <omp.h>
#include "cuda_taxpy_interface.h"


//Result struct
struct timeResult {
    int vectorSize;
    double seqTimeResult;
    double ompTimeResult;
    double cudaTimeResultWithLoad;
    double cudaTimeResultWithoutLoad;
    char* description;


    void out() {
        printf("\t\t%s\n", description);
        printf("\tseqTimeResult \t\t\t\t%lfsec\n", seqTimeResult);
        printf("\tompTimeResult \t\t\t\t%lfsec\n", ompTimeResult);
        printf("\tcudaTimeResultWithLoad \t\t\t%lfsec\n", cudaTimeResultWithLoad);
        printf("\tcudaTimeResultWithoutLoad \t\t%lfsec\n", cudaTimeResultWithoutLoad);
        printf("\t----------------------------------------------------\n\n");
    }
};

struct timeResultCuda {
    int vectorSize;
    double cudaTimeResultWithLoad;
    double cudaTimeResultWithoutLoad;
    char* description;


    void out() {
        printf("\t\t%s\n", description);
        printf("\tcudaTimeResultWithLoad \t\t\t%lfsec\n", cudaTimeResultWithLoad);
        printf("\tcudaTimeResultWithoutLoad \t\t%lfsec\n", cudaTimeResultWithoutLoad);
        printf("\t----------------------------------------------------\n\n");
    }
};

//Vector operations
template <typename T>
void vector_outs(int size, T* vec) {
    printf("{");
    for (int i = 0; i < size; i++)
        printf(" %lf", vec[i], " ");
    printf("}\n");
}

template <typename T>
T* get_rand_vector(int size) {
    T* res = new T[size];
    for (int i = 0; i < size; i++) {
        res[i] = (T)rand() / (T)rand();
    }
    return res;
}

template <typename T>
T* copy_vector(int size, const T* vector) {
    T* res = new T[size];

    for (int i = 0; i < size; i++)
        res[i] = vector[i];

    return res;
}


//OpenMP version (for sequential execution set parameter nom_threads = 1)
template <typename T>
void taxpy(int n, T* X, int Xinc, T* Y, int Yinc, T alpfa) {
    int op_nom = std::ceil((T)((double)(n) / (double)(std::max(Xinc, Yinc))));
#pragma omp parallel for
    for (int i = 0; i < op_nom; i++) {
        Y[i * Yinc] += alpfa * X[i * Xinc];
    }
}

template <typename T>
timeResult GetResultExp(int vectorSize, int Xinc, int Yinc, T alpfa, char* descr, int omp_thread_nom, int blocksPerGrid, int threadsPerBlock) {
    T* vecA = get_rand_vector<T>(vectorSize);
    T* vecB = get_rand_vector<T>(vectorSize);

    //////////////SEQ
    omp_set_num_threads(1);
    double TimeStart = omp_get_wtime();
    taxpy(vectorSize, vecA, Xinc, vecB, Yinc, alpfa);
    double TimeEnd = omp_get_wtime();
    double seqResult = TimeEnd - TimeStart;

    //////////////OMP_PARALLEL
    omp_set_num_threads(omp_thread_nom);
    TimeStart = omp_get_wtime();
    taxpy(vectorSize, vecA, Xinc, vecB, Yinc, alpfa);
    TimeEnd = omp_get_wtime();
    double ompResult = TimeEnd - TimeStart;

    ////////////CUDA
    TimeStart = omp_get_wtime();
    double CudaTimeWithoutLoad = cuda_t_axpy<T>(vectorSize, vecA, Xinc, vecB, Yinc, alpfa, blocksPerGrid, threadsPerBlock);
    TimeEnd = omp_get_wtime();
    double CudaTimeWithLoad = TimeEnd - TimeStart;

    timeResult res;
    {
        res.vectorSize = vectorSize;
        res.seqTimeResult = seqResult;
        res.ompTimeResult = ompResult;
        res.cudaTimeResultWithLoad = CudaTimeWithLoad;
        res.cudaTimeResultWithoutLoad = CudaTimeWithoutLoad;
        res.description = descr;
    }

    delete[] vecA;
    delete[] vecB;

    return res;
}

template <typename T>
timeResultCuda GetResultExp(int vectorSize, int Xinc, int Yinc, T alpfa, char* descr, int blocksPerGrid, int threadsPerBlock) {
    T* vecA = get_rand_vector<T>(vectorSize);
    T* vecB = get_rand_vector<T>(vectorSize);
    
    double TimeStart = omp_get_wtime();
    double CudaTimeWithoutLoad = cuda_t_axpy<T>(vectorSize, vecA, Xinc, vecB, Yinc, alpfa, blocksPerGrid, threadsPerBlock);
    double TimeEnd = omp_get_wtime();
    double CudaTimeWithLoad = TimeEnd - TimeStart;

    timeResultCuda res;
    {
        res.vectorSize = vectorSize;
        res.cudaTimeResultWithLoad = CudaTimeWithLoad;
        res.cudaTimeResultWithoutLoad = CudaTimeWithoutLoad;
        res.description = descr;
    }

    delete[] vecA;
    delete[] vecB;

    return res;
}


#endif //TAXPY_INTERFACE__H
