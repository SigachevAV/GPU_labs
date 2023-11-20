#include <random>
#include <cmath>
#include <string>
#include "taxpy_interface.h"
#include "cuda_taxpy_template.h"
#include "test.h"

template <typename T>
bool CalcTest(int _num)
{
    omp_set_num_threads(1);
    int vec_size = _num;
    T* vecA = get_rand_vector<float>(vec_size);
    T* vecB = get_rand_vector<float>(vec_size);

    //run
    T* copyB = copy_vector(vec_size, vecB);
    //GPU res in vecB
    int threadsPerBlock = 256;
    int blocksPerGrid = (vec_size + threadsPerBlock - 1) / threadsPerBlock;
    cuda_t_axpy<T>(vec_size, vecA, 2, vecB, 3, 4.53, blocksPerGrid, threadsPerBlock);

    //CPU res in copyB
    taxpy<T>(vec_size, vecA, 2, copyB, 3, 4.53);
    //data destruction
    delete[] vecA;
    delete[] vecB;
    delete[] copyB;
    return is_equal(vec_size, vecB, copyB);

}

void Run_tests() {

    test((char*)"FLOAT CPU AND GPU 100 ELEMENTS", []() {
        return CalcTest<float>(100);
        });

    test((char*)"FLOAT CPU OMP4 AND GPU 1000 ELEMENTS", []() {
        return CalcTest<float>(1000);
        });

    test((char*)"FLOAT CPU OMP4 AND GPU RANDOM PARAMETERS", []() {
        return CalcTest<float>(rand() % 1800 + 200);
        });

    test((char*)"DOUBLE CPU AND GPU 100 ELEMENTS", []() {
        return CalcTest<float>(100);
        });
    test((char*)"DOUBLE CPU OMP4 AND GPU 1000 ELEMENTS", []() {
        return CalcTest<float>(1000);
        });
    test((char*)"DOUBLE CPU OMP4 AND GPU RANDOM PARAMETERS", []() {
        return CalcTest<float>(rand() % 1800 + 200);
        });
}

int main() {
    //tests
    Run_tests();
    int blocksPerGrid;
    int threadsPerBlock = 256;
    int num_elements;
    //Experiments
    //GetResultExp(int vectorSize, int Xinc, int Yinc, T alpfa, char* descr, int omp_thread_nom)


    for (int i = 0; i < 3; i++)
    {
        num_elements = std::pow(10, 4+i*2);
        blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
        GetResultExp<double>(num_elements, 7, 3, 5.31432, (char*)("DOUBLE EXPERIMENT "+ std::to_string(num_elements) +" ELEMENTS").c_str(), 6, blocksPerGrid, threadsPerBlock).out();
    }

    for (int i = 0; i < 3; i++)
    {
        num_elements = std::pow(10, 4 + i * 2);
        blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
        GetResultExp<float>(num_elements, 7, 3, 5.31432, (char*)("FLOAT EXPERIMENT " + std::to_string(num_elements) + " ELEMENTS").c_str(), 6, blocksPerGrid, threadsPerBlock).out();
    }

    printf("\n\n\t\t\t\x1b[32;47mBLOCKS CHANGE EXPERIMENTS DOUBLE\x1b[0m\n\n");
    //1
    num_elements = 6000000;

    for (int i = 3; i < 9; i++)
    {
        threadsPerBlock = std::pow(2, i);
        blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
        GetResultExp<double>(num_elements, 7, 3, 5.31432, (char*)("DOUBLE EXPERIMENT "+ std::to_string(threadsPerBlock) +" BLOCKS IN GRID").c_str(), blocksPerGrid, threadsPerBlock).out();
    }

    printf("\n\n\t\t\t\x1b[32;47mBLOCKS CHANGE EXPERIMENTS FLOAT\x1b[0m\n\n");
    for (int i = 3; i < 9; i++)
    {
        threadsPerBlock = std::pow(2, i);
        blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
        GetResultExp<float>(num_elements, 7, 3, 5.31432, (char*)("FLOAT EXPERIMENT " + std::to_string(threadsPerBlock) + " BLOCKS IN GRID").c_str(), blocksPerGrid, threadsPerBlock).out();
    }
}
