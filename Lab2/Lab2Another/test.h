#include<functional>
#include <stdio.h>
#include <random>
#include <functional>

template<typename T>
bool is_equal(int n, const T* a, const T* b){
    for(int i = 0; i < n; i++){
        if(a[i]!=b[i]) return false;
    }
    return true;
}

void test(char* descr, std::function<bool()> testF){
    printf("\t\x1b[32;47mRUNNING-TEST\t\x1b[35;47m%s\x1b[0m\n",descr);
    if(testF)
    printf("\t\x1b[32;47mTEST-SUCCESSFUL\t\x1b[35;47m%s\x1b[0m\n\t--------------------------------------------\n",descr);
    else{
    printf("\t\x1b[32;47mTEST-FAILED\t\x1b[35;47m%s\n\x1b[0m\t--------------------------------------------\n",descr);
    exit(EXIT_FAILURE);
    }
}