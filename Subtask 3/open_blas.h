#include <iostream>
#include "cblas.h"

// Points to be noted so far
// 1. *double is pointer pointing to double, also an input of openblas and mkl
// 2. first use malloc to allocate space
// 3. after that for accessing purposes treat it like an array

double* matrixMultiplyOpenBlas(int m,int n, int k,double alpha, double beta,double* A,double* B, double* C)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                 m,  n, k,  alpha, A, k, B, n, beta, C, n);
    // here CblasRowMajor means that the elements in the matrices are stored in
    // contiguous order 
    // This will basically compute the value of C = (alpha)AB + (beta)C
    return C;
}
 