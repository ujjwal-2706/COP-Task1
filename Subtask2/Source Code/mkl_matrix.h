#include <iostream>
#include "mkl.h"

double* matrixMultiplyMKL(int m,int n, int k,double alpha, double beta,double* A,double* B, double* C)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                 m,  n, k,  alpha, A, k, B, n, beta, C, n);
    // here CblasRowMajor means that the elements in the matrices are stored in row major
    // contiguous order 
    // This will basically compute the value of C = (alpha)AB + (beta)C
    return C;
}
 
