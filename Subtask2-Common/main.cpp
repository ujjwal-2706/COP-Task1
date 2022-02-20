// This is the cpp file containing our main function, others are header files
#include <iostream>
#include "mkl_matrix.h"
#include "test.h"
#include "open_blas.h"
using namespace std;

int main(){
    // int a = 5;
    // int b = 10;
    // int *p;
    // cout << sumOf(a,b) << endl;
    // checkiostream();
    // // Testing some pointers
    // p = &a;
    // b = *p;
    // cout << b << endl;
    // // So basically &a will give the address of an int a
    // // *p will give the value at address p
    // double* A;
    // int m = 3;
    // int n = 3;
    // int k = 3;
    // double alpha = 1;
    // double beta = 0;
    // A = (double *)mkl_malloc( m*k*sizeof( double ), 64 );
    // // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
    // //             m, n, k, alpha, A, k, B, n, beta, C, n);
    // for(int i=0; i < 9; i++)
    // {
    //     A[i] = i;
    // }
    // cout << "---- cblas_dscal ----" << endl;
    // cout << A[0] << " " << A[1] << " " << A[2] << endl;
    // cout << A[3] << " " << A[4] << " " << A[5] << endl;
    // cout << A[6] << " " << A[7] << " " << A[8] << endl;
    // cout << "hlowld";


    // Now will make a 3 * 3 multiplication using mkl
    int m = 3;
    int n = 3;
    int k = 3;
    double* A;
    double* B;
    double* C;
    double* D;
    double alpha = 1;
    double beta = 0;
    A = (double *)mkl_malloc( m*k*sizeof( double ), 64 );
    B = (double *)mkl_malloc( k*n*sizeof( double ), 64 );
    C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
    D = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
    for(int i = 0;i < 9;i ++ )
    {
        A[i] = i;
        B[i] = i + 10;
    }
    for(int i = 0; i < 9;i ++)
    {
        cout << A[i] << " ";
    }
    cout << "\n";
    for(int i =0; i < 9; i++)
    {
        cout << B[i] << " ";
    }
    cout << "\n";
    C = matrixMultiplyMKL(m,n,k,alpha,beta,A,B,C);
    for(int i=0; i < 9; i++)
    {
        cout << C[i] << " ";
    }
    cout << "\n";
    D = matrixMultiplyOpenBlas(m,n,k,alpha,beta,A,B,C);
    for(int i=0; i < 9; i++)
    {
        cout << D[i] << " ";
    }
    cout << "\n";
    return 0;
}