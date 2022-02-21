#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "mkl_matrix.h"
#include "open_blas.h"
using namespace std;

vector<vector<double>> matrixmul(vector<vector<double>> first,vector<vector<double>> second)
{
    vector<vector<double>> result;
    int m = first.size();
    int k = first[0].size();
    int n = second[0].size();
    for(int i = 0; i < m; i++)
    {
        vector<double> temp;
        for(int j =0; j <n; j++)
        {
            double add =0;
            for(int l = 0; l < k; l++)
            {
                add = add + first[i][l] * second[l][j];
            }
            temp.push_back(add);
        }
        result.push_back(temp);
    }  
    return result;
}

void writeVector(vector<float> data,string filename)
{
    int size = data.size();
    ofstream writing(filename);

    writing << size << endl;
    for(int i=0; i < size;i++)
    {
        writing << data[i] << endl;
    }
    writing.close();
}
int main(){
    int m = 500;
    int n = 500;
    int k = 500;
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
    vector<vector<double>> first,second,result;
    
    for(int i = 0; i < 2500; i++)
    {
        A[i] = i + 0.51;
        B[i] = i + 10.3;
    }
    for(int i =0; i <500; i++)
    {
        vector<double> temp1,temp2;
        for (int j = 0; j <500; j++)
        {
            temp1.push_back(A[500 *i + j]);
            temp2.push_back(B[500*i + j]);
        }
        first.push_back(temp1);
        second.push_back(temp2);
    }
    auto start0 = std:: chrono:: high_resolution_clock ::now();
        result  = matrixmul(first,second);
    auto stop0 = std:: chrono:: high_resolution_clock ::now();
    auto duration0 =std:: chrono:: duration_cast<std:: chrono::microseconds>(stop0 - start0);
    cout << "Time taken by my matrix is " << duration0.count() << " microseconds" <<endl;

    auto start = std:: chrono:: high_resolution_clock ::now();
        C = matrixMultiplyMKL(m,n,k,alpha,beta,A,B,C);
    auto stop = std:: chrono:: high_resolution_clock ::now();
    auto duration =std:: chrono:: duration_cast<std:: chrono::microseconds>(stop - start);
    cout << "Time taken by MKL " << duration.count() << " microseconds" <<endl;

    auto start1 = std:: chrono:: high_resolution_clock ::now();
        D = matrixMultiplyMKL(m,n,k,alpha,beta,A,B,C);
    auto stop1 = std:: chrono:: high_resolution_clock ::now();
    auto duration1 =std:: chrono:: duration_cast<std:: chrono::microseconds>(stop1 - start1);
    cout << "Time taken by Open blas " << duration1.count() << " microseconds" << endl;

    return 0;

}