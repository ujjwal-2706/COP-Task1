#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "mkl_matrix.h"
#include "open_blas.h"
#include "p_thread.h"
#include <algorithm>
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
    srand(time(0));
    vector<vector<double>> first,second,result;
    
    vector<vector<double>> mklarr;
    vector<vector<double>> openblasarr;
    vector<vector<double>> threadarr;
    for(int i=0; i<300; i++)
    {
        mklarr.push_back({});
        openblasarr.push_back({});
        threadarr.push_back({});
        for(int j=0; j<5; j++)
        {
            mklarr[i].push_back(0);
            openblasarr[i].push_back(0);
            threadarr[i].push_back(0);
        }
    }
    for(int b=0; b<5; b++)
    {
    for(int i=1; i<301; i++)
    {
        
        int m = i;
        int n = i;
        int k = i;
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

        for(int j=0; j< i*i; j++)
        {
            A[j] = rand();
            B[j] = rand();
        }
        for(int j=0; j<i; j++)
        {
            vector<double> temp1, temp2;
            for(int k=0; k<i; k++)
            {
                temp1.push_back(A[i*j+k]);
                temp2.push_back(B[i*j+k]);
            }
            first.push_back(temp1);
            second.push_back(temp2);
        }

        auto start = std:: chrono:: high_resolution_clock ::now();
            C = matrixMultiplyMKL(m,n,k,alpha,beta,A,B,C);
        auto stop = std:: chrono:: high_resolution_clock ::now();
        auto duration =std:: chrono:: duration_cast<std:: chrono::microseconds>(stop - start);
        mklarr[i-1][b] = duration.count();
        // ofstream outdata;
        // outdata.open("mkl.txt",std::ios_base::app);
        // outdata<<duration.count()<<endl;
        // outdata.close();

        auto start1 = std:: chrono:: high_resolution_clock ::now();
            D = matrixMultiplyOpenBlas(m,n,k,alpha,beta,A,B,C);
        auto stop1 = std:: chrono:: high_resolution_clock ::now();
        auto duration1 =std:: chrono:: duration_cast<std:: chrono::microseconds>(stop1 - start1);
        openblasarr[i-1][b] = duration1.count();
        // outdata.open("openblas.txt",std::ios_base::app);
        // outdata<<duration1.count()<<endl;
        // outdata.close();

        for(int l=0; l<i; l++)
        {
            vector<float> temp;
            for(int k=0; k<i; k++)
            {
                temp.push_back(A[l*i+k]);
            }
            pmatrix1.push_back(temp);
        }
        for(int l=0; l<i; l++)
        {
            vector<float> temp;
            for(int k=0; k<i; k++)
            {
                temp.push_back(B[l*i+k]);
            }
            pmatrix2.push_back(temp);
        }
        for(int l=0; l<i; l++)
        {
            vector<float> temp;
            for(int k=0; k<i; k++)
            {
                temp.push_back(0);
            }
            pmat_result.push_back(temp);
        }
        
        auto start2 = std:: chrono:: high_resolution_clock ::now();
            pthreadMatrix();
        auto stop2 = std:: chrono:: high_resolution_clock ::now();
        auto duration2 =std:: chrono:: duration_cast<std:: chrono::microseconds>(stop2 - start2);
        threadarr[i-1][b] = duration2.count();
        // outdata.open("pthread.txt",std::ios_base::app);
        // outdata<<duration2.count()<<endl;
        // outdata.close();

        for(int j=0; j<i; j++)
        {
            pmat_result.pop_back();
            pmatrix1.pop_back();
            pmatrix2.pop_back();
        }
    }
    }
    for(int i=0; i<300; i++)
    {
        sort(mklarr[i].begin(), mklarr[i].end());
        sort(openblasarr[i].begin(), openblasarr[i].end());
        sort(threadarr[i].begin(), threadarr[i].end());
        ofstream outdata;
        outdata.open("mklnew.dat",std::ios_base::app);
        outdata<<i+1<<" ";
        for(int j=0; j<5; j++)
        {
            outdata<<mklarr[i][j]<<" ";
        }
        outdata<<endl;
        outdata.close();
        outdata.open("openblasnew.dat", std::ios_base::app);
        outdata<<i+1<<" ";
        for(int j=0; j<5; j++)
        {
            outdata<<openblasarr[i][j]<<" ";
        }
        outdata<<endl;
        outdata.close();
        outdata.open("threadnew.dat", std::ios_base::app);
        outdata<<i+1<<" ";
        for(int j=0; j<5; j++)
        {
            outdata<<threadarr[i][j]<<" ";
        }
        outdata<<endl;
        outdata.close();
    }

    return 0;

}