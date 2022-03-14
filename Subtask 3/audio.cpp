#include<vector>
#include<iostream>
#include<cmath>
#include<fstream>
#include "mkl_matrix.h"
#include "open_blas.h"
using namespace std;

typedef struct pred_t
{
    int label;
    float prob;
}pred_t;


//Function to obtain softmax vector from a vector of floats
vector<float> softmax(vector<float> data)
{
    vector<float> result;
    float sum = 0;
    for(int i=0;i <data.size();i++)
    {
        sum = sum + exp(data[i]);
    }
    for(int i=0;i <data.size();i++)
    {
        float temp = exp(data[i])/sum;
        result.push_back(temp);
    }
    return result;
}

// Function to evaluate relu of matrix/vector elements (which are floats)
float relu_number(float x )
{
    if( x >0)
    {
        return x;
    }
    else
    {
        return 0;
    }
}

// Function to apply the relu function elementwise to a vector
vector<float> relu_vector(vector<float> input)
{
    vector<float> output;
    for(int i=0; i<input.size(); i++)
    {
        float a = relu_number(input[i]);
        output.push_back(a);
    }
    return output;
}

// Function to read vector stored in a file (only floating point entries without any other characters like commas etc) given its size
vector<float> readVector(string filename, int size)
{
    ifstream indata;
    indata.open(filename);
    if(!indata)
    {
        // If file cannot be opened, then throw error
        cerr<<"Error: File could not be opened! (Check if file exists)\n";
        exit(1);
    }
    if(indata.eof())
    {
        // If file is empty then vector cannot be created so throw error
        cerr<<"Error: File is empty!\n";
        exit(1);
    }

    vector<float> newVector(size,0);
    for(int i=0; i<size; i++)
    {
        if(indata.eof())
        {
            // Throw error if the number of entries provided are less than the size of the vector 
            cerr<<"Error: Less entries provided for vector than required!\n";
            exit(1);
        }
        indata>>newVector[i];
    }
    indata.close();
    return(newVector);
}

// Function to simulate a layer of the neural network, takes input a feature matrix of size m*k, weight matrix of size k*n
// and bias matrix of size m*n, multiplies feature and weight matrices and adds bias matrix to the result (all matrices are
// represented by vectors in row major order here)
vector<float> layer(int m, int k, int n, vector<float> feature, vector<float> weight, vector<float> bias)
{
    // Pointer to double to store values in the matrices in row major order (needed to perform multiplication using openblas)
    double* A;
    double* B;
    double* C;
    double* D;
    // Here we allocate the free space to A,B.C,D for storing the matrix elements
    A = (double *)mkl_malloc( m*k*sizeof( double ), 64 );
    B = (double *)mkl_malloc( k*n*sizeof( double ), 64 );
    C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
    D = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
    for(int i=0; i < (m * k); i++)
    {
        A[i] = feature[i];
    }
    for(int i=0; i < (k * n); i++)
    {
        B[i] = weight[i];
    }
    for(int i=0; i < (m * n); i++)
    {
        C[i] = bias[i];
    }
    // multiplication using openBlas
    D = matrixMultiplyOpenBlas(m,n,k,1.0,1.0,A,B,C);
    vector<float> result;
    for(int i=0; i < m*n ;i++)
    {
        result.push_back(D[i]);
    }
    return result;
}

//Function to give a vector of three integers which are the positions of the three highest entries of the input vector (assumed to
// have positive entries only)
vector<int> topThree(vector<float> final_result)
{
    vector<float> duplicate;
    for(int i=0; i<12; i++)
    {
        duplicate.push_back(final_result[i]);
    }
    int max1=0, max2=0, max3=0;
    for(int i=0; i<12; i++)
    {
        if(duplicate[max1]<duplicate[i])
        {
            max1 = i;
        }
    }
    // once first maximum is found, change that entry to negative so the next maximum is now the new maximum
    duplicate[max1] = -1;

    for(int i=0; i<12; i++)
    {
        if(duplicate[max2]<duplicate[i])
        {
            max2 = i;
        }
    }
    // same as for first maximum above
    duplicate[max2] = -1;

    for(int i=0; i<12; i++)
    {
        if(duplicate[max3]<duplicate[i])
        {
            max3 = i;
        }
    }

    vector<int> answer;
    answer.push_back(max1);
    answer.push_back(max2);
    answer.push_back(max3);
    return answer;
}

extern pred_t* libaudioAPI(const char* audiofeatures, pred_t* pred)
{
    pred_t element1, element2, element3;
    vector<float> weight1 = readVector("weight1.txt",36000); // 250*144 = 360 
    vector<float> weight2 = readVector("weight2.txt",20736); // 144*144 = 20736
    vector<float> weight3 = readVector("weight3.txt",20736); // 144*144 = 10736
    vector<float> weight4 = readVector("weight4.txt",1728); // 144*12 = 1728
    vector<float> bias1 = readVector("bias1.txt",144);
    vector<float> bias2 = readVector("bias2.txt",144);
    vector<float> bias3 = readVector("bias3.txt",144);
    vector<float> bias4 = readVector("bias4.txt",12);

    // vector containing the 12 keywords in order
    vector<string> keywords;
    keywords.push_back("silence");
    keywords.push_back("unknown");
    keywords.push_back("yes");
    keywords.push_back("no");
    keywords.push_back("up");
    keywords.push_back("down");
    keywords.push_back("left");
    keywords.push_back("right");
    keywords.push_back("on");
    keywords.push_back("off");
    keywords.push_back("stop");
    keywords.push_back("go");

    try
    {
        // features are present in the file defined by the first commmand line argument
        string feature_filename = audiofeatures;
        vector<float> features = readVector(feature_filename,250);

        // integers m, n, k to be given as arguments to the 'layer' function defined previously, will be updated as required
        int m = 1;
        int k = 250;
        int n = 144;
        
        //result of the first layer is in answer1 (after multiplying, adding and taking relu)
        vector<float> result1 = layer(m,k,n,features,weight1,bias1);
        vector<float> answer1 = relu_vector(result1);

        // new values of m, n, k for the next layer
        m = 1;
        k = 144;
        n = 144;
        // second result is in answer2
        vector<float> result2 = layer(m,k,n,answer1,weight2,bias2);
        vector<float> answer2 = relu_vector(result2);

        // new values of m, n, k for the next layer
        m=1;
        k=144;
        n=144;
        // third result is in answer3
        vector<float> result3 = layer(m,k,n,answer2,weight3,bias3);
        vector<float> answer3 = relu_vector(result3);

        // new values of m,n,k for the last layer
        m=1;
        k=144;
        n=12;
        vector<float> result4 = layer(m,k,n,answer3,weight4,bias4);

        // final result after taking softmax is in result4
        vector<float> final_result = softmax(result4);
        // top contains the three most probable results (indices)
        vector<int> top = topThree(final_result);

        element1.label = top[0];
        element1.prob = final_result[top[0]];
        element2.label = top[1];
        element2.prob = final_result[top[1]];
        element3.label = top[2];
        element3.prob = final_result[top[2]];
        pred[0] = element1;
        pred[1] = element2;
        pred[2] = element3;
        
    }     
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return pred;
}