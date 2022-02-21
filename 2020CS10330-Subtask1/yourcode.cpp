#include <iostream>  
#include <fstream> 
#include <vector> 
#include <cmath> 
#include <string.h> 
using namespace std;

//Defining a Matrix class that will be used to carry out the many functions defined below related to matrices 
class Matrix
{
    public:
    int rowsize; 
    int columnsize; 
    vector<vector<float>> data; // Attribute that is a vector of float vectors, to store the matrix entries
    Matrix(int rowsize, int columnsize); // constructor function for the Matrix class
};

// Constructor for Matrix class, that creates a matrix of dimensions rows x columns initializes with all zeroes.
Matrix::Matrix(int rows, int columns)
{
    rowsize = rows;
    columnsize = columns;
    for(int i=0; i<rows; i++)
    {
        data.push_back({});
        for(int j=0; j<columns; j++)
        {
            data[i].push_back(0);
        }
    }
}

// Function to write the contents of a Matrix object in a file, in the format specified in the assignment
void writeMatrix(string filename, Matrix a)
{
    ofstream outdata;
    outdata.open(filename);
    outdata<<a.columnsize<<endl<<a.rowsize<<endl; // The first two lines contain the number of columns and rows, in that order
    // Printing elements of the matrix in column major format
    for(int i=0; i<a.columnsize; i++)
    {
        for(int j=0; j<a.rowsize; j++)
        {
            outdata<<a.data[j][i]<<endl;
        }
    }
    outdata.close();
    return;
}

// Function to read a matrix specified in a text file
Matrix readMatrix(string filename)
{
    ifstream indata;
    int columnsize, rowsize;
    indata.open(filename);
    if(!indata)
    {
        // If file does not exist, throw error
        cerr<<"Error: File could not be opened! (Check if file exists)\n";
        exit(1);
    }
    if(indata.eof())
    {
        // If file exists but is empty, throw error
        cerr<<"Error: File is empty!\n";
        exit(1);
    }

    // Read the columnsize first, and then the rowsize
    indata>>columnsize>>rowsize;
    if(columnsize<=0 || rowsize<=0)
    {
        // If a non-positive integer is specified as the row or column dimension
        cerr<<"Error: Dimensions of the matrix must be positive integers only!\n";
        exit(1);
    }
    // Instantiating a new object of the Matrix class
    Matrix newMatrix(rowsize, columnsize);
    for(int i=0; i<columnsize; i++)
    {
        for(int j=0; j<rowsize; j++)
        {
            if(indata.eof())
            {
                // If the entries provided are less than needed for construction of the matrix with specified dimensions, throw error
                cerr<<"Error: Less entries provided for matrix than required!\n";
                exit(1);
            }
            indata>>newMatrix.data[j][i];
        }
    }
    indata.close();   
    return(newMatrix);
}

// Function to add two objects of the matrix class, and return a new object of the same class, without altering any of the 
// inputs
Matrix AddMat(Matrix a, Matrix b)
{
    // If the dimensions of the two matrices do not match, then throw error
    if(a.rowsize != b.rowsize || a.columnsize != b.columnsize)
    {
        cerr<<"Error: Matrices of different dimensions cannot be added!\n";
        exit(1);
    }
    // Instantiating a new matrix object
    Matrix result = Matrix(a.rowsize, a.columnsize);
    for(int i=0; i<a.rowsize; i++)
    {
        // Term by term addition of the two matrices
        for(int j=0; j<a.columnsize; j++)
        {
            result.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }
    return(result);
}

// Function to multiply two objects of the matrix class, and return a new object of the same class, without altering any of the
// inputs
Matrix MultMat(Matrix a, Matrix b)
{
    // If the dimensions of the two matrices are incompatible for multiplication, throw error
    if(a.columnsize != b.rowsize)
    {
        cerr<<"Error: Matrices cannot be multiplied! (Dimensions incompatible)\n";
        exit(1);
    }

    // The standard high-school algorithm to multiply two matrices, in O(nlm) time if the dimensions of the two matrices are n x l
    // and l x m
    Matrix result = Matrix(a.rowsize, b.columnsize);
    for(int i=0; i<a.rowsize; i++)
    {
        for(int j=0; j<b.columnsize; j++)
        {
            float element_ij = 0;
            for(int k=0; k<a.columnsize; k++)
            {
                element_ij += a.data[i][k]*b.data[k][j];
            }
            result.data[i][j] = element_ij;
        }
    }
    return(result);
}

// Function to apply the relu function on a floating point number
float ReLU(float a)
{
    if(a<0)
    {
        return 0;
    }
    else
    {
        return a;
    }
}

// Function that computes the hyperbolic tangent of a floating point number, implemented using the exp() function in cmath
float tanhyperbolic(float a)
{
    float x = exp(2*a) -1;
    float y = exp(2*a) +1;
    return(x/y);
}

// Function to activate a matrix using relu, by simply applying the relu function to all the floating point entries one by one
// Note that the name of the function is overloaded, and is the same as that used for floating point numbers
Matrix ReLU(Matrix a)
{
    Matrix result = Matrix(a.rowsize, a.columnsize);
    for(int i=0; i<a.rowsize; i++)
    {
        for(int j=0; j<a.columnsize; j++)
        {
            result.data[i][j] = ReLU(a.data[i][j]);
        }
    }
    return(result); // A new matrix is returned, and the original matrix is unchanged
}

// Function to activate a matrix using tanh, by simply applying the tanh function to all the floating point entries one by one
Matrix tanh(Matrix a)
{
    Matrix result = Matrix(a.rowsize, a.columnsize);
    for(int i=0; i<a.rowsize; i++)
    {
        for(int j=0; j<a.columnsize; j++)
        {
            result.data[i][j] = tanhyperbolic(a.data[i][j]);
        }
    }
    return(result); // A new matrix is returned, and the original matrix is unchanged
}

// Function to apply max pooling to a matrix, provided the matrix and stride length as the inputs
Matrix maxPool(Matrix a, int stride)
{
    // If the input matrix is not square, throw error
    if(a.rowsize != a.columnsize)
    {
        cerr<<"Error: Pooling requires square matrix only!\n";
        exit(1);
    }
    // If the dimension of the matrix is not divisible by the stride length, throw error
    if(a.rowsize % stride !=0)
    {
        cerr<<"Error: Stride length must divide the dimension of the square matrix!\n";
        exit(1);
    }

    // Dimension of the new output matrix after pooling
    int size = a.rowsize/stride;
    Matrix result = Matrix(size, size);
    for(int i=0; i<size; i++) // 'i' denotes the row number of the output matrix that will be used to access the i'th row
    {
        for(int j=0; j<size; j++) // 'j', similar to 'i', denotes the column number of the output matrix
        {
            float max = a.data[stride*i][stride*j]; // Among all the elements enclosed by the current stride matrix, one of them is taken as the temporary maximum for comparison purposes
            // The following two loops are used to calculate the maximum by going over all the elemnts in the current stride
            for(int k=0; k<stride; k++)
            {
                for(int l=0; l<stride; l++)
                {
                    if(a.data[stride*i+k][stride*j+l]>max)
                    {
                        max = a.data[stride*i+k][stride*j+l];
                    }
                }
            }
            result.data[i][j] = max;
        }
    }
    return(result);
}

// Function to apply average pooling to the input matrix, given stride length, exactly like the maxPool function just above
Matrix avgPool(Matrix a, int stride)
{
    if(a.rowsize != a.columnsize)
    {
        cerr<<"Error: Pooling requires square matrix only!\n";
        exit(1);
    }
    if(a.rowsize % stride !=0)
    {
        cerr<<"Error: Stride length must divide the dimension of the square matrix!\n";
        exit(1);
    }
    int size = a.rowsize/stride;
    Matrix result = Matrix(size, size);
    for(int i=0; i<size; i++)
    {
        for(int j=0; j<size; j++)
        {
            float sum = 0; // only difference is that instead of temporary maximum, there is a temporary sum that will be used to compute an average
            for(int k=0; k<stride; k++)
            {
                for(int l=0; l<stride; l++)
                {
                    sum += a.data[stride*i+k][stride*j+l];
                }
            }
            result.data[i][j] = sum/(stride*stride);
        }
    }
    return(result);
}

// Function to read a vector from input files, returning a new vector
vector<float> readVector(string filename)
{
    ifstream indata;
    int size;
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
    indata>>size;

    // Throw error if size of vector is not a positive integer
    if(size<=0)
    {
        cerr<<"Error: Size of vector must be a positive integer only!\n";
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

// Function for writing a vector to a file, in the format specified in the assignment
void writeVector(string filename, vector<float> v)
{
    ofstream outdata;
    outdata.open(filename);
    outdata<<v.size()<<endl; // printing the size of the vector
    // printing the vector's data
    for(int i=0; i<v.size(); i++)
    {
        outdata<<v[i]<<endl;
    }
    outdata.close();
    return;
}

// Function to apply the softmax transformation to a vector, and give another vector as an output
vector<float> softmax(vector<float> a)
{
    // The sum is used to hold the sum of all the entries of the vector, which will be used to divide all the elements of the original vector
    float sum = 0;
    for(int i=0; i<a.size(); i++)
    {
        a[i] = exp(a[i]);
        sum += a[i];
    }
    // Dividing all elements of the original vector by the sum
    vector<float> result(a.size(), 0);
    for(int i=0; i<a.size(); i++)
    {
        result[i] = a[i]/sum;
    }
    return(result);
}

// Function that applies the sigmoid transformation to a vector, and gives a new vector as output
vector<float> sigmoid(vector<float> a)
{
    vector<float> result(a.size(), 0);
    for(int i=0; i<a.size(); i++)
    {
        // Definition of the sigmoid
        result[i] = 1/(1+exp(-1*a[i]));
    }
    return(result);
}

// Function to implement a fully connected layer by taking inputs of the three files that contain the input, weight and bias matrices, and then writing it to the output file which is also specified in the arguments
void fullyConnected(string input, string weight, string bias, string output)
{
    Matrix inpMat = readMatrix(input); // the input matrix
    Matrix weightMat = readMatrix(weight); // the weight matrix
    Matrix biasMat = readMatrix(bias); // the bias matrix
    Matrix mult = MultMat(inpMat, weightMat);  // multiplication result
    Matrix result = AddMat(mult, biasMat); // adding the bias matrix to the multiplication resul
    writeMatrix(output, result); // writing result to output
    return;
}

// Function to write the output of a relu activation of a matrix present in input file to the output file (both files are passed as arguments)
void relu(string input, string output)
{
    Matrix inpMat = readMatrix(input);
    Matrix result = ReLU(inpMat);
    writeMatrix(output, result);
    return;
}

// Function to write the output of a tanh activation of a matrix present in input file to the output file, similar to the relu function above
void tanh(string input, string output)
{
    Matrix inpMat = readMatrix(input);
    Matrix result = tanh(inpMat);
    writeMatrix(output, result);
    return;
}

// Function to apply max pooling to a matrix present in input file and write output to a file, both specified in arguments, along with the stride length
void maxPool(string input, int stride, string output)
{
    Matrix inpMat = readMatrix(input);
    Matrix result = maxPool(inpMat, stride);
    writeMatrix(output, result);
    return;
}


// Function to apply average pooling to a matrix present in input file, similar to the one on max pooling above
void avgPool(string input, int stride, string output)
{
    Matrix inpMat = readMatrix(input);
    Matrix result = avgPool(inpMat, stride);
    writeMatrix(output, result);
    return;
}

// Function to apply the softmax transformation to a vector in input file and write result to the output (both in arguments)
void softmax(string input, string output)
{
    vector<float> inpv = readVector(input);
    vector<float> result = softmax(inpv);
    writeVector(output, result);
    return;
}

// Functtion to apply the sigmoid transformation to a vector in input file and write result to the output (both in arguments)
void sigmoid(string input, string output)
{
    vector<float> inpv = readVector(input);
    vector<float> result = sigmoid(inpv);
    writeVector(output, result);
    return;
}

int main(int argc, char** argv)
{
    if(argc<=1)
    {
        // If no arguments provided through the command line, throw error
        cerr<<"Error: Please enter correct number of arguments!\n";
        exit(1);
    }

    // bunch of strings as character arrays for comparison purposes using strcmp below
    char fullyconnected[] = "fullyconnected";
    char activation[] = "activation";
    char pooling[] = "pooling";
    char prob[] = "probability";

    // If first argument is fullyconnected
    if(strcmp(argv[1], fullyconnected)==0)
    {
        if(argc<6)
        {
            // If proper number of arguments aren't provided, throw error
            cerr<<"Error: Please enter more arguments\n";
            exit(1);
        }

        fullyConnected(argv[2], argv[3], argv[4], argv[5]);
    }
    
    // If first argument in activation
    else if(strcmp(argv[1], activation)==0)
    {
        if(argc<5)
        {
            // If proper number of arguments aren't provided, throw error
            cerr<<"Error: Please enter more arguments\n";
            exit(1);
        }

        // some more charcter arrays for comparison purposes used below
        char relustring[] = "relu";
        char tanhstring[] = "tanh";

        // this part should be pretty self explanatory, just applying tanh or relu according to the input encountered
        if(strcmp(argv[2], relustring)==0)
        {
            relu(argv[3], argv[4]);
        }
        
        else if(strcmp(argv[2], tanhstring)==0)
        {
            tanh(argv[3], argv[4]);
        }

        else
        {
            // If something other than relu or tanh is input, then throw error
            cerr<<"Error: The only activation types valid are 'relu' and 'tanh'\n";
            exit(1);
        }
    }

    // If first argument is pooling
    else if(strcmp(argv[1], pooling)==0)
    {   
        if(argc<6)
        {
            // Throw error if the appropriate number of arguments aren't passed
            cerr<<"Error: Please enter more arguments\n";
            exit(1);
        }

        // again, character arrays for comparison purposes using strcmp below
        char max[] = "max";
        char avg[] = "average";

        // max pooling
        if(strcmp(argv[2], max)==0)
        {
            // try catch block used for detecting if something other than a number is passed in the stride argument
            try
            {
                int stride = stoi(argv[4]);
                maxPool(argv[3], stride, argv[5]);
            }
            catch(...)
            {
                cerr<<"Error: The stride should be a number\n";
                exit(1);
            }
        }
        
        // average pooling
        else if(strcmp(argv[2], avg)==0)
        {
            // try catch block used for detecting if something other than a number is passed in the stride argument
            try
            {
                int stride = stoi(argv[4]);
                avgPool(argv[3], stride, argv[5]);
            }
            catch(...)
            {
                cerr<<"Error: The stride should be a number\n";
                exit(1);
            }
        }

        else
        {
            // If something other than max or average is passed, throw error
            cerr<<"Error: Only max pooling and average pooling are defined\n";
            exit(1);
        }
    }

    // if first argument is probability
    else if(strcmp(argv[1], prob)==0)
    {
        if(argc<5)
        {
            // if number of arguments is less than needed, throw error
            cerr<<"Error: Please enter more arguments\n";
            exit(1);
        }

        // character arrays for comparison purposes using strcmp below
        char soft[] = "softmax";
        char sigm[] = "sigmoid";

        // softmax
        if(strcmp(argv[2], soft)==0)
        {
            softmax(argv[3], argv[4]);
        }
        
        // sigmoid
        else if(strcmp(argv[2], sigm)==0)
        {
            softmax(argv[3], argv[4]);
        }

        else
        {
            // If something other than softmax or sigmoid is passed
            cerr<<"Error: The only transformations are softmax and sigmoid\n";
            exit(1);
        }
    }

    else
    {   
        // If the first argument is none out of the four specified in the assignment throw error
        cerr<<"Error: The first argument must be either 'fullyconnected', 'activation', 'pooling' or 'probability'\n";
        exit(1);
    }
    return 0;
}