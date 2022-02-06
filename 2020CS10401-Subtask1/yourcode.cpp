#include<vector>
#include<iostream>
#include<cmath>
#include<fstream>
using namespace std;

//This function will read vector.txt file and give vector
vector<float> readVector(string filename)
{
    ifstream reading(filename);
    if(!reading)
    {
       throw invalid_argument("File Does not exist");
    }
    string traverse;
    getline(reading,traverse);
    if(reading.peek() == std::ifstream::traits_type::eof())// To check if file empty.
    {
       throw invalid_argument("Empty File Given");
    }
    int length = stoi(traverse);
    vector<float> result;
    while(getline(reading,traverse))
    {
        result.push_back(stof(traverse));
    }
    if(result.size()== length)
    {
        return result;
    }
    else
    {
        throw invalid_argument("Wrong input given");
    }
}

//This function will be used to write the output vector file
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

//This function will be used to obtain softmax vector
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

// This function will be used to the sigmoid conversion
vector<float> sigmoid(vector<float> data)
{
    vector<float> result;
    for(int i=0;i < data.size();i++)
    {
        float temp = 1 / (1 + exp(-1 * data[i]));
        result.push_back(temp);
    }
    return result;
}

// This function will be used to evaluate relu of matrix elements
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

//This function will be used to evaluate the tanh of individual element of matrix
float tanh_number(float x)
{
    float temp = exp(2*x);
    float result = (temp -1)/(temp+1);
    return result;
}
class Matrix //Initialize the matrix by passing row and col as parameter and not empty
{
public:
    vector<vector<float>> matrix;
    // This is our constructor to make a matrix the elements value as zero
    Matrix(int row,int col){
        if(row > 0 && col >0)
        {
            for(int i =0; i < row;i++)
            {
                vector<float> empty_vector;
                matrix.push_back(empty_vector);
                for(int j = 0;j < col;j++)
                {
                    matrix[i].push_back(0.0);
                }
            }
        }
        else
        {
            throw invalid_argument("given matrix size is not possible");
        }
    }
    // This method will print the matrix in a grid format
    void print_matrix(){
        int rows = matrix.size();
        int columns = matrix[0].size();
        for(int i= 0;i < rows;i++)
        {
            for(int j=0; j < columns;j++)
            {
                cout << matrix[i][j]<< " ";
            }
            cout << "\n";
        }
    }
    // This will initialize our matrix after reading the file in column wise fashion
    void initialize(vector<float> data)
    {
        int row = matrix.size();
        int col = matrix[0].size();
        int pointer = 0; // This acts as the index of data
        if(row * col == data.size())
        {
            for(int j=0; j <col;j++)
            {
                for(int i=0;i <row;i++)
                {
                    matrix[i][j] = data[pointer];
                    pointer++;
                }
            }
        }
        else
        {
            throw invalid_argument("Data not in proper format or insufficient");
        }
    }
    // This method will be used for matrix multiplication
    Matrix multiply(Matrix matrix2){
        int row1 = matrix.size();
        int col1 = matrix[0].size();
        int row2 = matrix2.matrix.size();
        int col2 = matrix2.matrix[0].size();
        if(col1 == row2)
        {
            Matrix result(row1,col2);
            for(int i=0; i < row1;i++)
            {
                for(int j=0; j < col2;j++)
                {
                    for(int k=0; k < row2;k++)
                    {
                        result.matrix[i][j] = result.matrix[i][j] + matrix[i][k] * matrix2.matrix[k][j];
                    }
                }
            }
            return result;
        }
        else
        {
            throw invalid_argument("The dimensions of matrices aren't compatible");
        }
    }
    // This method will be used for the matrix addition
    Matrix addition(Matrix bias_matrix)
    {
        int row1 = matrix.size();
        int col1 = matrix[0].size();
        int row2 = bias_matrix.matrix.size();
        int col2 = bias_matrix.matrix[0].size();
        if( row1 == row2 && col1 == col2)
        {
            Matrix result(row1,col1);
            for(int i =0 ; i< row1;i ++)
            {
                for(int j=0;j < col1;j++)
                {
                    result.matrix[i][j] = matrix[i][j] + bias_matrix.matrix[i][j];
                }
            }
            return result;
        }
        else
        {
            throw invalid_argument("Wrong dimensions added");
        }
    }

    //This function will give the matrix output after applying the relu function
    Matrix relu()
    {
        int row = matrix.size();
        int col = matrix[0].size();
        Matrix result(row,col);
        for(int i=0;i < row;i++)
        {
            for(int j=0; j < col;j++)
            {
                result.matrix[i][j] = relu_number(matrix[i][j]);
            }
        }
        return result;
    }

    //This method will give the matrix output after applying the tanh function
    Matrix tanh()
    {
        int row = matrix.size();
        int col = matrix[0].size();
        Matrix result(row,col);
        for(int i=0;i < row;i++)
        {
            for(int j=0; j < col;j++)
            {
                result.matrix[i][j] = tanh_number(matrix[i][j]);
            }
        }
        return result;
    }

    //This method will return a max_pooled matrix after taking the given stride
    Matrix max_pooling(int stride)
    {
        int row = matrix.size();
        int col = matrix[0].size();
        if(row != col)
        {
            throw invalid_argument("Not a square matrix");
        }
        if(row % stride != 0)
        {
            throw invalid_argument("Not possible to pool the matrix with the given stride");
        }
        int dim = row /stride;
        Matrix result(dim,dim);
        for(int i=0; i < row; i = i + stride)
        {
            for(int j=0; j < col;j = j + stride)
            {
                float value = matrix[i][j];
                int xtraverse = i;
                while(xtraverse < i + stride)
                {
                    int ytraverse = j;
                    while(ytraverse < j + stride)
                    {
                        value = max(value,matrix[xtraverse][ytraverse]);
                        ytraverse++;
                    }
                    xtraverse++;
                }
                result.matrix[i/stride][j/stride] = value;
            }
        }
        return result;
    }

    Matrix avg_pooling(int stride)
    {
        int row = matrix.size();
        int col = matrix[0].size();
        if(row != col)
        {
            throw invalid_argument("Not a square matrix");
        }
        if(row % stride != 0)
        {
            throw invalid_argument("Not possible to pool the matrix with the given stride");
        }
        int dim = row /stride;
        Matrix result(dim,dim);
        for(int i=0; i < row; i = i + stride)
        {
            for(int j=0; j < col;j = j + stride)
            {
                float value = 0;
                int xtraverse = i;
                while(xtraverse < i + stride)
                {
                    int ytraverse = j;
                    while(ytraverse < j + stride)
                    {
                        value = value + matrix[xtraverse][ytraverse];
                        ytraverse++;
                    }
                    xtraverse++;
                }
                result.matrix[i/stride][j/stride] = value /(stride * stride);
            }
        }
        return result;
    }


};


// This function will read the file to give a matrix
Matrix readMatrix(string filename)
{
    ifstream reading(filename);
    string traverse;
    if(!reading)
    {
        throw invalid_argument("File Does not exist");
    }
    getline(reading,traverse);
    if(reading.peek() == std::ifstream::traits_type::eof())// To check if file empty.
    {
       throw invalid_argument("Empty File Given");
    }
    int col = stoi(traverse);
    getline(reading,traverse);
    int row = stoi(traverse);
    vector<float> column_order;
    while(getline(reading,traverse))
    {
        column_order.push_back(stof(traverse));
    }
    Matrix result(row,col);
    result.initialize(column_order);
    reading.close();
    return result;
}

// This function will be used to write the output matrix
void writeMatrix(Matrix mat,string filename)
{
    ofstream writing(filename);
    int row = mat.matrix.size();
    int col = mat.matrix[0].size();
    writing << col << endl;
    writing << row << endl;
    for(int j= 0; j < col;j ++)
    {
        for(int i=0; i < row;i++)
        {
            writing << mat.matrix[i][j] << endl;
        }
    }
    writing.close();
}


int main(int argc,char** argv){
    try
    {
        string function_name = argv[1];
        if(function_name.compare("fullyconnected") == 0)
        {
            string inputmatrix = argv[2];
            string weightmatrix = argv[3];
            string biasmatrix = argv[4];
            string outputmatrix = argv[5];
            Matrix matrix1 = readMatrix(inputmatrix);
            Matrix matrix2 = readMatrix(weightmatrix);
            Matrix matrix_bias = readMatrix(biasmatrix);
            Matrix multi = matrix1.multiply(matrix2);
            Matrix result = multi.addition(matrix_bias);
            writeMatrix(result, outputmatrix);
        }
        
        else if(function_name.compare("activation") == 0)
        {
            string active_mode = argv[2];
            if(active_mode.compare("relu")==0)
            {
                string inputmatrix = argv[3];
                Matrix input = readMatrix(inputmatrix);
                Matrix result = input.relu();
                string outputmatrix = argv[4];
                writeMatrix(result,outputmatrix);
            }
            else if(active_mode.compare("tanh") == 0)
            {
                string inputmatrix = argv[3];
                Matrix input = readMatrix(inputmatrix);
                Matrix result = input.tanh();
                string outputmatrix = argv[4];
                writeMatrix(result,outputmatrix);
            }
            else 
            {
                throw invalid_argument("Wrong activation entered");
            }
        }

        else if(function_name.compare("pooling") == 0)
        {
            string pool_mode = argv[2];
            if(pool_mode.compare("max")==0)
            {
                string inputmatrix = argv[3];
                try
                {
                    int stride = stoi(argv[4]);
                     string outputmatrix = argv[5];
                    Matrix input = readMatrix(inputmatrix);
                    Matrix result = input.max_pooling(stride);
                    writeMatrix(result,outputmatrix);
                }
                catch(exception &err)
                {
                    throw invalid_argument("Something's wrong the given arguments");
                }
               
            }
            else if(pool_mode.compare("average") == 0)
            {
                string inputmatrix = argv[3];
                try
                {
                    int stride = stoi(argv[4]);
                    string outputmatrix = argv[5];
                    Matrix input = readMatrix(inputmatrix);
                    Matrix result = input.avg_pooling(stride);
                    writeMatrix(result,outputmatrix);
                }
                catch(exception &err)
                {
                    throw invalid_argument("Something's wrong the given arguments");
                }
            }
            else
            {
                throw invalid_argument("Wrong pool mode entered");
            }
        }

        else if(function_name.compare("probability")==0)
        {
            string prob_mode = argv[2];
            if(prob_mode.compare("softmax") ==0)
            {
                string inputvector = argv[3];
                string outputvector = argv[4];
                vector<float> result = readVector(inputvector);
                result = softmax(result);
                writeVector(result,outputvector);
            }
            else if(prob_mode.compare("sigmoid") == 0)
            {
                string inputvector = argv[3];
                string outputvector = argv[4];
                vector<float> result = readVector(inputvector);
                result = sigmoid(result);
                writeVector(result,outputvector);
            }
            else 
            {
                throw invalid_argument("Wrong probability mode entered");
            }
        }

        else
        {
            throw invalid_argument("Incorrect arguments passed");
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return 0;
}