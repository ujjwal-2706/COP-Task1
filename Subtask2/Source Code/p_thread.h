#include <iostream>
#include <vector>
using namespace std;

// These vectors will be used globally in order to compute the 
// matrix multiplication using threads 
// The number of threads used will depend on the rows in resultant matrices 
vector<vector<float>> pmatrix1;
vector<vector<float>> pmatrix2;
vector<vector<float>> pmat_result;

// This will indicate the row which is being evaluated by thread
int step_i = 0;
 
// This will perform evaluation of a particular row using a thread
void* singleThreadMultipy(void* arg)
{
    int i = step_i++; //i denotes row number of resultant pmat_result
   if(pmatrix1[0].size() != pmatrix2.size())
   {
       throw invalid_argument("Wrong Dimensions of Matrix");
   }
   if(i < pmatrix1.size()){
    for (int j = 0; j < pmatrix2[0].size(); j++)
      for (int k = 0; k < pmatrix1[0].size(); k++)
        pmat_result[i][j] += pmatrix1[i][k] * pmatrix2[k][j];
   }
   
   return 0;
}

// This will perform the overall multiplication of matrix
void pthreadMatrix()
{
    int MAX_THREAD = pmatrix1.size();
     // declaring four threads
    pthread_t threads[MAX_THREAD];
 
    // Creating four threads, each evaluating its own part
    for (int i = 0; i < MAX_THREAD; i++) {
        int* p;
        pthread_create(&threads[i], NULL, singleThreadMultipy, (void*)(p));
    }
 
    // joining and waiting for all threads to complete
    for (int i = 0; i < MAX_THREAD; i++)
        pthread_join(threads[i], NULL);   
 
}
 