#include <iostream>
#include "mkl.h"
// #include "cblas.h"

using namespace std;

int main() {

    double A[9]={
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };
    
    cout << A[0] << " " << A[1] << " " << A[2] << endl;
    cout << A[3] << " " << A[4] << " " << A[5] << endl;
    cout << A[6] << " " << A[7] << " " << A[8] << endl;

    cblas_dscal(9, 2.0, A, 1);
    
    cout << "---- cblas_dscal ----" << endl;
    cout << A[0] << " " << A[1] << " " << A[2] << endl;
    cout << A[3] << " " << A[4] << " " << A[5] << endl;
    cout << A[6] << " " << A[7] << " " << A[8] << endl;
    cout << "hlowld";
    return 0;
}