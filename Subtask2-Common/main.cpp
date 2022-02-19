// This is the cpp file containing our main function, others are header files
#include <iostream>
#include "test.h"
using namespace std;

int main(){
    int a = 5;
    int b = 10;
    int *p;
    cout << sumOf(a,b) << endl;
    checkiostream();
    // Testing some pointers
    p = &a;
    b = *p;
    cout << b << endl;
    // So basically &a will give the address of an int a
    // *p will give the value at address p
    return 0;
}