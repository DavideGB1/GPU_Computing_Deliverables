#include "../include/my_time_lib.h"

#include <math.h>

// Put here the implementation of arithmetic_mean and geometric_mean

double arithmetic_mean(double array[],int len){
    double res = 0.0;
    for(int i=0;i<len;i++){
        res += array[i];
    }
    return res/len;
}

double geometric_mean(double array[],int len){
    double res = 1.0;
    for(int i=0;i<len;i++){
        res *= array[i] > 0.0 ? array[i] : 1;
    }
    return pow(res,1.0/len);
}
// -------------------------------------------------
