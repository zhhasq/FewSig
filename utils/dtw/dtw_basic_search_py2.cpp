//c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup $(python3 -m pybind11 --includes) dtw_basic_search_py2.cpp -o dtw_basic_search2$(python3-config --extension-suffix)
//linux c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) dtw_basic_search_py2.cpp -o dtw_basic_search2$(python3-config --extension-suffix)

#include <pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))
#define dist(x,y) ((x-y)*(x-y))

#define INF 1e20       //Pseudo Infitinte number for this code

using namespace std;
namespace py = pybind11;

double dtw(double* A, double* B, int m, int r) {

    double *cost;
    double *cost_prev;
    double *cost_tmp;
    int i,j,k;
    double x,y,z,min_cost;

    /// Instead of using matrix of size O(m^2) or O(mr), we will reuse two array of size O(r).
    cost = (double*)malloc(sizeof(double)*(2*r+1));
    for(k=0; k<2*r+1; k++) {
        cost[k]=INF;
    }

    cost_prev = (double*)malloc(sizeof(double)*(2*r+1));
    for(k=0; k<2*r+1; k++) {
        cost_prev[k]=INF;
    }


    for (i=0; i<m; i++) {
        k = max(0,r-i);
        min_cost = INF;

        for(j=max(0,i-r); j<=min(m-1,i+r); j++, k++) {
            /// Initialize all row and column
            if ((i==0)&&(j==0)) {
                cost[k]=dist(A[0],B[0]);
                min_cost = cost[k];
                continue;
            }

            if ((j-1<0)||(k-1<0)) {
                y = INF;
            } else {
                y = cost[k-1];
            }
            if ((i-1<0)||(k+1>2*r)) {
                x = INF;
            } else {
                x = cost_prev[k+1];
            }
            if ((i-1<0)||(j-1<0)) {
                z = INF;
            } else {
                z = cost_prev[k];
            }

            /// Classic DTW calculation
            if (x <= y && x <= z) {
                cost[k] = x+dist(A[i],B[j]);
            } else if (y <= x && y <= z) {
                cost[k] = y+dist(A[i],B[j]);
            } else {
                cost[k] = z+dist(A[i],B[j]);
            }
            // cost[k] = min( min( x, y) , z) + dist(A[i],B[j]);

            /// Find minimum cost in row for early abandoning (possibly to use column instead of row).
            if (cost[k] < min_cost) {
                min_cost = cost[k];
            }
        }

        /// Move current array to previous array.
        cost_tmp = cost;
        cost = cost_prev;
        cost_prev = cost_tmp;
    }
    k--;

    /// the DTW distance is in the last cell in the matrix of size O(m^2) or at the middle of our array.
    double final_dtw = cost_prev[k];

    free(cost);
    free(cost_prev);
    return sqrt(final_dtw);
}

py::array_t<double> dtw_basic_search2(py::array_t<double>& x, py::array_t<double>& y, int w) {
    // x is search space, y is query
    py::buffer_info x_buffer = x.request();
    py::buffer_info y_buffer = y.request();
    double *x_ptr = (double *)x_buffer.ptr;
    double *y_ptr = (double *)y_buffer.ptr;


    int n = x_buffer.shape[0];
    int m = y_buffer.shape[0];
    int i = 0;


    double *q, *search_buffer;
    double ex , ex2 , mean, std;
    q = (double *)malloc(sizeof(double)*m);


    // normalize y
    for(i = 0 ; i < m ; i++ ) {
        q[i] = y_ptr[i];
    }
    ex = ex2 = 0;
    for (i = 0; i < m; i++) {
        ex += q[i];
        ex2 += q[i] * q[i];
    }
    mean = ex/m;
    std = ex2/m;
    std = sqrt(std-mean*mean);
    for( i = 0 ; i < m ; i++ ) {
        q[i] = (q[i] - mean)/std;
    }
    //buffer for x
    search_buffer = (double *)malloc(sizeof(double)*m);
    //z-norm current buffer
    ex = ex2 = 0;
    for (i = 0; i < m - 1; i++) {
        ex += x_ptr[i];
        ex2 += x_ptr[i] * x_ptr[i];
    }

    auto result_complete = py::array_t<double>((n-m+1));
    py::buffer_info result_complete_buff = result_complete.request();
    double *result_complete_ptr = (double *)result_complete_buff.ptr;


    for (i = 0; i < n - m + 1; i++) {
        ex += x_ptr[m - 1 + i];
        ex2 += x_ptr[m - 1 + i] * x_ptr[m - 1 + i];

        mean = ex/m;
        std = ex2/m;
        std = sqrt(std-mean*mean);

        for(int j = 0 ; j < m; j++ ) {
            search_buffer[j] = (x_ptr[i + j] - mean)/std;
        }

        double dist = dtw(search_buffer, q, m, w);
        //result[0] is dtw distance
        result_complete_ptr[i] = dist;
        ex -= x_ptr[i];
        ex2 -= x_ptr[i] * x_ptr[i];

    }
    free(q);
    free(search_buffer);
    return result_complete;
}


PYBIND11_MODULE(dtw_basic_search2, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("dtw_basic_search2", &dtw_basic_search2);
}