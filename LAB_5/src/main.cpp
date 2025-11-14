#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <omp.h>

#define THREADS_COUNT 8

int split(int* S, int left, int right) {
    int pivot = S[right];
    int i = left-1;
    for (int j = left; j < right; j++) {
        if (S[j] < pivot) {
            i++;
            std::swap(S[i], S[j]);
        }
    }
    std::swap(S[i+1], S[right]);
    return i + 1;
}

void quickSort(int* S, int left, int right) {
    if (left >= right)
        return;

    int pivotindex = split(S, left, right);

    #pragma omp task shared(S)
    quickSort(S, left, pivotindex - 1);

    #pragma omp task shared(S)
    quickSort(S, pivotindex + 1, right);

    #pragma omp taskwait
}

void print(int* S, int n){
    for (int i = 0; i < n; i++) {
        std::cout << S[i] << " ";
    }
    std::cout << std::endl;
}

int main() {

    omp_set_num_threads(THREADS_COUNT);

    int n = 10;
    int* S = new int[n];

    srand(time(0));
    for (int i = 0; i < n; i++) {
        S[i] = rand() % 100;
    }
    print(S, n);

    #pragma omp parallel
    {
        #pragma omp single
        quickSort(S, 0, n - 1);
    }
    print(S, n); 
    
    delete[] S;
    return 0;
}