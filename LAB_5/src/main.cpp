#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <unistd.h>

#define THREADS_COUNT 8
#define INSERTION_SORT_THRESHOLD 50
#define TASK_THRESHOLD 1000

int taskCount = 0;
int finalTaskCount = 0;
#pragma omp threadprivate(taskCount, finalTaskCount)

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

void insertionSort(int* S, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int aux = S[i];
        int j = i - 1;
        while (j >= left && S[j] > aux) {
            S[j + 1] = S[j];
            j--;
        }
        S[j + 1] = aux;
    }
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

void quickSortFinal(int* S, int left, int right) {
    if (left >= right)
        return;

    int size = right - left + 1;

    if (size <= INSERTION_SORT_THRESHOLD) {
        insertionSort(S, left, right);
        return;
    }

    int pivotindex = split(S, left, right);
    bool use_final = size <= TASK_THRESHOLD;

    if (use_final)
        finalTaskCount += 2;
    else
        taskCount += 2;

    #pragma omp task shared(S) firstprivate(left, pivotindex) final(use_final)
    quickSortFinal(S, left, pivotindex - 1);

    #pragma omp task shared(S) firstprivate(pivotindex, right) final(use_final)
    quickSortFinal(S, pivotindex + 1, right);

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

    int n = 100000;
    int* S_1 = new int[n];
    int* S_2 = new int[n];


    srand(time(0));
    for (int i = 0; i < n; i++) {
        int num = rand() % 100000;
        S_1[i] = num;
        S_2[i] = num;
    }

    double start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        quickSort(S_1, 0, n - 1);
    }

    double end_time = omp_get_wtime();
    std::cout << end_time - start_time << std::endl;


    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        quickSortFinal(S_2, 0, n - 1);
        
        #pragma omp barrier
        int thread_id = omp_get_thread_num();
        usleep(thread_id * 100000);
        
        std::cout << "taskCount: " << taskCount << " ID: " << thread_id << std::endl;
        std::cout << "finalTaskCount: " << finalTaskCount << " ID: " << thread_id << std::endl;
    }
    end_time = omp_get_wtime();
    
    std::cout << end_time - start_time << std::endl;

    delete[] S_1;
    delete[] S_2;

    return 0;
}