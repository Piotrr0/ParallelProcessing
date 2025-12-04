#include <cstdio>

#define N 10000000
#define ITERS 1000

__global__ void add(int *a, int *b, int *out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;

    out[tid] = a[tid] + b[tid];
}

__global__ void add_stride(int *a, int *b, int *out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

__global__ void divergence_functions(float *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    float x = 0.5f;
    for (int k = 0; k < ITERS; k++) {
        if (tid % 2 == 0) 
            x += acosf(x);
        else
            x += asinf(x);
    }
    data[tid] = x;
}

__global__ void divergence_arithmetic(float *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    float x = 0.5f;
    for (int k = 0; k < ITERS; k++) {
        if (tid % 2 == 0) 
            x += acosf(x);
        else 
            x -= acosf(x);
    }
    data[tid] = x;
}

int main()
{
    size_t bytes = N * sizeof(int);
    int* a = (int*)malloc(bytes);
    int* b = (int*)malloc(bytes);
    int* out = (int*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        a[i] = i; 
        b[i] = i * 2;
    }

    int *c_a, *c_b, *c_out;
    cudaMalloc(&c_a, bytes);
    cudaMalloc(&c_b, bytes);
    cudaMalloc(&c_out, bytes);
    cudaMemcpy(c_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(c_b, b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    
    add<<<blocks, threads>>>(c_a, c_b, c_out, N);
    cudaDeviceSynchronize();

    int blocks_stride = 32;    
    add_stride<<<blocks_stride, threads>>>(c_a, c_b, c_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(out, c_out, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("%d ", out[i]);
    }
    printf("\n");

    cudaFree(c_a); 
    cudaFree(c_b); 
    cudaFree(c_out);
    free(a); 
    free(b); 
    free(out);

    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float)); 

    float time1, time2;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    divergence_functions<<<blocks, threads>>>(d_data);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time1, start, stop);

    cudaEventRecord(start);
    divergence_arithmetic<<<blocks, threads>>>(d_data);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time2, start, stop);

    printf("Function Divergence: %f ms\n", time1);
    printf("Arithmetic Divergence: %f ms\n", time2);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    
    return 0;
}