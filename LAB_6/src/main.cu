#include <cstdio>

#define N 1024


class Timer {
public:
    Timer() {
        cudaEventCreate(&startGPU);
        cudaEventCreate(&stopGPU);

        cudaEventRecord(startGPU);
    }
    ~Timer() {
        cudaEventRecord(stopGPU);

        cudaDeviceSynchronize();
        cudaEventElapsedTime(&timeGPU, startGPU, stopGPU);
        printf("Elapsed time: %f ms\n", timeGPU);

        cudaEventDestroy(startGPU);
        cudaEventDestroy(stopGPU);
    }
private:
        float timeGPU;
        cudaEvent_t startGPU, stopGPU;
};

void printDeviceProperties(const cudaDeviceProp& devProp)
{
    printf("Device Name: %s\n", devProp.name);
    printf("Total Global Memeory: %lu bytes\n", devProp.totalGlobalMem);

    printf("Multiprocessor (SM) Count: %d\n", devProp.multiProcessorCount);

    printf("Max Threads per Block: %d\n", devProp.maxThreadsPerBlock);
    printf("Shared Memory per Block: %lu bytes\n", devProp.sharedMemPerBlock);
    printf("Registers per Block: %d\n", devProp.regsPerBlock);

    printf("Warp Size: %d\n", devProp.warpSize);

    printf("Active threads per SM: %d\n", devProp.maxThreadsPerMultiProcessor);
    printf("Max active blocks per SM: %d\n", devProp.maxBlocksPerMultiProcessor);
}

__global__ void square(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        data[tid] = data[tid] * data[tid];
    }
}

__global__ void kernel(float*data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        data[tid] = blockIdx.x;
    }
}
__global__ void kernel_intrinsic(float* data, int n, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;

    float x = data[tid];
    for (int k = 0; k < iters; k++) {
        x = __sinf(x);
    }
    data[tid] = x;
}

__global__ void kernel_single(float* data, int n, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;

    float x = data[tid];
    for (int k = 0; k < iters; k++) {
        x = sinf(x);
    }
    data[tid] = x;
}

__global__ void kernel_double(double* data, int n, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;

    double x = data[tid];
    for (int k = 0; k < iters; k++) {
        x = sin(x);
    }
    data[tid] = x;
}

int main()
{
    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device);

    printDeviceProperties(devProp);

    float* data = (float*)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
        data[i] = (float)i;
    }

    float* gpuData;
    cudaMalloc((void**)&gpuData, sizeof(float) * N);
    cudaMemcpy(gpuData, data, sizeof(float) * N, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    
    kernel<<<blocks, threads>>>(gpuData, N);
    //square<<<blocks, threads>>>(gpuData, N);
    cudaMemcpy(data, gpuData, sizeof(float) * N, cudaMemcpyDeviceToHost);

    /*
    for (int i = 0; i<N; i++)
    {
        printf("%f\n", data[i]);
    }
    */

    {
        Timer timer;
        kernel_intrinsic<<<blocks, threads>>>(gpuData, N, 10000);
    }

    {
        Timer timer;
        kernel_single<<<blocks, threads>>>(gpuData, N, 10000);
    }

    double* ddata = (double*)malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) {
        ddata[i] = (double)i;
    }

    double* gpuDouble;
    cudaMalloc(&gpuDouble, sizeof(double) * N);
    cudaMemcpy(gpuDouble, ddata, sizeof(double) * N, cudaMemcpyHostToDevice);

    {
        Timer timer;
        kernel_double<<<blocks, threads>>>(gpuDouble, N, 10000);
    }

    cudaFree(gpuData);
    cudaFree(gpuDouble);
    free(data);
    free(ddata);
    return 0;
}