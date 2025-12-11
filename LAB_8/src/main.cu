#include <iostream>
#include <cuda_runtime.h>

__global__ void reduction(int* data, int* result, int n) {
    extern __shared__ int sdata_int[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;    
    sdata_int[tid] = (i < n) ? data[i] : 0;

    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_int[tid] += sdata_int[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata_int[0];
    }
}

__device__ float f(float x) {
    return 4.0f / (1.0f + x * x);
}

__global__ void integrate_reduce(float* result, int n, float h) {
    extern __shared__ float sdata_float[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < n) {
        float x0 = idx * h;
        float x1 = (idx + 1) * h;
        val = 0.5f * h * (f(x0) + f(x1));
    }
    sdata_float[tid] = val;

    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_float[tid] += sdata_float[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata_float[0];
    }
}

__global__ void reduce_array(float* input, float* output, int n) {
    extern __shared__ float sdata_float[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_float[tid] = (idx < n) ? input[idx] : 0.0f;
    
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_float[tid] += sdata_float[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata_float[0];
    }
}

int main() {
    int N = 1 << 24;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t bytes = N * sizeof(int);

    int *data, *result, *tempBuffer;
    cudaMallocManaged(&data, bytes);
    cudaMallocManaged(&result, blocksPerGrid * sizeof(int));
    cudaMallocManaged(&tempBuffer, blocksPerGrid * sizeof(int));
    for (int i = 0; i < N; i++) {
        data[i] = 1;
    }

    int sharedMemSizeInt = threadsPerBlock * sizeof(int);
    reduction<<<blocksPerGrid, threadsPerBlock, sharedMemSizeInt>>>(data, result, N);
    cudaDeviceSynchronize();

    int remaining = blocksPerGrid;
    while (remaining > 1) {
        int newBlocksPerGrid = (remaining + threadsPerBlock - 1) / threadsPerBlock;

        int* swap = result;
        result = tempBuffer;
        tempBuffer = swap;

        reduction<<<newBlocksPerGrid, threadsPerBlock, sharedMemSizeInt>>>(tempBuffer, result, remaining);
        cudaDeviceSynchronize();

        remaining = newBlocksPerGrid;
    }

    std::cout << "Calculated Sum: " << result[0] << std::endl;
    std::cout << "Expected Sum:   " << N << std::endl;
    cudaFree(data);
    cudaFree(result);
    cudaFree(tempBuffer);


    // ---------------------------------------------------------

    int N_trap = 10000000;
    float h = 1.0f / N_trap;    
    blocksPerGrid = (N_trap + threadsPerBlock - 1) / threadsPerBlock;

    float *result_f, *tempBuffer_f;
    cudaMallocManaged(&result_f, blocksPerGrid * sizeof(float));
    cudaMallocManaged(&tempBuffer_f, blocksPerGrid * sizeof(float));

    int sharedMemSizeFloat = threadsPerBlock * sizeof(float);
    integrate_reduce<<<blocksPerGrid, threadsPerBlock, sharedMemSizeFloat>>>(result_f, N_trap, h);
    cudaDeviceSynchronize();

    remaining = blocksPerGrid;
    while (remaining > 1) {
        int newBlocksPerGrid = (remaining + threadsPerBlock - 1) / threadsPerBlock;

        float* swap = result_f;
        result_f = tempBuffer_f;
        tempBuffer_f = swap;

        reduce_array<<<newBlocksPerGrid, threadsPerBlock, sharedMemSizeFloat>>>(tempBuffer_f, result_f, remaining);
        (cudaDeviceSynchronize());

        remaining = newBlocksPerGrid;
    }

    std::cout << "Result: " << result_f[0] << std::endl;
    cudaFree(result_f);
    cudaFree(tempBuffer_f);

    return 0;
}