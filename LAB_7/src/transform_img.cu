#include <cstdlib>
#include <fstream>
#include <cuda_runtime.h>

#define CHANNEL_NUM 3

bool readImageFile(unsigned char* img, int size) {
    std::ifstream f("image.raw", std::ios::binary);
    if (!f) {
        return false;
    }

    f.read((char*)img, size);
    return f.good();
}

void writeImageFile(unsigned char* img, int size) {    
    std::ofstream f("output.raw", std::ios::binary);
    f.write((char*)img, size);
}

__global__ void brightness_kernel(unsigned char* img, int w, int h, int delta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        int idx = (y * w + x) * 3;

        for (int c = 0; c < 3; c++) {
            int val = (int)img[idx + c] + delta;
            img[idx + c] = (unsigned char)min(255, max(0, val));
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
        return 1;
    }

    int WIDTH = atoi(argv[1]);
    int HEIGHT = atoi(argv[2]);
    int DELTA = atoi(argv[3]);

    int img_total_size = WIDTH * HEIGHT * CHANNEL_NUM;
    unsigned char* img = (unsigned char*)malloc(img_total_size);
    
    if (!img) {
        return 1;
    }

    if (!readImageFile(img, img_total_size)) {
        free(img);
        return 1;
    }

    unsigned char* gpuIMG;
    cudaError_t err = cudaMalloc((void**)&gpuIMG, img_total_size);
    cudaMemcpy(gpuIMG, img, img_total_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16); 
    dim3 dimGrid((WIDTH + dimBlock.x - 1) / dimBlock.x, (HEIGHT + dimBlock.y - 1) / dimBlock.y);

    brightness_kernel<<<dimGrid, dimBlock>>>(gpuIMG, WIDTH, HEIGHT, DELTA);

    cudaDeviceSynchronize();
    cudaMemcpy(img, gpuIMG, img_total_size, cudaMemcpyDeviceToHost);

    writeImageFile(img, img_total_size);

    cudaFree(gpuIMG);
    free(img);
    
    return 0;
}