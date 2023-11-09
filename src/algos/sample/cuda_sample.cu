#include "cuda_sample.cuh"

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int index = threadIdx.x;
    int stride = blockDim.x;

    for(int i = index; i < n; i += stride){
        out[i] = a[i] + b[i];
    }
}

void run(float* d_out, float* d_a, float* d_b, int N) {
    vector_add<<<1,256>>>(d_out, d_a, d_b, N);
}