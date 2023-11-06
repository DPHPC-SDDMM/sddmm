#include "cuda_sddmm.cuh"

__global__ void k_sddmm() {
    int index = threadIdx.x;
    int stride = blockDim.x;
}

void cuda_tiled_sddmm() {
    k_sddmm<<<1,256>>>();
}