#include "cuda_plus_x_struct.cuh"

__global__ void k_cuda(
    SDDMM::CUDA_EXAMPLES::triplet *in, 
    SDDMM::CUDA_EXAMPLES::triplet *out, 
    SDDMM::Types::vec_size_t len, 
    SDDMM::Types::expmt_t x
) {
    SDDMM::Types::vec_size_t index = static_cast<SDDMM::Types::vec_size_t>(threadIdx.x);
    SDDMM::Types::vec_size_t stride = static_cast<SDDMM::Types::vec_size_t>(blockDim.x);

    for(SDDMM::Types::vec_size_t i = index; i < len; i += stride){
        out[i].value = in[i].value + x;
        out[i].row = in[i].row + 1;
        out[i].col = in[i].col - 1;
    }
}

void run_k_struct(
    SDDMM::CUDA_EXAMPLES::triplet *in,
    SDDMM::CUDA_EXAMPLES::triplet *out, 
    SDDMM::Types::vec_size_t len, 
    SDDMM::Types::expmt_t x
) {
    // M thread blocks with T threads each
    // <<<M, T>>>
    int thread_num = 256;
    k_cuda<<<1,thread_num>>>(in, out, len, x);
}