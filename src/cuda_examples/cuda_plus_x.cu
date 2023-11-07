#include "cuda_plus_x.cuh"

__global__ void k_cuda(SDDMM::Types::expmt_t *in, SDDMM::Types::expmt_t *out, SDDMM::Types::vec_size_t len, SDDMM::Types::expmt_t x) {
    SDDMM::Types::vec_size_t index = static_cast<SDDMM::Types::vec_size_t>(threadIdx.x);
    SDDMM::Types::vec_size_t stride = static_cast<SDDMM::Types::vec_size_t>(blockDim.x);

    for(SDDMM::Types::vec_size_t i = index; i < len; i += stride){
        out[i] = in[i] + x;
    }
}

void run_k(SDDMM::Types::expmt_t *in, SDDMM::Types::expmt_t *out, SDDMM::Types::vec_size_t len, SDDMM::Types::expmt_t x) {
    // M thread blocks with T threads each
    // <<<M, T>>>
    k_cuda<<<1,256>>>(in, out, len, x);
}