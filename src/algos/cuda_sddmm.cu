#include "cuda_sddmm.cuh"

__global__ void k_sddmm(
    SDDMM::Types::COO::triplet* a_sparse,
    SDDMM::Types::expmt_t* x_dense,
    SDDMM::Types::expmt_t* y_dense,
    SDDMM::Types::COO::triplet* out_sparse
) {
    int index = threadIdx.x;
    int stride = blockDim.x;
}

void cuda_tiled_sddmm(
    const SDDMM::Types::COO::triplet* a_sparse,
    const SDDMM::Types::vec_size_t a_size,
    const SDDMM::Types::expmt_t* x_dense,
    const SDDMM::Types::vec_size_t x_size,
    const SDDMM::Types::expmt_t* y_dense,
    const SDDMM::Types::vec_size_t y_size,
    SDDMM::Types::COO::triplet* out_sparse
) 
{
    // SDDMM::Types::COO::triplet* a_sparse_d;
    // SDDMM::Types::expmt_t* x_dense_d;
    // SDDMM::Types::expmt_t* y_dense_d;
    // SDDMM::Types::COO::triplet* out_sparse_d;

    const void* a_sparse_loc =  reinterpret_cast<const void*>(a_sparse);
    void* out_sparse_loc =  reinterpret_cast<void*>(out_sparse);
    const void* x_dense_loc =  reinterpret_cast<const void*>(x_dense);
    const void* y_dense_loc =  reinterpret_cast<const void*>(y_dense);

    cudaMalloc(&a_sparse_loc, a_size);
    cudaMalloc(&out_sparse_loc, a_size);
    cudaMalloc(&x_dense_loc, x_size);
    cudaMalloc(&y_dense_loc, y_size);

    SDDMM::Types::COO::triplet* a_sparse_d;
    SDDMM::Types::expmt_t* x_dense_d;
    SDDMM::Types::expmt_t* y_dense_d;
    SDDMM::Types::COO::triplet* out_sparse_d;

    cudaMemcpy(a_sparse_d, a_sparse_loc, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(x_dense_d, x_dense_loc, x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_dense_d, y_dense_loc, y_size, cudaMemcpyHostToDevice);

    k_sddmm<<<1,1>>>(a_sparse_d, x_dense_d, y_dense_d, out_sparse_d);
}