#include "cuda_sddmm.cuh"

__global__ void k_sddmm(
    SDDMM::Types::COO::triplet* A_sparse_d, 
    SDDMM::Types::expmt_t* X_dense_d,
    SDDMM::Types::expmt_t* Y_dense_d,
    SDDMM::Types::vec_size_t X_m, 
    SDDMM::Types::vec_size_t Y_m,
    SDDMM::Types::COO::triplet* out_d
) {
    // Each CUDA thread is responsible for computing one entry
    // of the output sparse matrix `out_d`.
    
    int index = threadIdx.x;
    int stride = blockDim.x;
    int blockNum = blockIdx.x;

    SDDMM::Types::vec_size_t access_ind = index + blockNum*stride;
    SDDMM::Types::COO::triplet p = A_sparse_d[access_ind];
    SDDMM::Types::expmt_t inner_product = 0;
    
    // the ind index has to be tiled later
    // X == X_n x X_m
    // Y == Y_n x Y_m
    // ==> X_m == Y_n (if Y_n existed)
    for(SDDMM::Types::vec_size_t ind=0; ind < X_m; ++ind){
        inner_product += X_dense_d[p.row * X_m + ind]*Y_dense_d[ind * Y_m + p.col];
    }

    out_d[access_ind] = SDDMM::Types::COO::triplet{
        .row = p.row, 
        .col = p.col, 
        .value = p.value * inner_product
    };
}

void CudaTiledSDDMM(
    SDDMM::Types::COO::triplet* A_sparse_d, 
    SDDMM::Types::expmt_t* X_dense_d,
    SDDMM::Types::expmt_t* Y_dense_d,
    SDDMM::Types::vec_size_t sparse_len,
    SDDMM::Types::vec_size_t X_m, 
    SDDMM::Types::vec_size_t Y_m,
    SDDMM::Types::COO::triplet* out_d
) 
{
    // How many blocks do we need to fit the required threads?
    SDDMM::Types::vec_size_t block_num = 
        static_cast<SDDMM::Types::vec_size_t>(
            std::ceil(
                static_cast<double>(sparse_len) / static_cast<double>(SDDMM::Defines::warp_size)
            ));
    k_sddmm<<<block_num, SDDMM::Defines::warp_size>>>(A_sparse_d, X_dense_d, Y_dense_d, X_m, Y_m, out_d);
}