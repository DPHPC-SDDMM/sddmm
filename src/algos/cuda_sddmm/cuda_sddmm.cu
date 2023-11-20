#include "cuda_sddmm.cuh"

__global__ void k_sddmm(
    SDDMM::Types::expmt_t* A_sparse_values_d,
    SDDMM::Types::vec_size_t* A_sparse_rows_d,
    SDDMM::Types::vec_size_t* A_sparse_cols_d,
    SDDMM::Types::expmt_t* X_dense_d,
    SDDMM::Types::expmt_t* Y_dense_d,
    // !!!! SDDMM::Types::vec_size_t sparse_len,
    SDDMM::Types::vec_size_t X_m, 
    SDDMM::Types::vec_size_t Y_m,
    SDDMM::Types::expmt_t* out_values_d,
    SDDMM::Types::vec_size_t* out_row_d,
    SDDMM::Types::vec_size_t* out_col_d
) {
    // Each CUDA thread is responsible for computing one entry
    // of the output sparse matrix `out_d`.
    
    int index = threadIdx.x;
    int stride = blockDim.x;
    int blockNum = blockIdx.x;

    SDDMM::Types::vec_size_t access_ind = index + blockNum*stride;
    SDDMM::Types::expmt_t val = A_sparse_values_d[access_ind];
    SDDMM::Types::vec_size_t row = A_sparse_rows_d[access_ind];
    SDDMM::Types::vec_size_t col = A_sparse_cols_d[access_ind];

    SDDMM::Types::expmt_t inner_product = 0;
    
    // the ind index has to be tiled later
    // X == X_n x X_m
    // Y == Y_n x Y_m
    // ==> X_m == Y_n (if Y_n existed)
    for(SDDMM::Types::vec_size_t ind=0; ind < X_m; ++ind){
        inner_product += X_dense_d[row * X_m + ind]*Y_dense_d[ind * Y_m + col];
    }

    out_values_d[access_ind] = val*inner_product;
    out_row_d[access_ind] = row;
    out_col_d[access_ind] = col;
}

void CudaTiledSDDMM(
    SDDMM::Types::expmt_t* A_sparse_values_d,
    SDDMM::Types::vec_size_t* A_sparse_rows_d,
    SDDMM::Types::vec_size_t* A_sparse_cols_d,
    SDDMM::Types::expmt_t* X_dense_d,
    SDDMM::Types::expmt_t* Y_dense_d,
    SDDMM::Types::vec_size_t sparse_len,
    SDDMM::Types::vec_size_t X_m, 
    SDDMM::Types::vec_size_t Y_m,
    SDDMM::Types::expmt_t* out_values_d,
    SDDMM::Types::vec_size_t* out_row_d,
    SDDMM::Types::vec_size_t* out_col_d
) {
    // How many blocks do we need to fit the required threads?
    SDDMM::Types::vec_size_t block_num = 
        static_cast<SDDMM::Types::vec_size_t>(
            std::ceil(
                static_cast<double>(sparse_len) / static_cast<double>(SDDMM::Defines::warp_size)
            ));

    k_sddmm<<<block_num, SDDMM::Defines::warp_size>>>(
        A_sparse_values_d,
        A_sparse_rows_d,
        A_sparse_cols_d,
        X_dense_d, 
        Y_dense_d, 
        X_m, 
        Y_m, 
        out_values_d,
        out_row_d,
        out_col_d
    );
}