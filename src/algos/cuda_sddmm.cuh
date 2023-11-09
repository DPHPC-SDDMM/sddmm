#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../data_structures/matrix/matrix.h"
#include "../data_structures/coo/coo.h"
#include <cmath>

extern "C" void CudaTiledSDDMM(
    SDDMM::Types::COO::triplet* A_sparse_d, 
    SDDMM::Types::expmt_t* X_dense_d,
    SDDMM::Types::expmt_t* Y_dense_d,
    SDDMM::Types::vec_size_t sparse_len,
    SDDMM::Types::vec_size_t X_m, 
    SDDMM::Types::vec_size_t Y_m,
    SDDMM::Types::COO::triplet* out_d
);