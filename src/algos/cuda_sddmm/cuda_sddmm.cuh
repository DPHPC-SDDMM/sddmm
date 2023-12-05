#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../data_structures/matrix/matrix.h"
#include "../../data_structures/coo/coo.h"
#include <cmath>

namespace SDDMM {
    namespace Algo {
        namespace CUDA_SDDMM {
            extern "C" void CudaSDDMM(
                SDDMM::Types::expmt_t* A_sparse_values_d,
                SDDMM::Types::vec_size_t* A_sparse_rows_d,
                SDDMM::Types::vec_size_t* A_sparse_cols_d,
                SDDMM::Types::expmt_t* X_dense_d,
                SDDMM::Types::expmt_t* Y_dense_d,
                SDDMM::Types::vec_size_t sparse_len,
                SDDMM::Types::vec_size_t X_m, 
                SDDMM::Types::vec_size_t Y_n,
                SDDMM::Types::expmt_t* out_values_d,
                SDDMM::Types::vec_size_t* out_row_d,
                SDDMM::Types::vec_size_t* out_col_d
            );
        }
    }
}