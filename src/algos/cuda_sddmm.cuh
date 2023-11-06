#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../data_structures/matrix/matrix.h"
#include "../data_structures/coo/coo.h"

extern "C" void cuda_tiled_sddmm(
    const SDDMM::Types::COO::triplet* a_sparse,
    const SDDMM::Types::vec_size_t a_size,
    const SDDMM::Types::expmt_t* x_dense,
    const SDDMM::Types::vec_size_t x_size,
    const SDDMM::Types::expmt_t* y_dense,
    const SDDMM::Types::vec_size_t y_size,
    SDDMM::Types::COO::triplet* out_sparse
);