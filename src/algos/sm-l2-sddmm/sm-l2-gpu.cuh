#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../defines.h"

namespace SDDMM {
    namespace Algo {
        namespace SML2SDDMM_Kernel {
            extern "C" void run_kernel(int num_threadblocks,
                const SDDMM::Types::vec_size_t* __restrict__ S_tile_rows, 
                const SDDMM::Types::vec_size_t* __restrict__ S_tile_cols, 
                const float* __restrict__ S_tile_values,
                float* P_tile_values,
                const SDDMM::Types::vec_size_t* S_tile_starts,
                const SDDMM::Types::vec_size_t* __restrict__ active_rows, 
                SDDMM::Types::vec_size_t active_rows_size,
                const float* __restrict__ A, 
                const float* __restrict__ B,
                SDDMM::Types::vec_size_t Tk, 
                SDDMM::Types::vec_size_t Ti,
                SDDMM::Types::vec_size_t tile_k_id,
                SDDMM::Types::vec_size_t num_k_tiles,
                SDDMM::Types::vec_size_t K);
        }
    }
}