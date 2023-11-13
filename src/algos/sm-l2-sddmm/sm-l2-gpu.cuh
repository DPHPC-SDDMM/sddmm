#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../defines.h"

void run_kernel(int num_threadblocks,
                SDDMM::Types::vec_size_t* S_tile_rows, SDDMM::Types::vec_size_t* S_tile_cols, float* S_tile_values, SDDMM::Types::vec_size_t S_size,
                SDDMM::Types::vec_size_t* S_tile_starts, SDDMM::Types::vec_size_t S_tile_starts_size,
                float* P_tile_values,
                SDDMM::Types::vec_size_t* active_rows, SDDMM::Types::vec_size_t active_rows_size,
                float* A, float* B,
                SDDMM::Types::vec_size_t Tj, SDDMM::Types::vec_size_t Tk, SDDMM::Types::vec_size_t Ti,
                SDDMM::Types::vec_size_t tile_j_id,
                SDDMM::Types::vec_size_t tile_k_id,
                SDDMM::Types::vec_size_t num_Tks,
                SDDMM::Types::vec_size_t N, SDDMM::Types::vec_size_t M, SDDMM::Types::vec_size_t K);