#include "sm-l2-gpu.cuh"
#include "../../defines.h"
#include "cstdio"

#define WARP_SIZE 32
#define V_WARP_SIZE 4

namespace SDDMM {
    namespace Algo {
        namespace SML2SDDMM_Kernel {
            __global__ void SM_L2_GPU(const SDDMM::Types::vec_size_t* __restrict__ S_tile_rows, const SDDMM::Types::vec_size_t* __restrict__ S_tile_cols, const float* __restrict__ S_tile_values,
                                    float* P_tile_values,
                                    const SDDMM::Types::vec_size_t* S_tile_starts,
                                    const SDDMM::Types::vec_size_t* __restrict__ active_rows, SDDMM::Types::vec_size_t active_rows_size,
                                    const float* __restrict__ A, const float* __restrict__ B,
                                    SDDMM::Types::vec_size_t Tk, SDDMM::Types::vec_size_t Ti,
                                    SDDMM::Types::vec_size_t tile_k_id,
                                    SDDMM::Types::vec_size_t num_k_tiles,
                                    SDDMM::Types::vec_size_t K) {

                auto t_id = threadIdx.x;
                auto tile_no = blockIdx.x;

                auto lane_id = t_id % V_WARP_SIZE;

                auto tile_start = S_tile_starts[tile_no];
                auto tile_end = S_tile_starts[tile_no + 1];

                // allocate shared memory and load active rows of the current threadblock into it
                extern __shared__ float sh_actv_A[];

                auto w_id = t_id / WARP_SIZE;
                auto num_warps = blockDim.x / WARP_SIZE;
                auto t = t_id % WARP_SIZE;

                for (auto i = w_id; i < Ti && (tile_no * Ti + i) < active_rows_size; i += num_warps) {
                    auto active_r_id = active_rows[tile_no * Ti + i];
                    for (int j = 0; j < Tk; j += WARP_SIZE) {
                        sh_actv_A[i * Tk + t + j] = A[active_r_id * K + tile_k_id * Tk + t + j];
                    }
                }

                __syncthreads();

                // process non-zero elements

            //    // 1 thread per 1 element (no V_WARPs)
            //    for (auto idx = t_id + tile_start; idx < tile_end; idx += blockDim.x){
            //        float sum = 0;
            //        auto sh_rID = S_tile_rows[idx] - tile_no * Ti;
            //
            //        for (auto l = 0; l < Tk; l++) {
            //            sum += sh_actv_A[sh_rID * Tk + l] * B[S_tile_cols[idx] * K + l + tile_k_id * Tk];
            //        }
            //
            //        P_tile_values[idx] += sum;
            //
            //        if (tile_k_id == num_Tks - 1) {
            //            P_tile_values[idx] *= S_tile_values[idx];
            //        }
            //    }

                // V_WARPs + vectorisation (float4) + unrolling (2x)
                for (auto idx = t_id / V_WARP_SIZE + tile_start; idx < tile_end; idx += blockDim.x / V_WARP_SIZE) {
                    auto sh_r_id = S_tile_rows[idx] - tile_no * Ti;
            //        auto lane_id = t_id % V_WARP_SIZE;

                    // unroll the loop and vectorize
                    float sum1 = 0;
                    float sum2 = 0;

                    // divide the row-col product calculation into chunks (each of size Tk / V_WARP_SIZE)
                    int u_factor = 8; // unrolling factor * vector size
            //        int u_factor = 2;
                    for (auto l = lane_id * Tk / V_WARP_SIZE; l < (lane_id + 1) * Tk / V_WARP_SIZE - u_factor + 1; l += u_factor) {
                        float4 sh1 = *((float4*)& sh_actv_A[sh_r_id * Tk + l]);
                        float4 B1 = *((float4*)& B[S_tile_cols[idx] * K + l + tile_k_id * Tk]);
                        sum1 += sh1.w * B1.w + sh1.x * B1.x + sh1.y * B1.y + sh1.z * B1.z;

                        float4 sh2 = *((float4*)& sh_actv_A[sh_r_id * Tk + l + 4]);
                        float4 B2 = *((float4*)& B[S_tile_cols[idx] * K + l + tile_k_id * Tk + 4]);
                        sum2 += sh2.w * B2.w + sh2.x * B2.x + sh2.y * B2.y + sh2.z * B2.z;

                        // no vectorisation and/or no unrolling
            //            float sh1 = sh_actv_A[sh_r_id * Tk + l];
            //            float B1 = B[S_tile_cols[idx] * K + l + tile_k_id * Tk];
            //            sum1 += (sh1 * B1);
            //
            //            float sh2 = sh_actv_A[sh_r_id * Tk + l + 1];
            //            float B2 = B[S_tile_cols[idx] * K + l + tile_k_id * Tk + 1];
            //            sum2 += (sh2 * B2);
                    }

                    // reduce in a virtual warp
                    for (int vws = V_WARP_SIZE / 2; vws > 0; vws /= 2) {
                        // set the first V_WARP_SIZE bits to 1 so that only threads within a V_WARP participate in reduction
                        unsigned mask = (1 << V_WARP_SIZE) - 1;
            //            unsigned mask = 0xffffffff;

                        // __shfl_xor is deprecated in cuda >=  9.0
                        sum1 += __shfl_xor_sync(mask, sum1, vws);
                        sum2 += __shfl_xor_sync(mask, sum2, vws);
                        // ===================================================
                    }

                    // P_tile_values[idx] = S_tile_values[idx] * (sum1 + sum2);
                    P_tile_values[idx] += (sum1 + sum2);
            //        P_tile_values[idx] += (sum1);

                    // in the last K-tile, multiple with S
                    if (tile_k_id == num_k_tiles - 1) {
                        P_tile_values[idx] *= S_tile_values[idx];
                    }
                }

                __syncthreads();
            }

            void run_kernel(int num_threadblocks,
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
                            SDDMM::Types::vec_size_t K,
                            cudaStream_t stream
                ) {

                int sm_size = Ti * Tk * sizeof(float);
                SM_L2_GPU <<<num_threadblocks, 512, sm_size, stream>>> (S_tile_rows, S_tile_cols, S_tile_values,
                                                                P_tile_values,
                                                                S_tile_starts,
                                                                active_rows, active_rows_size,
                                                                A, B,
                                                                Tk, Ti,
                                                                tile_k_id,
                                                                num_k_tiles,
                                                                K);
            }
        }
    }
}