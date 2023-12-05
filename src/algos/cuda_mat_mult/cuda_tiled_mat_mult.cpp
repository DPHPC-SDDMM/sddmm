#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include "../../defines.h"
#include "../../results.h"
#include "../../data_structures/matrix/matrix.h"

#include "cuda_tiled_mat_mult.cuh"

/**
 * @brief This one should be used to test the cache size of a GPU and maybe general single core
 * performance of the GPU
*/

namespace SDDMM {
    namespace Algo {
        Types::Matrix cuda_tiled_mat_mult(
            const Types::Matrix& X, 
            const Types::Matrix& Y,
            Types::vec_size_t ts,
            Results::ExperimentData* measurements = nullptr
        ) {
            assert(X.m == Y.n && "Size of cols(X_dense) and rows(Y) must match!");
            assert(X.n>0 && X.m>0 && Y.n>0 && Y.m > 0 && "All involved matrices must be non-empty!");

            auto start = std::chrono::high_resolution_clock::now();

            Types::vec_size_t inner_dense_dimension = X.m;

            Types::vec_size_t x_len_values = X.data.size();
            Types::vec_size_t x_len_values_d = sizeof(Types::expmt_t) * x_len_values;
            Types::vec_size_t y_len_values = Y.data.size();
            Types::vec_size_t y_len_values_d = sizeof(Types::expmt_t) * y_len_values;
            Types::vec_size_t xy_len_values = X.n*Y.m;
            Types::vec_size_t xy_len_values_d = sizeof(Types::expmt_t) * xy_len_values;

            Types::expmt_t* X_d;
            Types::expmt_t* Y_d;
            Types::expmt_t* XY_out_d;
            cudaMalloc(reinterpret_cast<void**>(&X_d), x_len_values_d);
            cudaMalloc(reinterpret_cast<void**>(&Y_d), y_len_values_d);
            cudaMalloc(reinterpret_cast<void**>(&XY_out_d), xy_len_values_d);

            auto err4 = cudaMemcpy(X_d, X.data.data(), x_len_values_d, cudaMemcpyHostToDevice);
            auto err5 = cudaMemcpy(Y_d, Y.data.data(), y_len_values_d, cudaMemcpyHostToDevice);
            auto err8 = cudaMemcpy(XY_out_d, std::vector<Types::expmt_t>(xy_len_values, 0.0).data(), xy_len_values_d, cudaMemcpyHostToDevice);

            CUDA_TILED_MAT_MULT::CudaTiledMatMult (
                X_d, X.n, X.m, Y_d, Y.m, ts, XY_out_d
            );

            Types::Matrix XY_out(X.n, Y.m);
            auto err6 = cudaMemcpy(XY_out.data.data(), XY_out_d, xy_len_values_d, cudaMemcpyDeviceToHost);

            cudaFree(X_d);
            cudaFree(Y_d);
            cudaFree(XY_out_d);

            auto end = std::chrono::high_resolution_clock::now();

            if(measurements != nullptr){
                Types::time_duration_unit duration = std::chrono::duration_cast<Types::time_measure_unit>(end - start).count();
                measurements->durations.push_back(duration);
            }

            return XY_out;
        }
    }
}