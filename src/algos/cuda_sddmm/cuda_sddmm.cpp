#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include "../../defines.h"
#include "../../data_structures/matrix/matrix.h"
#include "../../data_structures/coo/coo.h"

#include "cuda_sddmm.cuh"

namespace SDDMM {
    namespace Algo {
        Types::COO cuda_tiled_sddmm(
            const Types::COO& A_sparse, 
            const Types::Matrix& X_dense, 
            const Types::Matrix& Y_dense,
            Results::ExperimentData* measurements = nullptr
        ) {
            assert(X_dense.m == Y_dense.n && "Size of cols(X_dense) and rows(Y) must match!");
            assert(A_sparse.n>0 && A_sparse.m>0 && X_dense.n>0 && X_dense.m>0 && Y_dense.n>0 && Y_dense.m && "All involved matrices must be non-empty!");
            assert(A_sparse.n==X_dense.n && A_sparse.m==Y_dense.m && "Matrix dimensions must match!");

            auto start = std::chrono::high_resolution_clock::now();

            Types::vec_size_t inner_dense_dimension = X_dense.m;

            // get sparse data length and produce one in bytes
            Types::vec_size_t sparse_len = A_sparse.data.size();
            Types::vec_size_t sparse_len_d = sizeof(Types::COO::triplet) * sparse_len;
            Types::vec_size_t x_dense_len = X_dense.data.size();
            Types::vec_size_t x_dense_len_d = sizeof(Types::expmt_t) * x_dense_len;
            Types::vec_size_t y_dense_len = Y_dense.data.size();
            Types::vec_size_t y_dense_len_d = sizeof(Types::expmt_t) * y_dense_len;

            Types::COO::triplet* out_d;
            cudaMalloc(reinterpret_cast<void**>(&out_d), sparse_len_d);

            Types::COO::triplet* A_sparse_d;
            Types::expmt_t* X_dense_d;
            Types::expmt_t* Y_dense_d;
            cudaMalloc(reinterpret_cast<void**>(&A_sparse_d), sparse_len_d);
            cudaMalloc(reinterpret_cast<void**>(&X_dense_d), x_dense_len_d);
            cudaMalloc(reinterpret_cast<void**>(&Y_dense_d), y_dense_len_d);

            cudaMemcpy(A_sparse_d, A_sparse.data.data(), sparse_len_d, cudaMemcpyHostToDevice);
            cudaMemcpy(X_dense_d, X_dense.data.data(), x_dense_len_d, cudaMemcpyHostToDevice);
            cudaMemcpy(Y_dense_d, Y_dense.data.data(), y_dense_len_d, cudaMemcpyHostToDevice);

            CudaTiledSDDMM(
                A_sparse_d, X_dense_d, Y_dense_d, sparse_len,
                X_dense.m, Y_dense.m, out_d
            );

            Types::COO::triplet* out = new Types::COO::triplet[sparse_len];
            cudaMemcpy(out, out_d, sparse_len_d, cudaMemcpyDeviceToHost);

            cudaFree(A_sparse_d);
            cudaFree(X_dense_d);
            cudaFree(Y_dense_d);
            cudaFree(out_d);

            Types::COO out_sparse;
            out_sparse.n = A_sparse.n;
            out_sparse.m = A_sparse.m;
            out_sparse.data.reserve(A_sparse.data.size());
            // make some space
            auto s = A_sparse.data.size();
            for(Types::vec_size_t i=0; i<s; ++i){
                auto v = out[i];
                if(v.value != 0) {
                    out_sparse.data.push_back(v);
                }
            }
            out_sparse.data.shrink_to_fit();

            auto end = std::chrono::high_resolution_clock::now();

            if(measurements != nullptr){
                Types::time_duration_unit duration = std::chrono::duration_cast<Types::time_measure_unit>(end - start).count();
                measurements->durations.push_back(duration);
            }

            return out_sparse;
        }
    }
}