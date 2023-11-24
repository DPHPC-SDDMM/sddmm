#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include "../../defines.h"
#include "../../results.h"
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
            assert(A_sparse.n>0 && A_sparse.m>0 && X_dense.n>0 && X_dense.m>0 && Y_dense.n>0 && Y_dense.m > 0 && "All involved matrices must be non-empty!");
            assert(A_sparse.n==X_dense.n && A_sparse.m==Y_dense.m && "Sparse and dense matrices dimensions must match!");

            auto start = std::chrono::high_resolution_clock::now();

            Types::vec_size_t inner_dense_dimension = X_dense.m;

            // get sparse data length and produce one in bytes
            Types::vec_size_t sparse_len = A_sparse.values.size();
            Types::vec_size_t sparse_len_values_d = sizeof(Types::expmt_t) * sparse_len;
            Types::vec_size_t sparse_len_rows_d = sizeof(Types::vec_size_t) * sparse_len;
            Types::vec_size_t sparse_len_cols_d = sizeof(Types::vec_size_t) * sparse_len;

            Types::vec_size_t x_dense_len_values = X_dense.data.size();
            Types::vec_size_t x_dense_len_values_d = sizeof(Types::expmt_t) * x_dense_len_values;
            Types::vec_size_t y_dense_len_values = Y_dense.data.size();
            Types::vec_size_t y_dense_len_values_d = sizeof(Types::expmt_t) * y_dense_len_values;

            Types::vec_size_t* out_col_d;
            cudaMalloc(reinterpret_cast<void**>(&out_col_d), sparse_len_cols_d);
            Types::vec_size_t* out_row_d;
            cudaMalloc(reinterpret_cast<void**>(&out_row_d), sparse_len_rows_d);
            Types::expmt_t* out_values_d;
            cudaMalloc(reinterpret_cast<void**>(&out_values_d), sparse_len_values_d);

            Types::expmt_t* A_sparse_values_d;
            Types::vec_size_t* A_sparse_rows_d;
            Types::vec_size_t* A_sparse_cols_d;

            Types::expmt_t* X_dense_d;
            Types::expmt_t* Y_dense_d;
            cudaMalloc(reinterpret_cast<void**>(&A_sparse_values_d), sparse_len_values_d);
            cudaMalloc(reinterpret_cast<void**>(&A_sparse_rows_d), sparse_len_rows_d);
            cudaMalloc(reinterpret_cast<void**>(&A_sparse_cols_d), sparse_len_cols_d);
            cudaMalloc(reinterpret_cast<void**>(&X_dense_d), x_dense_len_values_d);
            cudaMalloc(reinterpret_cast<void**>(&Y_dense_d), y_dense_len_values_d);

            auto err1 = cudaMemcpy(A_sparse_values_d, A_sparse.values.data(), sparse_len_values_d, cudaMemcpyHostToDevice);
            auto err2 = cudaMemcpy(A_sparse_rows_d, A_sparse.rows.data(), sparse_len_rows_d, cudaMemcpyHostToDevice);
            auto err3 = cudaMemcpy(A_sparse_cols_d, A_sparse.cols.data(), sparse_len_cols_d, cudaMemcpyHostToDevice);

            auto err4 = cudaMemcpy(X_dense_d, X_dense.data.data(), x_dense_len_values_d, cudaMemcpyHostToDevice);
            auto err5 = cudaMemcpy(Y_dense_d, Y_dense.data.data(), y_dense_len_values_d, cudaMemcpyHostToDevice);

            CudaTiledSDDMM(
                A_sparse_values_d, 
                A_sparse_rows_d, 
                A_sparse_cols_d, 
                X_dense_d, 
                Y_dense_d, 
                sparse_len,
                X_dense.m, 
                Y_dense.m, 
                out_values_d, 
                out_row_d, 
                out_col_d
            );

            Types::expmt_t* out_values = new Types::expmt_t[sparse_len];
            Types::vec_size_t* out_rows = new Types::vec_size_t[sparse_len];
            Types::vec_size_t* out_cols = new Types::vec_size_t[sparse_len];
            auto err6 = cudaMemcpy(out_values, out_values_d, sparse_len_values_d, cudaMemcpyDeviceToHost);
            auto err7 = cudaMemcpy(out_rows, out_row_d, sparse_len_rows_d, cudaMemcpyDeviceToHost);
            auto err8 = cudaMemcpy(out_cols, out_col_d, sparse_len_cols_d, cudaMemcpyDeviceToHost);

            cudaFree(A_sparse_values_d);
            cudaFree(A_sparse_rows_d);
            cudaFree(A_sparse_cols_d);
            cudaFree(X_dense_d);
            cudaFree(Y_dense_d);
            cudaFree(out_values_d);
            cudaFree(out_row_d);
            cudaFree(out_col_d);

            Types::COO out_sparse;
            out_sparse.n = A_sparse.n;
            out_sparse.m = A_sparse.m;
            out_sparse.values.reserve(A_sparse.values.size()); // pre-emptively allocate the required memory

            auto s = A_sparse.values.size();
            for(Types::vec_size_t i=0; i<s; ++i){
                auto v = out_values[i];
                if(v != 0) {
                    out_sparse.values.push_back(v);
                    out_sparse.rows.push_back(out_rows[i]);
                    out_sparse.cols.push_back(out_cols[i]);
                }
            }

            auto end = std::chrono::high_resolution_clock::now();

            if(measurements != nullptr){
                Types::time_duration_unit duration = std::chrono::duration_cast<Types::time_measure_unit>(end - start).count();
                measurements->durations.push_back(duration);
            }

            out_sparse.values.shrink_to_fit(); // SDDMM may have less entries than A_sparse, due to zero inner products forming.
            out_sparse.rows.shrink_to_fit();
            out_sparse.cols.shrink_to_fit();

            return out_sparse;
        }
    }
}