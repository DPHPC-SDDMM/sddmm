#include <vector>
#include <chrono>
#include <cusparse.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../defines.h"
#include "../../results.h"
#include "../../data_structures/matrix/matrix.h"
#include "../../data_structures/csr/csr.h"

// https://docs.nvidia.com/cuda/cusparse/

namespace SDDMM {
    namespace Algo {
        Types::CSR cuSPARSE_SDDMM(
            const Types::CSR& A_sparse, 
            const Types::Matrix& X_dense, 
            const Types::Matrix& Y_dense, 
            Results::ExperimentData* measurements = nullptr
        ){
            // A_sparse.col_idx;
            // A_sparse.row_ptr;
            // A_sparse.values;
            Types::vec_size_t* col_idx_d;
            Types::vec_size_t* row_ptr_d;
            Types::expmt_t* values_d;

            Types::vec_size_t sparse_len_values_d = sizeof(Types::expmt_t) * A_sparse.values.size();
            Types::vec_size_t sparse_len_row_ptr_d = sizeof(Types::vec_size_t) * A_sparse.row_ptr.size();
            Types::vec_size_t sparse_len_col_idx_d = sizeof(Types::vec_size_t) * A_sparse.col_idx.size();
            gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&values_d), sparse_len_values_d));
            gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&row_ptr_d), sparse_len_row_ptr_d));
            gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&col_idx_d), sparse_len_col_idx_d));
            
            Types::vec_size_t x_dense_len_values = X_dense.data.size();
            Types::vec_size_t x_dense_len_values_d = sizeof(Types::expmt_t) * x_dense_len_values;
            Types::vec_size_t y_dense_len_values = Y_dense.data.size();
            Types::vec_size_t y_dense_len_values_d = sizeof(Types::expmt_t) * y_dense_len_values;

            Types::expmt_t* x_values_d;
            Types::expmt_t* y_values_d;
            gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&x_values_d), x_dense_len_values_d));
            gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&y_values_d), y_dense_len_values_d));

            // copy dense data
            gpuErrchk(cudaMemcpy(x_values_d, X_dense.data.data(), x_dense_len_values_d, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(y_values_d, Y_dense.data.data(), y_dense_len_values_d, cudaMemcpyHostToDevice));

            // copy sparse data
            gpuErrchk(cudaMemcpy(values_d, A_sparse.values.data(), sparse_len_values_d, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(row_ptr_d, A_sparse.row_ptr.data(), sparse_len_row_ptr_d, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(col_idx_d, A_sparse.col_idx.data(), sparse_len_col_idx_d, cudaMemcpyHostToDevice));

            // cusparse api
            cusparseHandle_t     handle = NULL;
            cusparseDnMatDescr_t mat_X, mat_Y;
            cusparseSpMatDescr_t mat_spA;
            void*                dBuffer    = NULL;
            size_t               bufferSize = 0;

            float alpha = 1.0f;
            float beta = 0.0f;

            sparse_gpuErrchk(cusparseCreate(&handle));
            Types::vec_size_t leading_dimension_X = X_dense.m; // num cols
            Types::vec_size_t leading_dimension_Y = Y_dense.n; // num rows
            sparse_gpuErrchk(cusparseCreateDnMat(&mat_X, X_dense.n, X_dense.m, leading_dimension_X, x_values_d, Types::cuda_expmt_t, CUSPARSE_ORDER_ROW));
            sparse_gpuErrchk(cusparseCreateDnMat(&mat_Y, Y_dense.n, Y_dense.m, leading_dimension_Y, y_values_d, Types::cuda_expmt_t, CUSPARSE_ORDER_COL));

            sparse_gpuErrchk(cusparseCreateCsr(&mat_spA, A_sparse.n, A_sparse.m, A_sparse.values.size() /*nnz*/,
                               row_ptr_d, col_idx_d, values_d, 
                               Types::cuda_vec_size_t, Types::cuda_vec_size_t, 
                               CUSPARSE_INDEX_BASE_ZERO, Types::cuda_expmt_t));

            // allocate an external buffer if needed
            sparse_gpuErrchk(cusparseSDDMM_bufferSize(
                                        handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, mat_X, mat_Y, &beta, mat_spA, Types::cuda_expmt_t,
                                        CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize));
            gpuErrchk(cudaMalloc(&dBuffer, bufferSize));

            // execute preprocess (optional)
            sparse_gpuErrchk(cusparseSDDMM_preprocess(
                                        handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, mat_X, mat_Y, &beta, mat_spA, Types::cuda_expmt_t,
                                        CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer));

            auto start = std::chrono::high_resolution_clock::now();

            // execute SpMM
            // alpha(op(A)*op(B)).hadamard(spy(C)) + beta*C
            sparse_gpuErrchk(cusparseSDDMM(handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, mat_X, mat_Y, &beta, mat_spA, Types::cuda_expmt_t,
                                        CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer));

            auto end = std::chrono::high_resolution_clock::now();
            if(measurements != nullptr){
                Types::time_duration_unit duration = std::chrono::duration_cast<Types::time_measure_unit>(end - start).count();
                measurements->durations.push_back(duration);
            }

            // destroy matrix/vector descriptors
            sparse_gpuErrchk(cusparseDestroyDnMat(mat_X));
            sparse_gpuErrchk(cusparseDestroyDnMat(mat_Y));
            sparse_gpuErrchk(cusparseDestroySpMat(mat_spA));
            sparse_gpuErrchk(cusparseDestroy(handle));

            Types::CSR res;
            res.n = A_sparse.n;
            res.m = A_sparse.m;
            res.col_idx.resize(A_sparse.col_idx.size());
            res.row_ptr.resize(A_sparse.row_ptr.size());
            res.values.resize(A_sparse.values.size());

            // copy back
            cudaMemcpy(res.values.data(), values_d, sparse_len_values_d, cudaMemcpyDeviceToHost);

            gpuErrchk(cudaFree(dBuffer));
            gpuErrchk(cudaFree(x_values_d));
            gpuErrchk(cudaFree(y_values_d));
            gpuErrchk(cudaFree(col_idx_d));
            gpuErrchk(cudaFree(row_ptr_d));
            gpuErrchk(cudaFree(values_d));

            return res;
        }
    }
}