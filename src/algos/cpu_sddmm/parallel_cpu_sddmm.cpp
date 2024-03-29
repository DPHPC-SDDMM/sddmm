#pragma once

#include <vector>
#include <omp.h>
#include <iostream>
#include <math.h>
#include "../../defines.h"
#include "../../results.h"
#include "../../data_structures/matrix/matrix.h"
#include "../../data_structures/coo/coo.h"

/**
 * @brief Turn zero filtering of the results on or off
 * @note If this is turned of, the resulting matrices may contain zeros which doesn't conform to sparse matrix format
 * @note parallel_sddmm_cuda_simulation is not included in this define because it makes not sense to benchmark it
*/
// #define SDDMM_PARALLEL_CPU_ZERO_FILTER

namespace SDDMM {
    namespace Algo {
        /**
         * @brief This is the version taken from the github page of the paper people. It doesn't have a proper return value and
         * serves as baseline for an "algo only" run, without filtering zero values. This is the fastest non-tiled algo.
        */
        Types::expmt_t parallel_sddmm_git(
            const Types::COO& A_sparse, 
            const Types::Matrix& X_dense, 
            const Types::Matrix& Y_dense, 
            Types::vec_size_t num_threads,
            Results::ExperimentData* measurements = nullptr
        ){
            Types::vec_size_t k = X_dense.m;
            Types::vec_size_t nnz = A_sparse.values.size();
            std::vector<Types::expmt_t> p_ind(nnz);

            auto start = std::chrono::high_resolution_clock::now();
            // omp_set_num_threads(28);
            #pragma omp parallel for //reduction(+:tot)
            for (int ind = 0; ind < nnz; ind++){
                float sm =0 ;
                int row = A_sparse.rows[ind];
                int col = A_sparse.cols[ind]; 
                for (int t = 0; t < k; ++t)
                    sm += X_dense.data[row * k + t] * Y_dense.data[col * k + t];
                p_ind[ind] = sm * A_sparse.values[ind];
                // cout << "ind " << row<<" "<<col << ":: "  <<" "<< p_ind[ind] << " = " << sm <<" * "<< val_ind[ind]<< endl;  
                // }                
            }
            auto end = std::chrono::high_resolution_clock::now();

            Types::time_duration_unit duration = std::chrono::duration_cast<Types::time_measure_unit>(end - start).count();
            measurements->durations.push_back(duration);

            // return 1.0 so that there is something to be summed up for the compiler in release mode :D
            return 1.0;
        }

        /**
         * @brief NOTE: [This one "simulates" the way the algo is run on a GPU and is insanely slow on a CPU!] 
         * Computes the Sampled Dense-Dense Matrix Multiplication (SDDMM) operation
         * __using CPU parallelization__ with a sparse matrix in the COO matrix representation format.
         * The SDDMM consists of a Hadamard product (i.e. element-wise multiplication) between
         * - the dense matrix product XY between X_dense and Y_dense and
         * - a sparse matrix
         * 
         * @param A_sparse: A sparse matrix using the COO matrix representation format.
         * @param X_dense: The left-hand side (LHS) of the dense matrix product.
         * @param Y_dense: The right-hand side (RHS) of the dense matrix product.
         * @param num_threads: The number of threads which will be utilized for performing this operation.
         * @param measurements: Optional variable pointer which stores the time required to perform the operation. The duration time measure unit is defined in @ref "defines.h"
         * @returns (X @ Y) * A_sparse
         * 
         * @remark Potential zero values arising during computations are ignored, so as to comply with the COO format. 
         * 
         * @warning Dimensionality of matrices are expected to match each operation used, i.e.
         *  1) If X in R^{n x k}, then Y must be in R^{k x m}
         *  2) A_sparse must be in R^{n x k}
         * 
         * @sa
         * - [COO matrix format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO))
         * - [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
        */
        Types::COO parallel_sddmm_cuda_simulation(
            const Types::COO& A_sparse, 
            const Types::Matrix& X_dense, 
            const Types::Matrix& Y_dense, 
            Types::vec_size_t num_threads,
            Results::ExperimentData* measurements = nullptr
        ) {
            assert(X_dense.m == Y_dense.n && "Size of cols(X_dense) and rows(Y) must match!");
            assert(A_sparse.n>0 && A_sparse.m>0 && X_dense.n>0 && X_dense.m>0 && Y_dense.n>0 && Y_dense.m>0 && "All involved matrices must be non-empty!");
            assert(A_sparse.n==X_dense.n && A_sparse.m==Y_dense.m && "Matrix dimensions must match!");
            assert(X_dense.is_row_major() && Y_dense.is_col_major() && "X_dense must be row major, Y_dense must be col major!");

            auto start = std::chrono::high_resolution_clock::now();
            // auto start = omp_get_wtime();

            Types::COO res;
            res.n = A_sparse.n;
            res.m = A_sparse.m;

            Types::vec_size_t s = A_sparse.values.size();
            for(Types::vec_size_t i=0; i<s; i+=num_threads){
                // auto m = std::min(s - i, num_threads); // This line is commented out to resemble the *constant* number of threads when calling CUDA.

                // Subset of COO entries.
                // The term "block" was chosen to be reminiscent of a CUDA blocks, i.e. a set of CUDA threads.
                // std::vector<Types::COO::triplet> block(num_threads, {0,0,0});
                std::vector<Types::expmt_t> block_values(num_threads, 0);
                std::vector<Types::vec_size_t> block_rows(num_threads, 0);
                std::vector<Types::vec_size_t> block_cols(num_threads, 0);
                // // std::stringstream ss[2];

                // add about 750ns overhead
                const Types::vec_size_t m = X_dense.m;
                #pragma omp parallel
                // for(int tn=0; tn<num_threads; ++tn) // This for loop simulates the execution of each thread in the block.
                {
                    auto tn = omp_get_thread_num();
                    auto idx = i+tn;
                    if(idx < s) {
                        /* COO entry.
                        Determines
                        - row and column of X and Y respectively corresponding to this inner product,
                        - the constant with which to multiply the (final) inner product
                        Reminder: SDDMM consists of a Hadamard product between
                        - dense matrix multiplication --inner products-- XY and
                        - a sparse matrix A.
                        */
                        Types::vec_size_t row = A_sparse.rows[idx];
                        Types::vec_size_t col = A_sparse.cols[idx]; 
                        Types::expmt_t val = A_sparse.values[idx];

                        // ss[tn] << "[" << i << " " << tn << " " << p.row << " " << idx << "]\n";
                        Types::expmt_t inner_product = 0;
                        
                        // the ind index has to be tiled later
                        for(SDDMM::Types::vec_size_t ind=0; ind < m; ++ind){
                            inner_product += X_dense.at(row, ind)*Y_dense.at(ind, col);
                        }
                        /*
                        Here we have the entire inner prouct inside `inner_product`.

                        TODO:
                        This is due to change when tiling & streaming are taken into consideration,
                        since subsums of inner products will be computed instead.
                        These subsums will finally be combined (i.e. summed)
                        to generate the complete inner product.

                        e.g. Compute 1 row of the COO matrix.
                        STEP 1) "Freeze" PART of one row r of dense matrix X into the cache.
                        STEP 2) In each thread load PART of column c of dense matrix Y.
                                Obviously, this should have the same number of elements
                                as the part of the row that was loaded in STEP 1.
                                This is known as "streaming".
                        STEP 3) In each thread, compute partial inner products using the elements
                                loaded by STEP 1 and STEP 2.
                        STEP 4) GOTO STEP 1 by freezing a new part of the same row.
                                The new part should not have any overlaps with any previous iteration(s).
                        */
                        block_values[tn] = val * inner_product;
                        block_rows[tn] = row;
                        block_cols[tn] = col;
                    }
                }

                // about 100ns overhead per insertion
                for(int tn = 0; tn<num_threads; ++tn){
                    if(block_values[tn] != 0){ // Comply with the definition of a COO matrix (i.e. hold only non-zero values).
                        res.values.push_back(block_values[tn]);
                        res.rows.push_back(block_rows[tn]);
                        res.cols.push_back(block_cols[tn]);
                    }
                }

            }

            auto end = std::chrono::high_resolution_clock::now();
            // auto end = omp_get_wtime();

            if(measurements != nullptr){
                Types::time_duration_unit duration = std::chrono::duration_cast<Types::time_measure_unit>(end - start).count();
                measurements->durations.push_back(duration);
            }

            // Shrink the size of the data structures in case zero-valued inner products appeared,
            // thus requiring less than initial space predicted (i.e. memory amount equal to the input sparse matrix ).
            res.values.shrink_to_fit();
            res.rows.shrink_to_fit();
            res.cols.shrink_to_fit();

            return res;
        }

        /**
         * @brief Compute sddmm using slower ways
        */
        Types::COO parallel_sddmm_slow(
            const Types::COO& A_sparse, 
            const Types::Matrix& X_dense, 
            const Types::Matrix& Y_dense, 
            Types::vec_size_t num_threads,
            Results::ExperimentData* measurements = nullptr
        )
        {
            assert(X_dense.is_row_major() && Y_dense.is_col_major() && "X_dense must be row major, Y_dense must be col major!");

            auto start = std::chrono::high_resolution_clock::now();

            Types::vec_size_t k = X_dense.m;
            Types::vec_size_t nnz = A_sparse.values.size();
#ifdef SDDMM_PARALLEL_CPU_ZERO_FILTER
            std::vector<Types::expmt_t> p_ind(nnz, 0.0);
#endif
//             Types::COO res;
//             res.n = A_sparse.n;
//             res.m = A_sparse.m;

// #ifndef SDDMM_PARALLEL_CPU_ZERO_FILTER
//             res.rows.resize(nnz); 
//             res.cols.resize(nnz);
//             std::copy(A_sparse.rows.begin(), A_sparse.rows.end(), res.rows.begin());
//             std::copy(A_sparse.cols.begin(), A_sparse.cols.end(), res.cols.begin());
//             res.values.resize(nnz);
// #endif

            Types::COO res;
            res.n = A_sparse.n;
            res.m = A_sparse.m;
#ifndef SDDMM_PARALLEL_CPU_ZERO_FILTER
            res.rows.resize(nnz); 
            res.cols.resize(nnz);
            std::copy(A_sparse.rows.begin(), A_sparse.rows.end(), res.rows.begin());
            std::copy(A_sparse.cols.begin(), A_sparse.cols.end(), res.cols.begin());
            res.values.resize(nnz);
#endif

            // omp_set_num_threads(28);
            #pragma omp parallel for
            for (int64_t ind = 0; ind < nnz; ind++){
                Types::vec_size_t row = A_sparse.rows[ind];
                Types::vec_size_t col = A_sparse.cols[ind]; 
                Types::expmt_t val = A_sparse.values[ind];
                Types::expmt_t inner_product = 0;
                for (Types::vec_size_t t = 0; t < k; ++t){
                    inner_product += X_dense.at(row,t) * Y_dense.at(t, col);
                }
#ifdef SDDMM_PARALLEL_CPU_ZERO_FILTER
                p_ind[ind] = inner_product * val;
#else
                res.values[ind] = inner_product * val;
#endif               
            }

#ifdef SDDMM_PARALLEL_CPU_ZERO_FILTER
            // cost: about +30%
            // filter zero values
            auto s = p_ind.size();
            for(Types::vec_size_t i=0; i<s; ++i){
                if(p_ind[i] != 0){
                    res.values.push_back(p_ind[i]);
                    res.cols.push_back(A_sparse.cols[i]);
                    res.rows.push_back(A_sparse.rows[i]);
                }
            }
#endif

            auto end = std::chrono::high_resolution_clock::now();

            if(measurements != nullptr){
                Types::time_duration_unit duration = std::chrono::duration_cast<Types::time_measure_unit>(end - start).count();
                measurements->durations.push_back(duration);
            }

            return res;
        }

        /**
         * @brief Calculate sddmm, trying to be efficient
        */
        Types::COO parallel_sddmm(
            const Types::COO& A_sparse, 
            const Types::Matrix& X_dense, 
            const Types::Matrix& Y_dense, 
            Types::vec_size_t num_threads,
            Results::ExperimentData* measurements = nullptr
        ){  
            assert(X_dense.is_row_major() && Y_dense.is_col_major() && "X_dense must be row major, Y_dense must be col major!");

            Types::vec_size_t xk = X_dense.m;
            Types::vec_size_t yk = Y_dense.m;
            Types::vec_size_t nnz = A_sparse.values.size();

            Types::COO res;
            res.n = A_sparse.n;
            res.m = A_sparse.m;
#ifdef SDDMM_PARALLEL_CPU_ZERO_FILTER
            std::vector<Types::expmt_t> p_ind(nnz, 0.0);
            // not reserving space cost: about +14%
            res.values.reserve(nnz);
            res.cols.reserve(nnz);
            res.rows.reserve(nnz);
#else
            // not reserving space cost: about +14%
            res.values.resize(nnz, 0.0);
            res.rows.resize(nnz); 
            res.cols.resize(nnz);
            std::copy(A_sparse.cols.begin(), A_sparse.cols.end(), res.cols.begin());
            std::copy(A_sparse.rows.begin(), A_sparse.rows.end(), res.rows.begin());
#endif

            auto start = std::chrono::high_resolution_clock::now();

            // omp_set_num_threads(28);
            #pragma omp parallel for
            for (int64_t ind = 0; ind < nnz; ind++){
                Types::vec_size_t row = A_sparse.rows[ind];
                Types::vec_size_t col = A_sparse.cols[ind]; 
                Types::expmt_t val = A_sparse.values[ind];
                Types::expmt_t inner_product = 0;
                for (Types::vec_size_t t = 0; t < xk; ++t){
                    // inner_product += X_dense.data[row * xk + t] * Y_dense.data[t * yk + col];
                    inner_product += X_dense.at(row,t) * Y_dense.at(t, col);
                }
#ifdef SDDMM_PARALLEL_CPU_ZERO_FILTER
                p_ind[ind] = inner_product * val;
#else
                res.values[ind] = inner_product * val;
#endif                    
            }

#ifdef SDDMM_PARALLEL_CPU_ZERO_FILTER
            // cost: about +30%
            // filter zero values
            auto s = p_ind.size();
            for(Types::vec_size_t i=0; i<s; ++i){
                if(p_ind[i] != 0){
                    res.values.push_back(p_ind[i]);
                    res.cols.push_back(A_sparse.cols[i]);
                    res.rows.push_back(A_sparse.rows[i]);
                }
            }
#endif
            auto end = std::chrono::high_resolution_clock::now();

            if (measurements != nullptr) {
                Types::time_duration_unit duration = std::chrono::duration_cast<Types::time_measure_unit>(end - start).count();
                measurements->durations.push_back(duration);
            }

            return res;
        }
    }
}