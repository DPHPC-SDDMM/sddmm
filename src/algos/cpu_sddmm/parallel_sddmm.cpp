#include <vector>
#include <omp.h>
#include <iostream>
#include <math.h>
#include "../../defines.h"
#include "../../results.h"
#include "../../data_structures/matrix/matrix.h"
#include "../../data_structures/coo/coo.h"

namespace SDDMM {
    namespace Algo {
        /**
         * @brief Computes the Sampled Dense-Dense Matrix Multiplication (SDDMM) operation
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
        Types::COO parallel_sddmm(
            const Types::COO& A_sparse, 
            const Types::Matrix& X_dense, 
            const Types::Matrix& Y_dense, 
            Types::vec_size_t num_threads,
            Results::ExperimentData* measurements = nullptr
        ) {
            assert(X_dense.m == Y_dense.n && "Size of cols(X_dense) and rows(Y) must match!");
            assert(A_sparse.n>0 && A_sparse.m>0 && X_dense.n>0 && X_dense.m>0 && Y_dense.n>0 && Y_dense.m>0 && "All involved matrices must be non-empty!");
            assert(A_sparse.n==X_dense.n && A_sparse.m==Y_dense.m && "Matrix dimensions must match!");

            auto start = std::chrono::high_resolution_clock::now();

            Types::COO res;
            res.n = A_sparse.n;
            res.m = A_sparse.m;

            auto s = A_sparse.data.size();
            for(Types::vec_size_t i=0; i<s; i+=num_threads){
                // auto m = std::min(s - i, num_threads); // This line is commented out to resemble the *constant* number of threads when calling CUDA.

                // Subset of COO entries.
                // The term "block" was chosen to be reminiscent of a CUDA blocks, i.e. a set of CUDA threads.
                std::vector<Types::COO::triplet> block(num_threads, {0,0,0});
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
                        Types::COO::triplet p = A_sparse.data.at(idx);
                        Types::expmt_t inner_product = 0;
                        
                        // the ind index has to be tiled later
                        for(SDDMM::Types::vec_size_t ind=0; ind < X_dense.m; ++ind){
                            inner_product += X_dense.at(p.row, ind)*Y_dense.at(ind, p.col);
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
                        block[tn] = {p.row, p.col, p.value * inner_product};
                    }
                }
                for(Types::COO::triplet& elem : block){
                    if(elem.value != 0){ // Comply with the definition of a COO matrix (i.e. hold only non-zero values).
                        res.data.push_back(elem);
                    }
                }
            }

            auto end = std::chrono::high_resolution_clock::now();

            if(measurements != nullptr){
                Types::time_duration_unit duration = std::chrono::duration_cast<Types::time_measure_unit>(end - start).count();
                measurements->durations.push_back(duration);
            }

            // Shrink the size of the data structures in case zero-valued inner products appeared,
            // thus requiring less than initial space predicted (i.e. memory amount equal to the input sparse matrix ).
            res.data.shrink_to_fit();

            return res;
        }
    }
}