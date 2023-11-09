
#include <vector>
#include <chrono>
#include "../defines.h"
#include "../data_structures/matrix/matrix.h"
#include "../data_structures/csr/csr.h"
#include "../data_structures/coo/coo.h"

namespace SDDMM {
    namespace Algo {
        /**
         * @brief Computes the Sampled Dense-Dense Matrix Multiplication (SDDMM) operation
         * __using a single CPU thread (sequential processing)__ with a sparse matrix in the CSR matrix representation format.
         * The SDDMM consists of a Hadamard product (i.e. element-wise multiplication) between
         * - the dense matrix product XY between X_dense and Y_dense and
         * - a sparse matrix
         * 
         * @param A_sparse: A sparse matrix using the CSR matrix representation format.
         * @param X_dense: The left-hand side (LHS) of the dense matrix product.
         * @param Y_dense: The right-hand side (RHS) of the dense matrix product.
         * @param num_threads: The number of threads which will be utilized for performing this operation.
         * @param measurements: Optional variable pointer which stores the time required to perform the operation. The duration time measure unit is defined in @ref "defines.h"
         * @returns (X @ Y) * A_sparse
         * 
         * @remark Potential zero values arising during computations are ignored, so as to comply with the CSR format. 
         * 
         * @warning Dimensionality of matrices are expected to match each operation used, i.e.
         *  1) If X in R^{n x k}, then Y must be in R^{k x m}
         *  2) A_sparse must be in R^{n x k}
         * 
         * @sa
         * - [CSR matrix format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO))
         * - [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
        */
        Types::CSR naive_sddmm(
            const Types::CSR& A_sparse, 
            const Types::Matrix& X_dense, 
            const Types::Matrix& Y_dense,
            Defines::ExperimentData* measurements = nullptr
        ) {
            assert(X_dense.m == Y_dense.n && "Size of cols(X_dense) and rows(Y) must match!");
            assert(A_sparse.n>0 && A_sparse.m>0 && X_dense.n>0 && X_dense.m>0 && Y_dense.n>0 && Y_dense.m && "All involved matrices must be non-empty!");
            assert(A_sparse.n==X_dense.n && A_sparse.m==Y_dense.m && "Matrix dimensions must match!");

            auto start = std::chrono::high_resolution_clock::now();

            Types::CSR res;
            res.n = A_sparse.n;
            res.m = A_sparse.m;
            // reserve space
            res.values.reserve(A_sparse.values.size());
            res.col_idx.reserve(A_sparse.col_idx.size());
            res.row_ptr.reserve(A_sparse.row_ptr.size());
            // std::copy(A_sparse.values.begin(), A_sparse.values.end(), std::back_inserter(res.values));
            // std::copy(A_sparse.col_idx.begin(), A_sparse.col_idx.end(), std::back_inserter(res.col_idx));
            // std::copy(A_sparse.row_ptr.begin(), A_sparse.row_ptr.end(), std::back_inserter(res.row_ptr));

            Types::vec_size_t o_col = 0;
            Types::vec_size_t o_row = 0;
            Types::vec_size_t s = A_sparse.row_ptr.size()-1;
            Types::vec_size_t v_ind = 0;
            
            // init val
            Types::vec_size_t new_ci = 0;
            res.row_ptr.push_back(new_ci);

            // row_ptr
            for(Types::vec_size_t r=0; r<s; ++r){
                
                Types::vec_size_t from = A_sparse.row_ptr[r];
                Types::vec_size_t to = A_sparse.row_ptr[r+1];

                // col_idx
                Types::vec_size_t ci;
                for(ci=from; ci<to; ++ci){
                    
                    Types::expmt_t inner_product = 0;
                    Types::vec_size_t c = A_sparse.col_idx[ci];
                    // Types::expmt_t new_value = values[v_ind] * other.at(r, c);
                    for(Types::vec_size_t ind=0; ind < X_dense.m; ++ind){
                        inner_product += X_dense.at(r, ind)*Y_dense.at(ind, c);
                    }
                    
                    if(inner_product != 0){
                        Types::expmt_t new_value = inner_product * A_sparse.values[v_ind];
                        res.values.push_back(new_value);
                        res.col_idx.push_back(c);
                        new_ci++;
                    }
                    v_ind++;
                }

                res.row_ptr.push_back(new_ci);
            }

            auto end = std::chrono::high_resolution_clock::now();

            if(measurements != nullptr){
                Types::time_duration_unit duration = std::chrono::duration_cast<Types::time_measure_unit>(end - start).count();
                measurements->durations.push_back(duration);
            }

            // Shrink the size of the data structures in case zero-valued inner products appeared,
            // thus requiring less than initial space predicted (i.e. memory amount equal to the input sparse matrix ).
            res.values.shrink_to_fit();
            res.col_idx.shrink_to_fit();
            res.row_ptr.shrink_to_fit();

            return res;
        }

        

        /**
         * @brief Computes the Sampled Dense-Dense Matrix Multiplication (SDDMM) operation
         * __using a single CPU thread (sequential processing)__ with a sparse matrix in the COO matrix representation format.
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
         * - [COO matrix format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
         * - [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
        */
        Types::COO naive_sddmm(
            const Types::COO& A_sparse, 
            const Types::Matrix& X_dense, 
            const Types::Matrix& Y_dense, 
            Defines::ExperimentData* measurements = nullptr
        ) {
            assert(X_dense.m == Y_dense.n && "Size of cols(X_dense) and rows(Y) must match!");
            assert(A_sparse.n>0 && A_sparse.m>0 && X_dense.n>0 && X_dense.m>0 && Y_dense.n>0 && Y_dense.m>0 && "All involved matrices must be non-empty!");
            assert(A_sparse.n==X_dense.n && A_sparse.m==Y_dense.m && "Matrix dimensions must match!");

            auto start = std::chrono::high_resolution_clock::now();

            Types::COO res;
            res.n = A_sparse.n;
            res.m = A_sparse.m;

            auto s = A_sparse.data.size();
            for(Types::vec_size_t i=0; i<s; i++){
                // auto m = std::min(s - i, num_threads); // This line is commented out to resemble the *constant* number of threads when calling CUDA.

                Types::COO::triplet p = A_sparse.data.at(i);
                Types::expmt_t inner_product = 0;
                
                // the ind index has to be tiled later
                for(SDDMM::Types::vec_size_t ind=0; ind < X_dense.m; ++ind){
                    inner_product += X_dense.at(p.row, ind)*Y_dense.at(ind, p.col);
                }

                if(inner_product != 0){ // Comply with the definition of a COO matrix (i.e. hold only non-zero values).
                    res.data.push_back({p.row, p.col, p.value * inner_product});
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