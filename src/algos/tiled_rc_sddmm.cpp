
#include "../defines.h"
#include <vector>
#include "../data_structures/matrix/matrix.h"
#include "../data_structures/csr/csr.h"

namespace SDDMM {
    namespace Algo {
        Types::CSR Tiled_RC_SDDMM(
            const Types::CSR& S, 
            const Types::Matrix& A, 
            const Types::Matrix& B, 
            int Ta, int Tb, int Ti
        ) {
            assert(B.format() == Types::MatrixFormat::ColMajor 
                && A.format() == Types::MatrixFormat::RowMajor
                && "A must be in RowMajor and B in ColMajor format!");

            // // initialise output matrix P
            // Types::CSR P;
            // P.n = S.n;
            // P.m = S.m;
            // P.col_idx = S.col_idx;
            // P.row_ptr = S.row_ptr;
            // P.values.resize(S.values.size(), 0.);

            // Types::vec_size_t N = S.n;
            // Types::vec_size_t M = S.m;
            // Types::vec_size_t K = A.m; // == B.n ==> inner dimension of dense matrices
            
            // Types::CSR res;
            // res.n = A_sparse.n;
            // res.m = A_sparse.m;
            // // reserve space
            // // res.values.reserve(A_sparse.values.size());
            // res.col_idx.reserve(A_sparse.col_idx.size());
            // res.row_ptr.reserve(A_sparse.row_ptr.size());

            // res.values.resize(A_sparse.values.size(), 0.0);
            // // std::copy(A_sparse.values.begin(), A_sparse.values.end(), std::back_inserter(res.values));
            // std::copy(A_sparse.col_idx.begin(), A_sparse.col_idx.end(), std::back_inserter(res.col_idx));
            // std::copy(A_sparse.row_ptr.begin(), A_sparse.row_ptr.end(), std::back_inserter(res.row_ptr));

            // SDDMM::Types::vec_size_t o_col = 0;
            // SDDMM::Types::vec_size_t o_row = 0;
            // SDDMM::Types::vec_size_t s = A_sparse.row_ptr.size()-1;
            // SDDMM::Types::vec_size_t v_ind = 0;
            // for(SDDMM::Types::vec_size_t r=0; r<s; ++r){
            //     SDDMM::Types::vec_size_t from = A_sparse.row_ptr[r];
            //     SDDMM::Types::vec_size_t to = A_sparse.row_ptr[r+1];
            //     for(SDDMM::Types::vec_size_t ci=from; ci<to; ++ci){
            //         SDDMM::Types::vec_size_t c = A_sparse.col_idx[ci];
                    
            //         SDDMM::Types::expmt_t inner_product = 0;
            //         for(SDDMM::Types::vec_size_t ind=0; ind < X_dense.m; ++ind){
            //             inner_product += X_dense.at(r, ind)*Y_dense.at(ind, c);
            //         }
                    
            //         res.values[v_ind] *= inner_product;
            //         v_ind++;
            //     }
            // }

            // return res;

            // // starting row idx of the current *tile*
            // for (Types::vec_size_t ii = 0; ii < N; ii += Ti) {

            //     // starting column idx of the current *tile*
            //     for (Types::vec_size_t jj = 0; jj < M; jj += Tj) {

            //         // streaming dimension
            //         for (Types::vec_size_t kk = 0; kk < K; kk += Tk) {

            //             // iterate over all rows of the tile
            //             for (Types::vec_size_t i = ii; i < std::min(ii + Ti, N); i++) {

            //                 // iterate over all elements in the row (i.e. all columns)
            //                 for (Types::vec_size_t ptr = S.row_ptr[i]; ptr < S.row_ptr[i+1]; ptr++) {

            //                     // get current column index
            //                     Types::vec_size_t j = S.col_idx[ptr];

            //                     // check if column index j is within the current tile
            //                     if (j >= jj && j < jj + Tj) {

            //                         // compute dot product
            //                         for (Types::vec_size_t k = kk; k < std::min(kk + Tk, K); ++k) {
            //                             P.values[ptr] += A.at(i,k) * B.at(k,j);
            //                         }
            //                     }
            //                 }
            //             }
            //         }
            //     }
            // }

            // scale
            // for (Types::vec_size_t i = 0; i < S.values.size(); i++) {
            //     P.values[i] *= S.values[i];
            // }

            // return P;
        }
    }
}