
#include "../defines.h"
#include <vector>
#include "../data_structures/matrix/matrix.h"
#include "../data_structures/csr/csr.h"

namespace SDDMM {
    namespace Algo {
        Types::CSR NaiveSDDMM(const Types::CSR& A_sparse, const Types::Matrix& X_dense, const Types::Matrix& Y_dense) {
            assert(X_dense.m == Y_dense.n && "Size of cols(X_dense) and rows(Y) must match!");
            assert(A_sparse.n>0 && A_sparse.m>0 && X_dense.n>0 && X_dense.m>0 && Y_dense.n>0 && Y_dense.m && "All involved matrices must be non-empty!");
            assert(A_sparse.n==X_dense.n && A_sparse.m==Y_dense.m && "Matrix dimensions must match!");

            Types::CSR res;
            res.n = A_sparse.n;
            res.m = A_sparse.m;
            // reserve space
            res.values.reserve(A_sparse.values.size());
            res.col_idx.reserve(A_sparse.col_idx.size());
            res.row_ptr.reserve(A_sparse.row_ptr.size());
            std::copy(A_sparse.values.begin(), A_sparse.values.end(), std::back_inserter(res.values));
            std::copy(A_sparse.col_idx.begin(), A_sparse.col_idx.end(), std::back_inserter(res.col_idx));
            std::copy(A_sparse.row_ptr.begin(), A_sparse.row_ptr.end(), std::back_inserter(res.row_ptr));

            SDDMM::Types::vec_size_t o_col = 0;
            SDDMM::Types::vec_size_t o_row = 0;
            SDDMM::Types::vec_size_t s = A_sparse.row_ptr.size()-1;
            SDDMM::Types::vec_size_t v_ind = 0;
            for(SDDMM::Types::vec_size_t r=0; r<s; ++r){
                SDDMM::Types::vec_size_t from = A_sparse.row_ptr[r];
                SDDMM::Types::vec_size_t to = A_sparse.row_ptr[r+1];
                for(SDDMM::Types::vec_size_t ci=from; ci<to; ++ci){
                    SDDMM::Types::vec_size_t c = A_sparse.col_idx[ci];
                    
                    SDDMM::Types::expmt_t inner_product = 0;
                    for(SDDMM::Types::vec_size_t ind=0; ind < X_dense.m; ++ind){
                        inner_product += X_dense.at(r, ind)*Y_dense.at(ind, c);
                    }
                    
                    res.values[v_ind] *= inner_product;
                    v_ind++;
                }
            }

            return res;
        }
    }
}