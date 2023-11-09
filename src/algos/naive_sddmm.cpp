
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

            // So, now, we start cheating because we are too lazy to 
            // think of a correct, efficient solution that works without cheating
            // (Hehehehehehe... ^^ Muahahahahahaaaaaaaa XD)


            return res;
        }
    }
}