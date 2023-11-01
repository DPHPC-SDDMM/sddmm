
#include "../defines.h"
#include <vector>
#include "../data_structures/matrix/matrix.h"
#include "../data_structures/csr/csr.h"

namespace SDDMM {
    namespace Algo {
        Types::CSR TiledSDDMM(
            const Types::CSR& S, 
            const Types::Matrix& A, 
            const Types::Matrix& B, 
            int Ti, int Tj, int Tk
        ) {
            
            // initialise output matrix P
            Types::CSR P;
            P.n = S.n;
            P.m = S.m;
            P.col_idx = S.col_idx;
            P.row_ptr = S.row_ptr;
            P.values.resize(S.values.size(), 0.);

            Types::vec_size_t N = S.n;
            Types::vec_size_t M = S.m;
            Types::vec_size_t K = A.m; // == B.n ==> inner dimension of dense matrices
            
            // starting row idx of the current *tile*
            for (Types::vec_size_t ii = 0; ii < N; ii += Ti) {

                // starting column idx of the current *tile*
                for (Types::vec_size_t jj = 0; jj < M; jj += Tj) {

                    // streaming dimension
                    for (Types::vec_size_t kk = 0; kk < K; kk += Tk) {

                        // iterate over all rows of the tile
                        for (Types::vec_size_t i = ii; i < std::min(ii + Ti, N); i++) {

                            // iterate over all elements in the row (i.e. all columns)
                            for (Types::vec_size_t ptr = S.row_ptr[i]; ptr < S.row_ptr[i+1]; ptr++) {

                                // get current column index
                                Types::vec_size_t j = S.col_idx[ptr];

                                // check if column index j is within the current tile
                                if (j >= jj && j < jj + Tj) {

                                    // compute dot product
                                    for (Types::vec_size_t k = kk; k < std::min(kk + Tk, K); ++k) {
                                        P.values[ptr] += A.at(i,k) * B.at(k,j);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // scale
            for (Types::vec_size_t i = 0; i < S.values.size(); i++) {
                P.values[i] *= S.values[i];
            }

            return P;
        }
    }
}