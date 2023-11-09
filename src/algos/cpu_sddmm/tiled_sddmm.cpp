
#include <vector>
#include <chrono>
#include "../../defines.h"
#include "../../data_structures/matrix/matrix.h"
#include "../../data_structures/csr/csr.h"

namespace SDDMM {
    namespace Algo {
        Types::CSR tiled_sddmm(
            const Types::CSR& S, 
            const Types::Matrix& A, 
            const Types::Matrix& B, 
            int Ti, int Tj, int Tk,
            Results::ExperimentData* measurements = nullptr
        ) {
            
            auto start = std::chrono::high_resolution_clock::now();

            // initialise output matrix P
            Types::CSR P;
            P.n = S.n;
            P.m = S.m;

            std::vector<Types::expmt_t> intermediate;
            intermediate.resize(S.values.size(), 0.0);

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
                                        // P.values[ptr] += A.at(i,k) * B.at(k,j);
                                        intermediate[ptr] += A.at(i,k) * B.at(k,j);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // scale
            Types::vec_size_t s = S.values.size();
            for (Types::vec_size_t i = 0; i < s; i++) {
                auto val = intermediate[i];
                if(val != 0){
                    P.values.push_back(S.values[i] * val);   
                }
            }

            Types::vec_size_t ind = 0;
            Types::vec_size_t new_ci = 0;
            Types::vec_size_t rs = S.row_ptr.size()-1;
            P.row_ptr.push_back(new_ci);
            for(Types::vec_size_t r=0; r<rs; ++r){
                auto from = S.row_ptr[r];
                auto to = S.row_ptr[r+1];
                for(Types::vec_size_t ci=from; ci<to; ++ci){
                    Types::vec_size_t c = S.col_idx[ci];
                    if(intermediate[ind] != 0){
                        P.col_idx.push_back(c);
                        new_ci++;
                    }
                    ind++;
                }
                P.row_ptr.push_back(new_ci);
            }

            auto end = std::chrono::high_resolution_clock::now();

            if(measurements != nullptr){
                Types::time_duration_unit duration = std::chrono::duration_cast<Types::time_measure_unit>(end - start).count();
                measurements->durations.push_back(duration);
            }

            return P;
        }
    }
}