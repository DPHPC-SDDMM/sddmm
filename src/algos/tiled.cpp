#pragma once

#include "algo.h"
#include <vector>


namespace SDDMM {
    typedef std::vector<std::vector<double> > Matrix;

    class CSR {
    public:
        std::vector<double> values;
        std::vector<int> col_idx;
        std::vector<int> row_ptr;
    };

    class TiledAlgo : public Algo {
    public:
        int main(int argc, char **argv, const InitParams &initParams) override {
            return 0;
        }

        void run(const CSR& S, const Matrix& A, const Matrix& B, CSR& P, int Ti, int Tj, int Tk, int N, int M, int K) {
            // initialise output matrix P
            P.col_idx = S.col_idx;
            P.row_ptr = S.row_ptr;
            P.values.resize(S.values.size(), 0.);

            // starting row idx of the current *tile*
            for (int ii = 0; ii < N; ii += Ti) {

                // starting column idx of the current *tile*
                for (int jj = 0; jj < M; jj += Tj) {

                    // streaming dimension
                    for (int kk = 0; kk < K; kk += Tk) {

                        // iterate over all rows of the tile
                        for (int i = ii; i < std::min(ii + Ti, N); i++) {

                            // iterate over all elements in the row (i.e. all columns)
                            for (int ptr = S.row_ptr[i]; ptr < S.row_ptr[i+1]; ptr++) {

                                // get current column index
                                int j = S.col_idx[ptr];

                                // check if column index j is within the current tile
                                if (j >= jj && j < jj + Tj) {

                                    // compute dot product
                                    for (int k = kk; k < std::min(kk + Tk, K); ++k) {
                                        P.values[ptr] += A[i][k] * B[j][k];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // scale
            for (int i = 0; i < S.values.size(); i++) {
                P.values[i] *= S.values[i];
            }
        }
    };
}