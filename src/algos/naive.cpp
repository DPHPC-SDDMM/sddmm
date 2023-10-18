#pragma once

#include "algo.h"
#include "coo.h"
#include <cstddef>
#include <iostream>
#include <vector>

namespace SDDMM {

class NaiveAlgo : public Algo {
  public:
    int main(int argc, char **argv, const InitParams &initParams) override {
        return 0;
    }

    // TODO: Maybe change the type to be more generic?
    // TODO: Maybe add this to algo.h::Algo signature?

    // Implements Algorithm 2 from "Sampled Dense Matrix Multiplication for
    // High-Performance Machine Learning"
    void run(COO &S, float *A, float *B, COO &C, size_t M, size_t N, size_t K) {
        // Initializing
        C.column_indexes = S.column_indexes;
        C.row_indexes = S.row_indexes;
        C.values.clear();

        // Dot product
        for (size_t i = 0; i < M; i++) {
            for (size_t j = S.row_indexes[i]; j < S.row_indexes[i + 1]; j++) {
                C.values.push_back(0);
                for (size_t k = 0; k < K; k++) {
                    C.values[j] +=
                        A[i * K + k] * B[S.column_indexes[j] * K + k];
                }
            }
        }

        // Scaling
        for (size_t i = 0; i < S.row_indexes.size(); i++) {
            for (size_t j = S.row_indexes[i]; j < S.row_indexes[i + 1]; j++) {
                C.values[j] *= S.values[j];
            }
        }
    }
};

} // namespace SDDMM