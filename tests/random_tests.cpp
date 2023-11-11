#include <iostream>
#include <cstdio>
#include <vector>
#include <math.h>
#include "test_helpers.hpp"
#include "../src/data_structures/matrix/matrix.h"
#include "../src/data_structures/csr/csr.h"
#include "../src/data_structures/coo/coo.h"
#include "../src/algos/cpu_sddmm/naive_sddmm.cpp"
#include "../src/algos/cpu_sddmm/tiled_sddmm.cpp"
#include "../src/algos/cpu_sddmm/parallel_sddmm.cpp"

UTEST_MAIN();

UTEST(Random, Vanilla_Mult) {
    // generate data
    using namespace SDDMM;
    
    SDDMM::Results::SerialExperimentInfo info(
        "t1",
        2,3,2,
        2,3,2,10
    );

    auto X = SDDMM::Types::Matrix::deterministic_gen(2, 3, {1,2,3,4,5,6});
    auto Y = SDDMM::Types::Matrix::deterministic_gen(3, 2, {1,2,3,4,5,6});
    auto matrix3 = SDDMM::Types::Matrix::deterministic_gen(2,2, {22, 28, 49, 64});
    auto res = X*Y;
    ASSERT_TRUE(matrix3 == res);

    Types::Matrix r1(info.x_num_row, info.y_num_col);
    for(auto r=0; r<info.x_num_row; ++r){
        for(auto c=0; c<info.y_num_col; ++c){
            for(auto i=0; i<info.xy_num_inner; ++i){
                r1.data[r*info.y_num_col + c] += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
            }
        }
    }

    ASSERT_TRUE(r1 == res);
}

UTEST(Random, Vanilla_Precalc_Mult) {
    // generate data
    using namespace SDDMM;
    
    SDDMM::Results::SerialExperimentInfo info(
        "t1",
        2,3,2,
        2,3,2,10
    );

    auto X = SDDMM::Types::Matrix::deterministic_gen(2, 3, {1,2,3,4,5,6});
    auto Y = SDDMM::Types::Matrix::deterministic_gen(3, 2, {1,2,3,4,5,6});
    auto matrix3 = SDDMM::Types::Matrix::deterministic_gen(2,2, {22, 28, 49, 64});
    auto res = X*Y;
    ASSERT_TRUE(matrix3 == res);

    Types::Matrix r1(info.x_num_row, info.y_num_col);
    for(auto r=0; r<info.x_num_row; ++r){
        for(auto c=0; c<info.y_num_col; ++c){
            Types::vec_size_t xyi = r*info.y_num_col + c;
            for(auto i=0; i<info.xy_num_inner; ++i){
                r1.data[xyi] += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
            }
        }
    }

    ASSERT_TRUE(r1 == res);
}

UTEST(Random, Vanilla_Mult_Use_Add) {
    // generate data
    using namespace SDDMM;
    
    SDDMM::Results::SerialExperimentInfo info(
        "t1",
        2,3,2,
        2,3,2,10
    );

    auto X = SDDMM::Types::Matrix::deterministic_gen(2, 3, {1,2,3,4,5,6});
    auto Y = SDDMM::Types::Matrix::deterministic_gen(3, 2, {1,2,3,4,5,6});
    auto matrix3 = SDDMM::Types::Matrix::deterministic_gen(2,2, {22, 28, 49, 64});
    auto res = X*Y;
    ASSERT_TRUE(matrix3 == res);

    Types::Matrix r1(info.x_num_row, info.y_num_col);
    Types::vec_size_t ni = 0;
    for(auto r=0; r<info.x_num_row; ++r){
        for(auto c=0; c<info.y_num_col; ++c){
            Types::vec_size_t xyi = ni+c;
            for(auto i=0; i<info.xy_num_inner; ++i){
                r1.data[xyi] += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
            }
        }
        ni += info.y_num_col;
    }

    ASSERT_TRUE(r1 == res);
}

UTEST(Random, Vanilla_Mult_Use_Loc_Var) {
    // generate data
    using namespace SDDMM;
    
    SDDMM::Results::SerialExperimentInfo info(
        "t1",
        2,3,2,
        2,3,2,10
    );

    auto X = SDDMM::Types::Matrix::deterministic_gen(2, 3, {1,2,3,4,5,6});
    auto Y = SDDMM::Types::Matrix::deterministic_gen(3, 2, {1,2,3,4,5,6});
    auto matrix3 = SDDMM::Types::Matrix::deterministic_gen(2,2, {22, 28, 49, 64});
    auto res = X*Y;
    ASSERT_TRUE(matrix3 == res);

    Types::Matrix r1(info.x_num_row, info.y_num_col);
    for(auto r=0; r<info.x_num_row; ++r){
        for(auto c=0; c<info.y_num_col; ++c){
            Types::vec_size_t xyi = r*info.y_num_col + c;
            Types::expmt_t val = 0;
            for(auto i=0; i<info.xy_num_inner; ++i){
                val += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
            }
            r1.data[xyi] = val;
        }
    }

    ASSERT_TRUE(r1 == res);
}