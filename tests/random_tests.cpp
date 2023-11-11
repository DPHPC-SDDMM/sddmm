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

UTEST(Random, Vanilla_Mult_Use_Loc_Var_Unroll) {
    // generate data
    using namespace SDDMM;
    
    SDDMM::Results::SerialExperimentInfo info(
        "t1",
        10,30,20,
        20,30,20,10
    );

    auto X = Types::Matrix::generate(info.x_num_row, info.xy_num_inner, 0.0);
    auto Y = Types::Matrix::generate(info.xy_num_inner, info.y_num_col, 0.0);

    // auto X = SDDMM::Types::Matrix::deterministic_gen(2, 3, {1,2,3,4,5,6});
    // auto Y = SDDMM::Types::Matrix::deterministic_gen(3, 2, {1,2,3,4,5,6});
    // auto matrix3 = SDDMM::Types::Matrix::deterministic_gen(2,2, {22, 28, 49, 64});
    auto res = X*Y;

    Types::Matrix r1(info.x_num_row, info.y_num_col);
    Types::vec_size_t ni = 0;
    const Types::vec_size_t j = 8;
    const Types::vec_size_t s = info.xy_num_inner-j;
    for(auto r=0; r<info.x_num_row; ++r){
        for(auto c=0; c<info.y_num_col; ++c){
            // precalculate the access index for the target
            Types::vec_size_t xyi = ni+c;
            Types::expmt_t var = 0;
            Types::vec_size_t i;
            for(i=0; i<s; i+=j){
                var += X.data[r*info.xy_num_inner + i+0] * Y.data[(i+0)*info.y_num_col + c];
                var += X.data[r*info.xy_num_inner + i+1] * Y.data[(i+1)*info.y_num_col + c];
                var += X.data[r*info.xy_num_inner + i+2] * Y.data[(i+2)*info.y_num_col + c];
                var += X.data[r*info.xy_num_inner + i+3] * Y.data[(i+3)*info.y_num_col + c];
                var += X.data[r*info.xy_num_inner + i+4] * Y.data[(i+4)*info.y_num_col + c];
                var += X.data[r*info.xy_num_inner + i+5] * Y.data[(i+5)*info.y_num_col + c];
                var += X.data[r*info.xy_num_inner + i+6] * Y.data[(i+6)*info.y_num_col + c];
                var += X.data[r*info.xy_num_inner + i+7] * Y.data[(i+7)*info.y_num_col + c];
            }
            while(i<info.xy_num_inner){
                var += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                i++;
            }
            r1.data[xyi] = var;
        }
        ni += info.y_num_col;
    }

    ASSERT_TRUE(r1 == res);
}

UTEST(Random, Vanilla_Mult_Use_Loc_Var_Unroll_Reassoc) {
    // generate data
    using namespace SDDMM;
    
    SDDMM::Results::SerialExperimentInfo info(
        "t1",
        10,30,20,
        2,4,2,10
    );

    // auto X = Types::Matrix::generate(info.x_num_row, info.xy_num_inner, 0.0);
    // auto Y = Types::Matrix::generate(info.xy_num_inner, info.y_num_col, 0.0);

    auto X = SDDMM::Types::Matrix::deterministic_gen(2, 4, {1,2,3,4,5,6,7,8});
    auto Y = SDDMM::Types::Matrix::deterministic_gen(4, 2, {1,2,3,4,5,6,7,8});
    // auto matrix3 = SDDMM::Types::Matrix::deterministic_gen(2,2, {50, 60, 114, 140});
    auto res = X*Y;

    Types::Matrix r1(info.x_num_row, info.y_num_col);
    Types::vec_size_t ni = 0;
    const Types::vec_size_t j = 4;
    const Types::vec_size_t s = info.xy_num_inner-j;
    for(auto r=0; r<info.x_num_row; ++r){
        for(auto c=0; c<info.y_num_col; ++c){
            // precalculate the access index for the target
            Types::vec_size_t xyi = ni+c;
            Types::expmt_t var = 0;
            Types::vec_size_t i;
            for(i=0; i<s; i+=j){
                var += X.data[r*info.xy_num_inner + i+0] * Y.data[(i+0)*info.y_num_col + c]
                       + (X.data[r*info.xy_num_inner + i+1] * Y.data[(i+1)*info.y_num_col + c]
                          + (X.data[r*info.xy_num_inner + i+2] * Y.data[(i+2)*info.y_num_col + c]
                             + (X.data[r*info.xy_num_inner + i+3] * Y.data[(i+3)*info.y_num_col + c])));
                // var += X.data[r*info.xy_num_inner + i+4] * Y.data[(i+4)*info.y_num_col + c];
                // var += X.data[r*info.xy_num_inner + i+5] * Y.data[(i+5)*info.y_num_col + c];
                // var += X.data[r*info.xy_num_inner + i+6] * Y.data[(i+6)*info.y_num_col + c];
                // var += X.data[r*info.xy_num_inner + i+7] * Y.data[(i+7)*info.y_num_col + c];
            }
            while(i<info.xy_num_inner){
                var += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                i++;
            }
            r1.data[xyi] = var;
        }
        ni += info.y_num_col;
    }

    ASSERT_TRUE(r1 == res);
}

UTEST(Random, Vanilla_Mult_Use_Loc_Var_Unroll_sep_acc) {
    // generate data
    using namespace SDDMM;
    
    SDDMM::Results::SerialExperimentInfo info(
        "t1",
        10,30,20,
        20,30,20,10
    );

    auto X = Types::Matrix::generate(info.x_num_row, info.xy_num_inner, 0.0);
    auto Y = Types::Matrix::generate(info.xy_num_inner, info.y_num_col, 0.0);

    // auto X = SDDMM::Types::Matrix::deterministic_gen(2, 3, {1,2,3,4,5,6});
    // auto Y = SDDMM::Types::Matrix::deterministic_gen(3, 2, {1,2,3,4,5,6});
    // auto matrix3 = SDDMM::Types::Matrix::deterministic_gen(2,2, {22, 28, 49, 64});
    auto res = X*Y;

    Types::Matrix r1(info.x_num_row, info.y_num_col);
    Types::vec_size_t ni = 0;
    const Types::vec_size_t j = 4;
    const Types::vec_size_t s = info.xy_num_inner-j;
    for(auto r=0; r<info.x_num_row; ++r){
        for(auto c=0; c<info.y_num_col; ++c){
            // precalculate the access index for the target
            Types::vec_size_t xyi = ni+c;
            Types::expmt_t var_1 = 0;
            Types::expmt_t var_2 = 0;
            Types::expmt_t var_3 = 0;
            Types::expmt_t var_4 = 0;
            Types::vec_size_t i;
            for(i=0; i<s; i+=j){
                var_1 += X.data[r*info.xy_num_inner + i+0] * Y.data[(i+0)*info.y_num_col + c];
                var_2 += X.data[r*info.xy_num_inner + i+1] * Y.data[(i+1)*info.y_num_col + c];
                var_3 += X.data[r*info.xy_num_inner + i+2] * Y.data[(i+2)*info.y_num_col + c];
                var_4 += X.data[r*info.xy_num_inner + i+3] * Y.data[(i+3)*info.y_num_col + c];
            }
            while(i<info.xy_num_inner){
                var_1 += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                i++;
            }
            r1.data[xyi] = var_1 + var_2 + var_3 + var_4;
        }
        ni += info.y_num_col;
    }

    ASSERT_TRUE(r1 == res);
}