#include <iostream>
#include <cstdio>
#include <vector>
#include <math.h>
#ifdef __AVX2__
    // needs add_compile_options(-mavx2)
    #include <immintrin.h>
#endif

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
    const Types::vec_size_t s = info.xy_num_inner-j/2;
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
    const Types::vec_size_t s = info.xy_num_inner-j/2;
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

#ifdef __AVX2__
// UTEST(Random, Vec_instr_1){
//     __m256 a = _mm256_set_ps(8.0, 7.0, 6.0, 5.0, 
//                              4.0, 3.0, 2.0, 1.0);
//     __m256 b = _mm256_set_ps(18.0, 17.0, 16.0, 15.0, 
//                              14.0, 13.0, 12.0, 11.0);

//     __m256 c = _mm256_add_ps(a, b);

//     float d[8];
//     _mm256_storeu_ps(d, c);

//     std::cout << "result equals " << d[0] << "," << d[1]
//               << "," << d[2] << "," << d[3] << ","
//               << d[4] << "," << d[5] << "," << d[6] << ","
//               << d[7] << std::endl;
// }

UTEST(Random, Vanilla_Mult_Test){
    using namespace SDDMM;

    std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> Y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f};
    std::vector<float> res(8, 0.0);

    __m256 var = _mm256_set1_ps(0.0f);
    __m256 x_vec = _mm256_set_ps(X[7], X[6], X[5], X[4], X[3], X[2], X[1], X[0]);
    __m256 y_vec = _mm256_set_ps(Y[7], Y[6], Y[5], Y[4], Y[3], Y[2], Y[1], Y[0]);

    var = _mm256_add_ps(var, _mm256_mul_ps(x_vec, y_vec));
    var = _mm256_add_ps(var, _mm256_mul_ps(x_vec, y_vec));

    _mm256_storeu_ps(&res[0], var);

    std::vector<float> expected = {4, 16, 36, 64, 100, 144, 196, 256};
    for(auto i=0; i<expected.size(); ++i){
        // std::cout << res[i] << " " << expected[i] << std::endl;
        ASSERT_TRUE(expected[i] == res[i]);
    }
}

UTEST(Random, Vanilla_Mult_Vectorized) {
    // generate data
    using namespace SDDMM;
    
    SDDMM::Results::SerialExperimentInfo info(
        "t1",
        10,30,20,
        5,128,11,10
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
    const Types::vec_size_t st = info.xy_num_inner;
    const Types::vec_size_t s = st;
    Types::expmt_t cache[8];
    for(auto r=0; r<info.x_num_row; ++r){
        for(auto c=0; c<info.y_num_col; ++c){
            // precalculate the access index for the target
            Types::vec_size_t xyi = ni+c;
            Types::vec_size_t i;
            __m256 var = _mm256_setzero_ps();
            for(i=0; i<s; i+=j){
                Types::vec_size_t i7 = i+7;
                Types::vec_size_t i6 = i+6;
                Types::vec_size_t i5 = i+5;
                Types::vec_size_t i4 = i+4;
                Types::vec_size_t i3 = i+3;
                Types::vec_size_t i2 = i+2;
                Types::vec_size_t i1 = i+1;
                Types::vec_size_t i0 = i+0;

                bool st7 = st > i7;
                bool st6 = st > i6;
                bool st5 = st > i5;
                bool st4 = st > i4;
                bool st3 = st > i3;
                bool st2 = st > i2;
                bool st1 = st > i1;
                bool st0 = st > i0;

                __m256 x = _mm256_set_ps(
                    st7 ? X.data[r*info.xy_num_inner + i7] : 0,
                    st6 ? X.data[r*info.xy_num_inner + i6] : 0,
                    st5 ? X.data[r*info.xy_num_inner + i5] : 0,
                    st4 ? X.data[r*info.xy_num_inner + i4] : 0,
                    st3 ? X.data[r*info.xy_num_inner + i3] : 0,
                    st2 ? X.data[r*info.xy_num_inner + i2] : 0,
                    st1 ? X.data[r*info.xy_num_inner + i1] : 0,
                    st0 ? X.data[r*info.xy_num_inner + i0] : 0
                );
                __m256 y = _mm256_set_ps(
                    st7 ? Y.data[i7*info.y_num_col + c] : 0,
                    st6 ? Y.data[i6*info.y_num_col + c] : 0,
                    st5 ? Y.data[i5*info.y_num_col + c] : 0,
                    st4 ? Y.data[i4*info.y_num_col + c] : 0,
                    st3 ? Y.data[i3*info.y_num_col + c] : 0,
                    st2 ? Y.data[i2*info.y_num_col + c] : 0,
                    st1 ? Y.data[i1*info.y_num_col + c] : 0,
                    st0 ? Y.data[i0*info.y_num_col + c] : 0
                );
                var = _mm256_add_ps(var, _mm256_mul_ps(x,y));
            }
            _mm256_storeu_ps(&cache[0], var);
            Types::expmt_t acc = cache[0] + cache[1] + cache[2] + cache[3] + cache[4] + cache[5] + cache[6] + cache[7];
            r1.data[xyi] += acc;
        }
        ni += info.y_num_col;
    }

    ASSERT_TRUE(r1 == res);
}
#endif