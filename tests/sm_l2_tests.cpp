#include <iostream>
#include "test_helpers.hpp"
#include "../src/data_structures/coo/coo.h"
#include "../src/data_structures/matrix/matrix.h"
#include "../src/algos/sm-l2-sddmm/sm-l2-sddmm.cpp"
#include "../src/algos/cuda_sddmm/cuda_sddmm.cpp"

UTEST_MAIN();

/**
 * Replicated from random_tests.cpp to make them be in this file too
 * => demo on how the row to col major order and vice versa behaves
*/
UTEST(Random, Row_To_Col_Major) {
    // generate data
    using namespace SDDMM;

    auto X_rm = SDDMM::Types::Matrix::deterministic_gen_row_major(2, 3, {1,2,3,4,5,6});
    auto X_cm = SDDMM::Types::Matrix::deterministic_gen_col_major(2, 3, {1,2,3,4,5,6});

    auto X_cm_to_rm = X_cm.get_dense_row_major();
    auto X_rm_to_cm = X_rm.get_dense_col_major();

    ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::expmt_t>(X_rm.data, {1,2,3,4,5,6}));
    ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::expmt_t>(X_cm_to_rm.data, {1,2,3,4,5,6}));
    ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::expmt_t>(X_cm.data, {1,4,2,5,3,6}));
    ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::expmt_t>(X_rm_to_cm.data, {1,4,2,5,3,6}));
    
    ASSERT_TRUE(X_rm == X_cm);
    ASSERT_TRUE(X_rm == X_rm_to_cm);
    ASSERT_TRUE(X_cm == X_cm_to_rm);
    ASSERT_TRUE(X_rm_to_cm == X_cm_to_rm);

    ASSERT_TRUE(X_rm.n == 2);
    ASSERT_TRUE(X_rm.m == 3);
    ASSERT_TRUE(X_rm_to_cm.n == 2);
    ASSERT_TRUE(X_rm_to_cm.m == 3);

    ASSERT_TRUE(X_cm.n == 2);
    ASSERT_TRUE(X_cm.m == 3);
    ASSERT_TRUE(X_cm_to_rm.n == 2);
    ASSERT_TRUE(X_cm_to_rm.m == 3);
}

/**
 * Replicated from random_tests.cpp to make them be in this file too
 * => demo on how the transpose behaves
*/
UTEST(Random, Transposed) {
    // generate data
    using namespace SDDMM;

    auto X = SDDMM::Types::Matrix::deterministic_gen_row_major(2, 3, {1,2,3,4,5,6});
    auto XT = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 2, {1,4,2,5,3,6});

    auto X_T = X.get_transposed();
    auto XTT = XT.get_transposed();

    ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::expmt_t>(X.data, {1,2,3,4,5,6}));
    ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::expmt_t>(XTT.data, {1,2,3,4,5,6}));
    ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::expmt_t>(XT.data, {1,4,2,5,3,6}));
    ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::expmt_t>(X_T.data, {1,4,2,5,3,6}));
    
    ASSERT_TRUE(X == XTT);
    ASSERT_TRUE(X_T == XT);

    ASSERT_TRUE(X.n == 2);
    ASSERT_TRUE(X.m == 3);
    ASSERT_TRUE(XTT.n == 2);
    ASSERT_TRUE(XTT.m == 3);

    ASSERT_TRUE(X_T.n == 3);
    ASSERT_TRUE(X_T.m == 2);
    ASSERT_TRUE(XT.n == 3);
    ASSERT_TRUE(XT.m == 2);

    ASSERT_TRUE(X.n == X_T.m);
    ASSERT_TRUE(X.m == X_T.n);
    ASSERT_TRUE(XTT.n == XT.m);
    ASSERT_TRUE(XTT.m == XT.n);
}

UTEST(SM_L2, Init_test) {
    {
        float sparsity = 0.5; // 0.97;
        SDDMM::Types::vec_size_t N = 32; //1024;
        SDDMM::Types::vec_size_t M = 32; //256;
        SDDMM::Types::vec_size_t K = 32; //256;
        // SDDMM::Types::COO S = SDDMM::Types::Matrix::generate_row_major(N, M, sparsity).to_coo();
        SDDMM::Types::COO S = SDDMM::Types::Matrix::deterministic_gen_row_major(N, M, 
            {
                1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,
                0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,
                1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,
                0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,
                1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,
                0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,
                1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,
                0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,

                1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,
                0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,
                1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,
                0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,
                1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,
                0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,
                1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,
                0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,

                1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,
                0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,
                1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,
                0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,
                1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,
                0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,
                1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,
                0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,

                1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,
                0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,
                1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,
                0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,
                1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,
                0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,
                1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,   1,0,1,0,1,0,1,0,
                0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1,   0,1,0,1,0,1,0,1
            }
        ).to_coo();
        SDDMM::Types::Matrix A = SDDMM::Types::Matrix::generate_row_major(N, K, 0);
        for(int i=0; i<A.data.size(); ++i){
            if(i%2==0 && i>0)
                A.data[i] = 1;
            else
                A.data[i] = 2;
        }
        SDDMM::Types::Matrix B = SDDMM::Types::Matrix::generate_col_major(K, M, 0);
        for(int i=0; i<B.data.size(); ++i){
            if(i%2==0 && i>0)
                B.data[i] = 2;
            else
                B.data[i] = 1;
        }

        /**
         * Run the S.hadamard(A*B) with B that is in col storage order
         * Run the SM L2 algo with A and BT where both are in "row storage order"
        */

        auto exp_result = S.hadamard(A*B);
        auto params = SDDMM::Algo::SML2SDDMM::preparations(
            S, sparsity,
            // N, M, K
            A.n, B.m, B.n, 
            A, B);
        auto result = SDDMM::Algo::SML2SDDMM::run_sm_l2(
            S, sparsity, 
            A, B,
            // N, M, K 
            A.n, B.m, B.n, 
            params);

        ASSERT_TRUE(result == exp_result);
        ASSERT_TRUE(SDDMM::Algo::SML2SDDMM::check_result(B.n /*K*/, exp_result, result, params));
    }
    {
        float sparsity = 0.97;
        SDDMM::Types::vec_size_t N = 1024;
        SDDMM::Types::vec_size_t M = 256;
        SDDMM::Types::vec_size_t K = 256;
        SDDMM::Types::COO S = SDDMM::Types::Matrix::generate_row_major(N, M, sparsity).to_coo();
        SDDMM::Types::Matrix A = SDDMM::Types::Matrix::generate_row_major(N, K, 0);
        SDDMM::Types::Matrix B = SDDMM::Types::Matrix::generate_col_major(K, M, 0);

        auto exp_result = S.hadamard(A*B);
        auto params = SDDMM::Algo::SML2SDDMM::preparations(
            S, sparsity,
            // N, M, K
            A.n, B.m, B.n, 
            A, B);
        auto result = SDDMM::Algo::SML2SDDMM::run_sm_l2(
            S, sparsity, 
            A, B,
            // N, M, K 
            A.n, B.m, B.n, 
            params);

        ASSERT_TRUE(result == exp_result);
        ASSERT_TRUE(SDDMM::Algo::SML2SDDMM::check_result(B.n /*K*/, exp_result, result, params));
    }
    {
        // very sparse, very long
        float sparsity = 0.999;
        SDDMM::Types::vec_size_t N = 128;
        SDDMM::Types::vec_size_t M = 128;
        SDDMM::Types::vec_size_t K = 32*1000;
        SDDMM::Types::COO S = SDDMM::Types::Matrix::generate_row_major(N, M, sparsity).to_coo();
        SDDMM::Types::Matrix A = SDDMM::Types::Matrix::generate_row_major(N, K, 0);
        SDDMM::Types::Matrix B = SDDMM::Types::Matrix::generate_col_major(K, M, 0);

        auto exp_result = S.hadamard(A*B);
        auto params = SDDMM::Algo::SML2SDDMM::preparations(
            S, sparsity,
            // N, M, K
            A.n, B.m, B.n, 
            A, B);
        auto result = SDDMM::Algo::SML2SDDMM::run_sm_l2(
            S, sparsity, 
            A, B,
            // N, M, K 
            A.n, B.m, B.n, 
            params);

        ASSERT_TRUE(result == exp_result);
        ASSERT_TRUE(SDDMM::Algo::SML2SDDMM::check_result(B.n /*K*/, exp_result, result, params));
    }
}
