#include <iostream>
#include "test_helpers.hpp"
#include "../src/data_structures/coo/coo.h"
#include "../src/data_structures/matrix/matrix.h"
#include "../src/algos/sm-l2-sddmm/sm-l2-sddmm.cpp"

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
    // const SDDMM::Types::vec_size_t max_thread_num = 50;
    {
        float sparsity = 0.97;
        SDDMM::Types::vec_size_t N = 1024;
        SDDMM::Types::vec_size_t M = 256;
        SDDMM::Types::vec_size_t K = 256;
        SDDMM::Types::COO S = SDDMM::Types::Matrix::generate_row_major(N, M, sparsity).to_coo();
        SDDMM::Types::Matrix A = SDDMM::Types::Matrix::generate_row_major(N, K, 0);
        SDDMM::Types::Matrix B = SDDMM::Types::Matrix::generate_col_major(K, M, 0);
        // to fit sddmm SM L2 algo requirements exactly, transpose B and then change
        // the definition of the storage to make indexing work correctly
        auto BT = B.get_transposed();
        B.set_matrix_format(SDDMM::Types::MatrixFormat::RowMajor);
        // **=> BT is now in row major storage and has sizes n=256, m=32 <=**

        // Sanity print outs ^^ (or check the two tests above)
        // ===================================================
        // std::cout << "Matrix specs ===================================" << std::endl;
        // std::cout << "A.n: \t" << A.n << ", A.m: \t" << A.m << std::endl;
        // std::cout << "B.n: \t" << B.n << ", B.m: \t" << B.m << std::endl;
        // std::cout << "BT.n:\t" << BT.n << ", BT.m:\t" << BT.m << " <= should be the same as A" << std::endl;
        // std::cout << "Contents of data ===============================" << std::endl;
        // std::cout << "A.data: \t{";
        // for(auto& c : A.data) std::cout << c << ", ";
        // std::cout << "}" << std::endl;
        // std::cout << "B.data: \t{";
        // for(auto& c : B.data) std::cout << c << ", ";
        // std::cout << "}" << std::endl;
        // std::cout << "BT.data:\t{";
        // for(auto& c : BT.data) std::cout << c << ", ";
        // std::cout << "}" << std::endl;
        // std::cout << "================================================" << std::endl;

        /**
         * Run the S.hadamard(A*B) with B that is in col storage order
         * Run the SM L2 algo with A and BT where both are in "row storage order"
        */

        // for(int num_threads = 1; num_threads<max_thread_num; ++num_threads){
        // omp_set_num_threads(num_threads);
        auto exp_result = S.hadamard(A*B);
        auto params = SDDMM::Algo::SML2SDDMM::preparations(
            S, sparsity,
            // N, M, K
            A.n, BT.n, BT.m, 
            A, BT);
        auto result = SDDMM::Algo::SML2SDDMM::run_sm_l2(
            S, sparsity, 
            A, BT,
            // N, M, K 
            A.n, BT.n, BT.m, 
            params);

        ASSERT_TRUE(SDDMM::Algo::SML2SDDMM::check_result(S, A, BT, result, params));

        // this test should "kind of" work too
        // => need to change the '==' operator to compare the correct indices of the COO matrix
        //    currently it assumes that both are ordered in the same way.
        //    Should not be much work to do that though, once the SM L2 check_results works
        // ASSERT_TRUE(exp_result == result);
    }
}
