#include "test_helpers.hpp"
#include "../src/data_structures/coo/coo.h"
#include "../src/data_structures/matrix/matrix.h"
#include "../src/algos/sm-l2-sddmm/sm-l2-sddmm.cpp"

UTEST_MAIN();

UTEST(SM_L2, Init_test) {
    // const SDDMM::Types::vec_size_t max_thread_num = 50;
    {
        float sparsity = 0.9;
        SDDMM::Types::COO S = SDDMM::Types::Matrix::generate_row_major(128, 256, sparsity).to_coo();
        SDDMM::Types::Matrix X_dense = SDDMM::Types::Matrix::generate_row_major(128, 32, 0);
        SDDMM::Types::Matrix Y_dense = SDDMM::Types::Matrix::generate_col_major(32, 256, 0);

        // for(int num_threads = 1; num_threads<max_thread_num; ++num_threads){
        // omp_set_num_threads(num_threads);
        auto exp_result = S.hadamard(X_dense*Y_dense);
        auto params = SDDMM::Algo::SML2SDDMM::preparations(S, sparsity, X_dense, Y_dense);
        auto result = SDDMM::Algo::SML2SDDMM::run_sm_l2(S, sparsity, X_dense, Y_dense, params);
        // std::cout << result << std::endl;
        // std::cout << "=============" << std::endl;
        // std::cout << exp_result << std::endl;

        ASSERT_TRUE(SDDMM::Algo::SML2SDDMM::check_result(S, X_dense, Y_dense, result, params));
        // }
    }
}
