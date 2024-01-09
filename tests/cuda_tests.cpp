#include "test_helpers.hpp"
#include "../src/algos/cuda_sddmm/cuda_sddmm.cpp"
#include "../src/algos/cuda_mat_mult/cuda_tiled_mat_mult.cpp"
#include "../src/algos/sm-l2-sddmm/sm-l2-sddmm.cpp"
#include "../src/cuda_examples/cusparse_example_1.cpp"
#include "../src/algos/cusparse_sddmm/cusparse_1.cpp"

UTEST_MAIN();

UTEST(Cuda_SDDMM, Non_Tiled) {
    // const SDDMM::Types::vec_size_t max_thread_num = 50;
    {
        auto X = SDDMM::Types::Matrix::deterministic_gen_row_major(5, 4, {
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12,
            13, 14, 15, 16,
            17, 18, 19, 20
        });

        auto Y = SDDMM::Types::Matrix::deterministic_gen_col_major(4, 5, {
            2,  4,  6,  8, 10,
            12, 14, 16, 18, 20,
            22, 24, 26, 28, 30,
            32, 34, 36, 38, 40
        });

        auto inner_prod_res = SDDMM::Types::Matrix::deterministic_gen_row_major(5,5, {
            220,  240,  260,  280,  300,
            492,  544,  596,  648,  700,
            764,  848,  932, 1016, 1100,
            1036, 1152, 1268, 1384, 1500,
            1308, 1456, 1604, 1752, 1900
        });

        auto res = X*Y;
        ASSERT_TRUE(inner_prod_res == res);

        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(5, 5, {
            0.5, 1.0, 0.5, 1.0, 0.5,
            1.0, 0.5, 1.0, 0.5, 1.0,
            0.5, 1.0, 0.5, 1.0, 0.5,
            1.0, 0.5, 1.0, 0.5, 1.0,
            0.5, 1.0, 0.5, 1.0, 0.5
        });
        auto coo_mat = temp.to_coo();

        // Expected COO outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(5, 5, {
             110,  240,  130,  280,  150,
             492,  272,  596,  324,  700,
             382,  848,  466, 1016,  550,
            1036,  576, 1268,  692, 1500,
             654, 1456,  802, 1752,  950,
        });

        // for(int num_threads = 1; num_threads<max_thread_num; ++num_threads){
        // omp_set_num_threads(num_threads);
        auto exp_result = result_temp.to_coo();
        auto result = SDDMM::Algo::cuda_sddmm(coo_mat, X, Y);
        // std::cout << result << std::endl;
        // std::cout << "=============" << std::endl;
        // std::cout << exp_result << std::endl;

        ASSERT_TRUE(result.equals(exp_result));
        // }
    }
}

//UTEST(Cuda_SDDMM, Tiled) {
//    // const SDDMM::Types::vec_size_t max_thread_num = 50;
//    {
//        auto X = SDDMM::Types::Matrix::deterministic_gen_row_major(5, 4, {
//            1,  2,  3,  4,
//            5,  6,  7,  8,
//            9, 10, 11, 12,
//            13, 14, 15, 16,
//            17, 18, 19, 20
//        });
//
//        auto Y = SDDMM::Types::Matrix::deterministic_gen_col_major(4, 5, {
//            2,  4,  6,  8, 10,
//            12, 14, 16, 18, 20,
//            22, 24, 26, 28, 30,
//            32, 34, 36, 38, 40
//        });
//
//        auto inner_prod_res = SDDMM::Types::Matrix::deterministic_gen_row_major(5,5, {
//            220,  240,  260,  280,  300,
//            492,  544,  596,  648,  700,
//            764,  848,  932, 1016, 1100,
//            1036, 1152, 1268, 1384, 1500,
//            1308, 1456, 1604, 1752, 1900
//        });
//
//        auto res = X*Y;
//        ASSERT_TRUE(inner_prod_res == res);
//
//        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(5, 5, {
//            0.5, 1.0, 0.5, 1.0, 0.5,
//            1.0, 0.5, 1.0, 0.5, 1.0,
//            0.5, 1.0, 0.5, 1.0, 0.5,
//            1.0, 0.5, 1.0, 0.5, 1.0,
//            0.5, 1.0, 0.5, 1.0, 0.5
//        });
//        auto coo_mat = temp.to_coo();
//
//        // Expected COO outputs.
//        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(5, 5, {
//             110,  240,  130,  280,  150,
//             492,  272,  596,  324,  700,
//             382,  848,  466, 1016,  550,
//            1036,  576, 1268,  692, 1500,
//             654, 1456,  802, 1752,  950,
//        });
//
//        // for(int num_threads = 1; num_threads<max_thread_num; ++num_threads){
//        // omp_set_num_threads(num_threads);
//        auto exp_result = result_temp.to_coo();
//        auto result = SDDMM::Algo::cuda_tiled_sddmm(coo_mat, X, Y);
//        // std::cout << result << std::endl;
//        // std::cout << "=============" << std::endl;
//        // std::cout << exp_result << std::endl;
//
//        ASSERT_TRUE(result.equals(exp_result));
//        // }
//    }
//}

UTEST(cuSPARSE, Test) {
    {
        // cuSPARSE test
        ASSERT_TRUE(SDDMM::CUDA_EXAMPLES::cuSPARSE_example_1());
    }
}

UTEST(cuSPARSE, SDDMM_Test) {
    {
        auto A = SDDMM::Types::Matrix::deterministic_gen_row_major(4, 4, {
            1.0f,   2.0f,  3.0f,  4.0f,
            5.0f,   6.0f,  7.0f,  8.0f,
            9.0f,  10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f 
        });

        auto B = SDDMM::Types::Matrix::deterministic_gen_col_major(4, 3, {
            1.0f,  2.0f,  3.0f,
            4.0f,  5.0f,  6.0f,
            7.0f,  8.0f,  9.0f,
            10.0f, 11.0f, 12.0f
        });

        SDDMM::Types::CSR S;
        S.n = 4;
        S.m = 3;
        S.row_ptr = { 0, 3, 4, 7, 9 };
        S.col_idx = { 0, 1, 2, 1, 0, 1, 2, 0, 2 };
        S.values = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

        auto exp_result = S.hadamard(A*B);
        auto result = SDDMM::Algo::cuSPARSE_SDDMM(S, A, B);

        ASSERT_TRUE(TestHelpers::compare_vectors(exp_result.values, result.values));
    }
    {
        // very sparse, very long
        float sparsity = 0.8;
        SDDMM::Types::vec_size_t N = 4;
        SDDMM::Types::vec_size_t M = 3;
        SDDMM::Types::vec_size_t K = 4;
        SDDMM::Types::Matrix Sd = SDDMM::Types::Matrix::generate_row_major(N, M, sparsity);
        SDDMM::Types::CSR S = Sd.to_csr();
        // set all S values to 1
        for(SDDMM::Types::vec_size_t i=0; i<S.values.size(); ++i){
            S.values[i] = 1.0f;
        }
        SDDMM::Types::Matrix A = SDDMM::Types::Matrix::generate_row_major(N, K, 0);
        SDDMM::Types::Matrix B = SDDMM::Types::Matrix::generate_col_major(K, M, 0);

        auto exp_result = S.hadamard(A*B);
        auto result = SDDMM::Algo::cuSPARSE_SDDMM(S, A, B);

        ASSERT_TRUE(TestHelpers::compare_vectors(exp_result.values, result.values));
    }
    {
        // very sparse, very long
        float sparsity = 0.8;
        SDDMM::Types::vec_size_t N = 1024;
        SDDMM::Types::vec_size_t M = 2048;
        SDDMM::Types::vec_size_t K = 128;
        SDDMM::Types::Matrix Sd = SDDMM::Types::Matrix::generate_row_major(N, M, sparsity);
        SDDMM::Types::CSR S = Sd.to_csr();
        // set all S values to 1
        for(SDDMM::Types::vec_size_t i=0; i<S.values.size(); ++i){
            S.values[i] = 1.0f;
        }
        SDDMM::Types::Matrix A = SDDMM::Types::Matrix::generate_row_major(N, K, 0);
        SDDMM::Types::Matrix B = SDDMM::Types::Matrix::generate_col_major(K, M, 0);

        auto exp_result = S.hadamard(A*B);
        auto result = SDDMM::Algo::cuSPARSE_SDDMM(S, A, B);

        ASSERT_TRUE(TestHelpers::compare_vectors(exp_result.values, result.values));
    }
}