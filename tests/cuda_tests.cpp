#include "test_helpers.hpp"
#include "../src/algos/cuda_sddmm/cuda_sddmm.cpp"
#include "../src/algos/cuda_tiled_sddmm/cuda_tiled_sddmm.cpp"
#include "../src/algos/cuda_mat_mult/cuda_tiled_mat_mult.cpp"
#include "../src/algos/sm-l2-sddmm/sm-l2-sddmm.cpp"
#include "../src/cuda_examples/cusparse_example_1.cpp"

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

        ASSERT_TRUE(result == exp_result);
        // }
    }
}

UTEST(Cuda_SDDMM, Tiled) {
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
        auto result = SDDMM::Algo::cuda_tiled_sddmm(coo_mat, X, Y);
        // std::cout << result << std::endl;
        // std::cout << "=============" << std::endl;
        // std::cout << exp_result << std::endl;

        ASSERT_TRUE(result == exp_result);
        // }
    }
}

UTEST(cuSPARSE, Test) {
    {
        // cuSPARSE test
        SDDMM::CUDA_EXAMPLES::cuSPARSE_example_1();
    }
}