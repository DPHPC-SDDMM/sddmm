#include <iostream>
#include <cstdio>
#include <vector>
#include <math.h>
#include <unordered_set>
#include "test_helpers.hpp"
#include "../src/data_structures/matrix/matrix.h"
#include "../src/data_structures/csr/csr.h"
#include "../src/data_structures/coo/coo.h"
#include "../src/algos/cpu_sddmm/naive_sddmm.cpp"
#include "../src/algos/cpu_sddmm/tiled_sddmm.cpp"
#include "../src/algos/cpu_sddmm/parallel_cpu_sddmm.cpp"

UTEST_MAIN();

UTEST(Matrix, TestEquals) {
    auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_row_major(2, 3, {1,2,3,4,5,6});
    auto matrix2 = SDDMM::Types::Matrix::deterministic_gen_col_major(2, 3, {1,2,3,4,5,6});

    ASSERT_TRUE(matrix1 == matrix2);
}

UTEST(Matrix, TestDenseMult) {
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_row_major(2, 3, {1,2,3,4,5,6});
        auto matrix2 = SDDMM::Types::Matrix::deterministic_gen_col_major(3, 2, {1,2,3,4,5,6});

        auto matrix3 = SDDMM::Types::Matrix::deterministic_gen_row_major(2,2, {22, 28, 49, 64});
        auto result = matrix1*matrix2;

        ASSERT_TRUE(result.n == matrix1.n && result.m == matrix2.m);
        ASSERT_TRUE(matrix3 == result);
    }
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_row_major(1, 1, {2});
        auto matrix2 = SDDMM::Types::Matrix::deterministic_gen_col_major(1, 1, {1});

        auto matrix3 = SDDMM::Types::Matrix::deterministic_gen_row_major(1, 1, {2});
        auto result = matrix1*matrix2;

        ASSERT_TRUE(result.n == matrix1.n && result.m == matrix2.m);
        ASSERT_TRUE(matrix3 == result);
    }
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_row_major(1, 10, {1,2,3,4,5,6,7,8,9,10});
        auto matrix2 = SDDMM::Types::Matrix::deterministic_gen_col_major(10, 1, {1,2,3,4,5,6,7,8,9,10});

        auto matrix3 = SDDMM::Types::Matrix::deterministic_gen_row_major(1, 1, {385});
        auto result = matrix1*matrix2;

        ASSERT_TRUE(result.n == matrix1.n && result.m == matrix2.m);
        ASSERT_TRUE(matrix3 == result);
    }
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 1, {1,2,3});
        auto matrix2 = SDDMM::Types::Matrix::deterministic_gen_col_major(1, 3, {1,2,3});

        auto matrix3 = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {1,2,3,2,4,6,3,6,9});
        auto result = matrix1*matrix2;

        ASSERT_TRUE(result.n == matrix1.n && result.m == matrix2.m);
        ASSERT_TRUE(matrix3 == result);
    }
}

UTEST(Matrix, ToCSR) {
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9,10,11,12
        });

        auto csr_mat = matrix1.to_csr();

        std::vector<SDDMM::Types::expmt_t> values = {1,2,3,4,5,6,7,8,9,10,11,12};
        std::vector<SDDMM::Types::vec_size_t> col_idx = {0,1,2,3,0,1,2,3,0,1,2,3};
        std::vector<SDDMM::Types::vec_size_t> row_ptr = {0,4,8,12};

        ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::expmt_t>(csr_mat.values, values));
        ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::vec_size_t>(csr_mat.col_idx, col_idx));
        ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::vec_size_t>(csr_mat.row_ptr, row_ptr));
    }
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            1, 0, 3, 0,
            5, 6, 0, 8,
            0,10, 0,12
        });

        auto csr_mat = matrix1.to_csr();

        std::vector<SDDMM::Types::expmt_t> values = {1,3,5,6,8,10,12};
        std::vector<SDDMM::Types::vec_size_t> col_idx = {0,2,0,1,3,1,3};
        std::vector<SDDMM::Types::vec_size_t> row_ptr = {0,2,5,7};

        ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::expmt_t>(csr_mat.values, values));
        ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::vec_size_t>(csr_mat.col_idx, col_idx));
        ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::vec_size_t>(csr_mat.row_ptr, row_ptr));
    }
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            0, 0, 0, 0,
            0, 0, 0, 0,
            0,10, 0,12
        });

        auto csr_mat = matrix1.to_csr();

        std::vector<SDDMM::Types::expmt_t> values = {10,12};
        std::vector<SDDMM::Types::vec_size_t> col_idx = {1,3};
        std::vector<SDDMM::Types::vec_size_t> row_ptr = {0,0,0,2};

        ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::expmt_t>(csr_mat.values, values));
        ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::vec_size_t>(csr_mat.col_idx, col_idx));
        ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::vec_size_t>(csr_mat.row_ptr, row_ptr));
    }
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        });

        auto csr_mat = matrix1.to_csr();

        std::vector<SDDMM::Types::expmt_t> values = {};
        std::vector<SDDMM::Types::vec_size_t> col_idx = {};
        std::vector<SDDMM::Types::vec_size_t> row_ptr = {0,0,0,0};

        ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::expmt_t>(csr_mat.values, values));
        ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::vec_size_t>(csr_mat.col_idx, col_idx));
        ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::vec_size_t>(csr_mat.row_ptr, row_ptr));
    }
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_row_major(0, 0, {
        });

        auto csr_mat = matrix1.to_csr();

        std::vector<SDDMM::Types::expmt_t> values = {};
        std::vector<SDDMM::Types::vec_size_t> col_idx = {};
        std::vector<SDDMM::Types::vec_size_t> row_ptr = {0};

        ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::expmt_t>(csr_mat.values, values));
        ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::vec_size_t>(csr_mat.col_idx, col_idx));
        ASSERT_TRUE(TestHelpers::compare_vectors<SDDMM::Types::vec_size_t>(csr_mat.row_ptr, row_ptr));
    }
}

UTEST(Matrix, Hadamard) {
    {
        auto dense = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            0.5, 0.5, 0.5, 0.5,
            2.0, 2.0, 2.0, 2.0,
            2.0, 0.5, 2.0, 0.5
        });

        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9,10,11,12
        });

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            0.5, 1, 1.5, 2,
            10, 12, 14, 16,
            18, 5, 22, 6 
        });

        auto csr_mat = temp.to_csr();
        auto csr_exp_result = result_temp.to_csr();
        auto csr_result = csr_mat.hadamard(dense);
        ASSERT_TRUE(csr_result.equals(csr_exp_result));

        auto coo_mat = temp.to_coo();
        auto coo_exp_result = result_temp.to_coo();
        auto coo_result = coo_mat.hadamard(dense);
        ASSERT_TRUE(coo_result.equals(coo_exp_result));
    }
    {
        auto dense = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            0.5, 0.5, 0.5, 0.5,
            2.0, 2.0, 2.0, 2.0,
            2.0, 0.5, 2.0, 0.5
        });

        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            0, 0, 0, 0,
            5, 6, 7, 8,
            9,10,11,12
        });

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            0, 0, 0, 0,
            10, 12, 14, 16,
            18, 5, 22, 6 
        });
        
        auto csr_mat = temp.to_csr();
        auto csr_exp_result = result_temp.to_csr();
        auto csr_result = csr_mat.hadamard(dense);
        ASSERT_TRUE(csr_result.equals(csr_exp_result));

        auto coo_mat = temp.to_coo();
        auto coo_exp_result = result_temp.to_coo();
        auto coo_result = coo_mat.hadamard(dense);
        ASSERT_TRUE(coo_result.equals(coo_exp_result));
    }
    {
        auto dense = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            0.5, 0.5, 0.5, 0.5,
            2.0, 2.0, 2.0, 2.0,
            2.0, 0.5, 2.0, 0.5
        });

        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            1, 2, 3, 4,
            0, 0, 0, 0,
            9,10,11,12
        });

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            0.5, 1, 1.5, 2,
            0, 0, 0, 0,
            18, 5, 22, 6 
        });

        auto csr_mat = temp.to_csr();
        auto csr_exp_result = result_temp.to_csr();
        auto csr_result = csr_mat.hadamard(dense);
        ASSERT_TRUE(csr_result.equals(csr_exp_result));

        auto coo_mat = temp.to_coo();
        auto coo_exp_result = result_temp.to_coo();
        auto coo_result = coo_mat.hadamard(dense);
        ASSERT_TRUE(coo_result.equals(coo_exp_result));
    }
    {
        auto dense = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            0.5, 0.5, 0.5, 0.5,
            2.0, 2.0, 2.0, 2.0,
            2.0, 0.5, 2.0, 0.5
        });

        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            1, 0, 3, 0,
            5, 0, 7, 8,
            0,10, 0, 0
        });

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            0.5, 0, 1.5, 0,
            10,  0, 14, 16,
             0, 5,  0, 0 
        });

        auto csr_mat = temp.to_csr();
        auto csr_exp_result = result_temp.to_csr();
        auto csr_result = csr_mat.hadamard(dense);
        ASSERT_TRUE(csr_result.equals(csr_exp_result));

        auto coo_mat = temp.to_coo();
        auto coo_exp_result = result_temp.to_coo();
        auto coo_result = coo_mat.hadamard(dense);
        ASSERT_TRUE(coo_result.equals(coo_exp_result));
    }
    {
        auto dense = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            0.5, 0.5, 0.5, 0.5,
            2.0, 2.0, 2.0, 2.0,
            2.0, 0.5, 2.0, 0.5
        });

        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        });

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0 
        });

        auto csr_mat = temp.to_csr();
        auto csr_exp_result = result_temp.to_csr();
        auto csr_result = csr_mat.hadamard(dense);
        ASSERT_TRUE(csr_result.equals(csr_exp_result));

        auto coo_mat = temp.to_coo();
        auto coo_exp_result = result_temp.to_coo();
        auto coo_result = coo_mat.hadamard(dense);
        ASSERT_TRUE(coo_result.equals(coo_exp_result));
    }
    {
        auto dense = SDDMM::Types::Matrix::deterministic_gen_row_major(1, 1, {
            0.5
        });

        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(1, 1, {
            1
        });

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(1, 1, {
            0.5 
        });
        
        auto csr_mat = temp.to_csr();
        auto csr_exp_result = result_temp.to_csr();
        auto csr_result = csr_mat.hadamard(dense);
        ASSERT_TRUE(csr_result.equals(csr_exp_result));

        auto coo_mat = temp.to_coo();
        auto coo_exp_result = result_temp.to_coo();
        auto coo_result = coo_mat.hadamard(dense);
        ASSERT_TRUE(coo_result.equals(coo_exp_result));
    }
    {
        auto dense = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            0.0, 0.5, 0.0, 0.5,
            2.0, 2.0, 2.0, 2.0,
            2.0, 0.5, 2.0, 0.5
        });

        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            1, 0, 3, 0,
            5, 0, 7, 8,
            0,10, 0, 0
        });

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
             0,  0,  0,  0,
            10,  0, 14, 16,
             0,  5,  0,  0 
        });
        
        auto csr_mat = temp.to_csr();
        auto csr_exp_result = result_temp.to_csr();
        auto csr_result = csr_mat.hadamard(dense);
        ASSERT_TRUE(csr_result.equals(csr_exp_result));

        auto coo_mat = temp.to_coo();
        auto coo_exp_result = result_temp.to_coo();
        auto coo_result = coo_mat.hadamard(dense);
        ASSERT_TRUE(coo_result.equals(coo_exp_result));
    }
}

UTEST(Matrix, Flip) {
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_row_major(2, 3, {1,2,3,4,5,6});
        auto matrix2 = SDDMM::Types::Matrix::deterministic_gen_row_major(2, 3, {1,2,3,4,5,6});

        matrix2.to_dense_col_major();
        std::vector<SDDMM::Types::expmt_t> newVals = {1,4,2,5,3,6};
        ASSERT_TRUE(TestHelpers::compare_vectors(matrix2.data, newVals));
        ASSERT_TRUE(matrix1 == matrix2);
    }
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_col_major(2, 3, {1,2,3,4,5,6});

        std::vector<SDDMM::Types::expmt_t> newVals = {1,4,2,5,3,6};
        ASSERT_TRUE(TestHelpers::compare_vectors(matrix1.data, newVals));
    }
}

UTEST(Matrix, CSR_COO_Conversion) {
    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
            0.5, 1.0, 0.5,
            1.0, 0.5, 1.0,
            0.5, 1.0, 0.5
        });

        auto csr_mat = temp.to_csr();
        auto coo_mat = temp.to_coo();

        auto csr_temp = coo_mat.to_csr();
        auto coo_temp = csr_mat.to_coo();
        auto coo_matrix_temp = coo_mat.to_matrix();
        auto csr_matrix_temp = csr_mat.to_matrix();

        ASSERT_TRUE(csr_mat.equals(csr_temp));
        ASSERT_TRUE(coo_mat.equals(coo_temp));
        ASSERT_TRUE(temp == coo_matrix_temp);
        ASSERT_TRUE(temp == csr_matrix_temp);
    }
    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
            0.5, 0.0, 0.5,
            0.0, 0.5, 0.0,
            0.5, 0.0, 0.5
        });

        auto csr_mat = temp.to_csr();
        auto coo_mat = temp.to_coo();

        auto csr_temp = coo_mat.to_csr();
        auto coo_temp = csr_mat.to_coo();
        auto coo_matrix_temp = coo_mat.to_matrix();
        auto csr_matrix_temp = csr_mat.to_matrix();

        ASSERT_TRUE(csr_mat.equals(csr_temp));
        ASSERT_TRUE(coo_mat.equals(coo_temp));
        ASSERT_TRUE(temp == coo_matrix_temp);
        ASSERT_TRUE(temp == csr_matrix_temp);
    }
    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
            -0.5, 0.0, 0.5,
            0.0, 0.5, 0.0,
            0.5, 0.0, 0.5
        });

        auto csr_mat = temp.to_csr();
        auto coo_mat = temp.to_coo();
        
        auto csr_temp = coo_mat.to_csr();
        auto coo_temp = csr_mat.to_coo();
        auto coo_matrix_temp = coo_mat.to_matrix();
        auto csr_matrix_temp = csr_mat.to_matrix();

        ASSERT_TRUE(csr_mat.equals(csr_temp));
        ASSERT_TRUE(coo_mat.equals(coo_temp));
        ASSERT_TRUE(temp == coo_matrix_temp);
        ASSERT_TRUE(temp == csr_matrix_temp);
    }
    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        });

        auto csr_mat = temp.to_csr();
        auto coo_mat = temp.to_coo();
        
        auto csr_temp = coo_mat.to_csr();
        auto coo_temp = csr_mat.to_coo();
        auto coo_matrix_temp = coo_mat.to_matrix();
        auto csr_matrix_temp = csr_mat.to_matrix();

        ASSERT_TRUE(csr_mat.equals(csr_temp));
        ASSERT_TRUE(coo_mat.equals(coo_temp));
        ASSERT_TRUE(temp == coo_matrix_temp);
        ASSERT_TRUE(temp == csr_matrix_temp);
    }
    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(5, 5, {
            0.5, 1.0, 0.5, 1.0, 0.5,
            1.0, 0.5, 1.0, 0.5, 1.0,
            0.5, 1.0, 0.5, 1.0, 0.5,
            1.0, 0.5, 1.0, 0.5, 1.0,
            0.5, 1.0, 0.5, 1.0, 0.5
        });

        auto csr_mat = temp.to_csr();
        auto coo_mat = temp.to_coo();
        
        auto csr_temp = coo_mat.to_csr();
        auto coo_temp = csr_mat.to_coo();
        auto coo_matrix_temp = coo_mat.to_matrix();
        auto csr_matrix_temp = csr_mat.to_matrix();

        ASSERT_TRUE(csr_mat.equals(csr_temp));
        ASSERT_TRUE(coo_mat.equals(coo_temp));
        ASSERT_TRUE(temp == coo_matrix_temp);
        ASSERT_TRUE(temp == csr_matrix_temp);
    }
}

UTEST(Matrix, SDDMM_op) {
    {
        auto X = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12
        });

        auto Y = SDDMM::Types::Matrix::deterministic_gen_col_major(4, 3, {
            2,  4,  6,
            8, 10, 12,
            14, 16, 18,
            20, 22, 24

        });

        auto inner_prod_res = SDDMM::Types::Matrix::deterministic_gen_row_major(3,3, {
            140, 160, 180, 
            316, 368, 420, 
            492, 576, 660
        });

        ASSERT_TRUE(inner_prod_res == X*Y);

        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
            0.5, 1.0, 0.5,
            1.0, 0.5, 1.0,
            0.5, 1.0, 0.5
        });

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
             70, 160,  90,
            316, 184, 420,
            246, 576, 330
        });
        auto csr_mat = temp.to_csr();
        auto coo_mat = temp.to_coo();

        auto exp_result = result_temp.to_csr();
        auto exp_result_coo = result_temp.to_coo();
        auto result1 = SDDMM::Algo::naive_sddmm(csr_mat, X, Y);
        ASSERT_TRUE(result1.equals(exp_result));
        auto result2 = SDDMM::Algo::naive_sddmm(coo_mat, X, Y);
        ASSERT_TRUE(result2.equals(exp_result_coo));
        auto result3 = SDDMM::Algo::tiled_sddmm(csr_mat, X, Y, 8, 8, 8);
        ASSERT_TRUE(result3.equals(exp_result));
        auto result4 = SDDMM::Algo::tiled_sddmm(csr_mat, X, Y, 128, 128, 128);
        ASSERT_TRUE(result4.equals(exp_result));
    }
    {
        auto X = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12
        });

        auto Y = SDDMM::Types::Matrix::deterministic_gen_col_major(4, 3, {
            2,  4,  6,
            8, 10, 12,
            14, 16, 18,
            20, 22, 24

        });

        auto inner_prod_res = SDDMM::Types::Matrix::deterministic_gen_row_major(3,3, {
            140, 160, 180, 
            316, 368, 420, 
            492, 576, 660
        });

        ASSERT_TRUE(inner_prod_res == X*Y);

        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
            0.5, 0.0, 0.5,
            0.0, 0.5, 0.0,
            0.5, 0.0, 0.5
        });

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
             70,   0,  90,
              0, 184,   0,
            246,   0, 330
        });
        auto csr_mat = temp.to_csr();
        auto coo_mat = temp.to_coo();

        auto exp_result = result_temp.to_csr();
        auto exp_result_coo = result_temp.to_coo();
        auto result1 = SDDMM::Algo::naive_sddmm(csr_mat, X, Y);
        ASSERT_TRUE(result1.equals(exp_result));
        auto result2 = SDDMM::Algo::naive_sddmm(coo_mat, X, Y);
        ASSERT_TRUE(result2.equals(exp_result_coo));
        auto result3 = SDDMM::Algo::tiled_sddmm(csr_mat, X, Y, 8, 8, 8);
        ASSERT_TRUE(result3.equals(exp_result));
        auto result4 = SDDMM::Algo::tiled_sddmm(csr_mat, X, Y, 128, 128, 128);
        ASSERT_TRUE(result4.equals(exp_result));
    }
    {
        auto X = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12
        });

        auto Y = SDDMM::Types::Matrix::deterministic_gen_col_major(4, 3, {
            2,  4,  6,
            8, 10, 12,
            14, 16, 18,
            20, 22, 24

        });

        auto inner_prod_res = SDDMM::Types::Matrix::deterministic_gen_row_major(3,3, {
            140, 160, 180, 
            316, 368, 420, 
            492, 576, 660
        });

        ASSERT_TRUE(inner_prod_res == X*Y);

        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
            -0.5, 0.0, 0.5,
            0.0, 0.5, 0.0,
            0.5, 0.0, 0.5
        });

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
             -70,   0,  90,
              0, 184,   0,
            246,   0, 330
        });
        auto csr_mat = temp.to_csr();
        auto coo_mat = temp.to_coo();
        
        auto exp_result = result_temp.to_csr();
        auto exp_result_coo = result_temp.to_coo();
        auto result1 = SDDMM::Algo::naive_sddmm(csr_mat, X, Y);
        ASSERT_TRUE(result1.equals(exp_result));
        auto result2 = SDDMM::Algo::naive_sddmm(coo_mat, X, Y);
        ASSERT_TRUE(result2.equals(exp_result_coo));
        auto result3 = SDDMM::Algo::tiled_sddmm(csr_mat, X, Y, 8, 8, 8);
        ASSERT_TRUE(result3.equals(exp_result));
        auto result4 = SDDMM::Algo::tiled_sddmm(csr_mat, X, Y, 128, 128, 128);
        ASSERT_TRUE(result4.equals(exp_result));
    }
    {
        auto X = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12
        });

        auto Y = SDDMM::Types::Matrix::deterministic_gen_col_major(4, 3, {
            2,  4,  6,
            8, 10, 12,
            14, 16, 18,
            20, 22, 24

        });

        auto inner_prod_res = SDDMM::Types::Matrix::deterministic_gen_row_major(3,3, {
            140, 160, 180, 
            316, 368, 420, 
            492, 576, 660
        });

        ASSERT_TRUE(inner_prod_res == X*Y);

        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        });

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
              0,   0,   0,
              0,   0,   0,
              0,   0,   0
        });
        auto csr_mat = temp.to_csr();
        auto coo_mat = temp.to_coo();
        
        auto exp_result = result_temp.to_csr();
        auto exp_result_coo = result_temp.to_coo();
        auto result1 = SDDMM::Algo::naive_sddmm(csr_mat, X, Y);
        ASSERT_TRUE(result1.equals(exp_result));
        auto result2 = SDDMM::Algo::naive_sddmm(coo_mat, X, Y);
        ASSERT_TRUE(result2.equals(exp_result_coo));
        auto result3 = SDDMM::Algo::tiled_sddmm(csr_mat, X, Y, 8, 8, 8);
        ASSERT_TRUE(result3.equals(exp_result));
        auto result4 = SDDMM::Algo::tiled_sddmm(csr_mat, X, Y, 128, 128, 128);
        ASSERT_TRUE(result4.equals(exp_result));
    }
    {
        auto X = SDDMM::Types::Matrix::deterministic_gen_row_major(5, 4, {
            6,  -1,  0,  0,
            5,   6,  7,  8,
            0,   0,  0,  0,
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
               0,   10,   20,   30,   40,
             492,  544,  596,  648,  700,
               0,    0,    0,    0,    0,
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

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(5, 5, {
               0,   10,   10,   30,   20,
             492,  272,  596,  324,  700,
               0,    0,    0,    0,    0,
            1036,  576, 1268,  692, 1500,
             654, 1456,  802, 1752,  950
        });

        auto csr_mat = temp.to_csr();
        auto coo_mat = temp.to_coo();
        
        auto exp_result = result_temp.to_csr();
        auto exp_result_coo = result_temp.to_coo();
        auto result1 = SDDMM::Algo::naive_sddmm(csr_mat, X, Y);
        ASSERT_TRUE(result1.equals(exp_result));
        auto result2 = SDDMM::Algo::naive_sddmm(coo_mat, X, Y);
        ASSERT_TRUE(result2.equals(exp_result_coo));
        auto result3 = SDDMM::Algo::tiled_sddmm(csr_mat, X, Y, 8, 8, 8);
        ASSERT_TRUE(result3.equals(exp_result));
        auto result4 = SDDMM::Algo::tiled_sddmm(csr_mat, X, Y, 128, 128, 128);
        ASSERT_TRUE(result4.equals(exp_result));
    }
}

UTEST(Matrix, COO_equal){
    auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_row_major(2, 3, {1,2,3,4,5,6});
    auto matrix2 = SDDMM::Types::Matrix::deterministic_gen_row_major(2, 3, {1,2,3,4,5,6});
    auto matrix3 = SDDMM::Types::Matrix::deterministic_gen_row_major(2, 3, {1,2,3,4,5,7});

    ASSERT_TRUE(matrix1 == matrix2);
    ASSERT_TRUE(matrix1.to_coo().equals(matrix2.to_coo()));
    ASSERT_FALSE(matrix1 == matrix3);
    ASSERT_FALSE(matrix1.to_coo().equals(matrix3.to_coo()));
}

UTEST(Matrix, SDDMM_parallel) {
    const SDDMM::Types::vec_size_t max_thread_num = 50;
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

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(5, 5, {
             110,  240,  130,  280,  150,
             492,  272,  596,  324,  700,
             382,  848,  466, 1016,  550,
            1036,  576, 1268,  692, 1500,
             654, 1456,  802, 1752,  950,
        });

        for(int num_threads = 1; num_threads<max_thread_num; ++num_threads){
            omp_set_num_threads(num_threads);
            auto exp_result = result_temp.to_coo();
            auto result1 = SDDMM::Algo::parallel_sddmm_cuda_simulation(coo_mat, X, Y, num_threads);
            ASSERT_TRUE(result1.equals(exp_result));
            auto result2 = SDDMM::Algo::parallel_sddmm(coo_mat, X, Y, num_threads);
            ASSERT_TRUE(result2.equals(exp_result));
            auto result3 = SDDMM::Algo::parallel_sddmm_slow(coo_mat, X, Y, num_threads);
            ASSERT_TRUE(result3.equals(exp_result));
        }
    }
    {
#ifdef SDDMM_PARALLEL_CPU_ZERO_FILTER
        auto X = SDDMM::Types::Matrix::deterministic_gen_row_major(5, 4, {
            6,  -1,  0,  0,
            5,   6,  7,  8,
            0,   0,  0,  0,
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
               0,   10,   20,   30,   40,
             492,  544,  596,  648,  700,
               0,    0,    0,    0,    0,
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

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(5, 5, {
               0,   10,   10,   30,   20,
             492,  272,  596,  324,  700,
               0,    0,    0,    0,    0,
            1036,  576, 1268,  692, 1500,
             654, 1456,  802, 1752,  950
        });

        for(int num_threads = 1; num_threads<max_thread_num; ++num_threads){
            omp_set_num_threads(num_threads);
            auto exp_result = result_temp.to_coo();
            auto result1 = SDDMM::Algo::parallel_sddmm_cuda_simulation(coo_mat, X, Y, num_threads);
            ASSERT_TRUE(result1.equals(exp_result));
            auto result2 = SDDMM::Algo::parallel_sddmm(coo_mat, X, Y, num_threads);
            ASSERT_TRUE(result2.equals(exp_result));
            auto result3 = SDDMM::Algo::parallel_sddmm_slow(coo_mat, X, Y, num_threads);
            ASSERT_TRUE(result3.equals(exp_result));
        }
#endif
    }
    {
        auto X = SDDMM::Types::Matrix::deterministic_gen_row_major(1, 1, {
            1
        });

        auto Y = SDDMM::Types::Matrix::deterministic_gen_col_major(1, 1, {
             2
        });

        auto inner_prod_res = SDDMM::Types::Matrix::deterministic_gen_row_major(1,1, {
               2
        });

        auto res = X*Y;
        ASSERT_TRUE(inner_prod_res == res);

        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(1, 1, {
            0.5
        });
        auto coo_mat = temp.to_coo();

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(1, 1, {
               1
        });

        for(int num_threads = 1; num_threads<max_thread_num; ++num_threads){
            omp_set_num_threads(num_threads);
            auto exp_result = result_temp.to_coo();
            auto result1 = SDDMM::Algo::parallel_sddmm_cuda_simulation(coo_mat, X, Y, num_threads);
            ASSERT_TRUE(result1.equals(exp_result));
            auto result2 = SDDMM::Algo::parallel_sddmm(coo_mat, X, Y, num_threads);
            ASSERT_TRUE(result2.equals(exp_result));
            auto result3 = SDDMM::Algo::parallel_sddmm_slow(coo_mat, X, Y, num_threads);
            ASSERT_TRUE(result3.equals(exp_result));
        }
    }
}

// UTEST(Matrix, SDDMM_giant_op) {
//     auto X = SDDMM::Types::Matrix::generate_row_major(5000, 8000);

//     auto Y = SDDMM::Types::Matrix::generate_row_major(8000, 5000);

//     // Note: for time testing purposes, do this outside of this
//     // method
//     Y.to_dense_col_major();

//     std::cout << "Finished transformation" << std::endl;

//     // Expected CSR outputs.
//     auto mat = SDDMM::Types::Matrix::generate_row_major(5000, 5000, 0.1);
//     auto csr_mat = mat.to_csr();
//     auto result = SDDMM::Algo::naive_sddmm(csr_mat, X, Y);

//     std::cout << "Result: " << result << std::endl;
// }

// UTEST(Matrix, SDDMM_op_zero) {
//     auto X = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
//         0,  0,  0,  0,
//         5,  6,  7,  8,
//         9, 10, 11, 12
//     });

//     auto Y = SDDMM::Types::Matrix::deterministic_gen_row_major(4, 3, {
//          2,  4,  6,
//          8, 10, 12,
//         14, 16, 18,
//         20, 22, 24

//     });

//     // Note: for time testing purposes, do this outside of this
//     // method
//     Y.to_dense_col_major();

//     auto inner_prod_res = SDDMM::Types::Matrix::deterministic_gen_row_major(3,3, {
//           0,   0,   0, 
//         316, 368, 420, 
//         492, 576, 660
//     });

//     ASSERT_TRUE(inner_prod_res == X*Y);
//     {
//         auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
//             0.5, 1.0, 0.5,
//             1.0, 0.5, 1.0,
//             0.5, 1.0, 0.5
//         });
//         auto csr_mat = temp.to_csr();

//         // Expected CSR outputs.
//         auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
//               0,   0,   0,
//             316, 184, 420,
//             246, 576, 330
//         });
//         auto exp_result = result_temp.to_csr();
//         auto result = SDDMM::Algo::naive_sddmm(csr_mat, X, Y);
//         ASSERT_TRUE(result == exp_result);
//     }
// }

UTEST(Matrix, Tiled_MM_Mult){
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_row_major(2, 4, {1,2,3,4,5,6,7,8});
        auto matrix2 = SDDMM::Types::Matrix::deterministic_gen_row_major(4, 2, {1,2,3,4,5,6,7,8});

        auto expected = matrix1*matrix2;
        auto result = matrix1.tmult(2, matrix2);

        ASSERT_TRUE(result.n == matrix1.n && result.m == matrix2.m);
        ASSERT_TRUE(expected == result);
    }
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_row_major(1, 1, {2});
        auto matrix2 = SDDMM::Types::Matrix::deterministic_gen_row_major(1, 1, {1});

        auto expected = matrix1*matrix2;
        auto result = matrix1.tmult(1, matrix2);

        ASSERT_TRUE(result.n == matrix1.n && result.m == matrix2.m);
        ASSERT_TRUE(expected == result);
    }
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_row_major(2, 6, {1,2,3,4,5,6,7,8,9,10,11,12});
        auto matrix2 = SDDMM::Types::Matrix::deterministic_gen_row_major(6, 2, {1,2,3,4,5,6,7,8,9,10,11,12});

        auto expected = matrix1*matrix2;
        auto result = matrix1.tmult(2, matrix2);

        ASSERT_TRUE(result.n == matrix1.n && result.m == matrix2.m);
        ASSERT_TRUE(expected == result);
    }
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 1, {1,2,3});
        auto matrix2 = SDDMM::Types::Matrix::deterministic_gen_row_major(1, 3, {1,2,3});

        auto expected = matrix1*matrix2;
        auto result = matrix1.tmult(1, matrix2);

        ASSERT_TRUE(result.n == matrix1.n && result.m == matrix2.m);
        ASSERT_TRUE(expected == result);
    }
}

UTEST(Matrix, SDDMM_tiled_op) {
    auto X = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 4, {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    });

    auto Y = SDDMM::Types::Matrix::deterministic_gen_col_major(4, 3, {
         2,  4,  6,
         8, 10, 12,
        14, 16, 18,
        20, 22, 24

    });

    auto inner_prod_res = SDDMM::Types::Matrix::deterministic_gen_row_major(3,3, {
        140, 160, 180, 
        316, 368, 420, 
        492, 576, 660
    });

    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
            0.5, 1.0, 0.5,
            1.0, 0.5, 1.0,
            0.5, 1.0, 0.5
        });
        auto csr_mat = temp.to_csr();

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
             70, 160,  90,
            316, 184, 420,
            246, 576, 330
        });
        auto exp_result = result_temp.to_csr();
        // auto result = SDDMM::Algo::naive_sddmm(csr_mat, X, Y);
        auto result = SDDMM::Algo::tiled_sddmm(csr_mat, X, Y, 8, 8, 8);
        ASSERT_TRUE(result.equals(exp_result));
    }
}

UTEST(Matrix, COO_To_Dense) {
    {
        // SDDMM::Types::vec_size_t r = 10;
        // SDDMM::Types::vec_size_t c = 10;
        // auto sparse_temp = SDDMM::Types::COO::generate_row_major(r, c, 0.6f, -1.f, 1.f, true, 250, false, true);
        SDDMM::Types::COO coo_mat;
        coo_mat.n = 5;
        coo_mat.m = 4;
        coo_mat.cols = {0,1,2,3,0};
        coo_mat.rows = {0,1,2,3,4};
        coo_mat.values = {1,2,3,4,5};
        
        SDDMM::Types::Matrix mat(5,4);
        mat.data = {
            1,0,0,0,
            0,2,0,0,
            0,0,3,0,
            0,0,0,4,
            5,0,0,0
        };

        auto dense = coo_mat.to_matrix();

        ASSERT_TRUE(dense == mat);
    }

}

UTEST(Matrix, Sparse_Mat_Gen) {
    {
        SDDMM::Types::vec_size_t r = 100;
        SDDMM::Types::vec_size_t c = 200;
        auto sparse_temp = SDDMM::Types::COO::generate_row_major<SDDMM::Types::sorted_coo_collector>(r, c, 0.1f, -1.f, 1.f, true, 250);
        std::cout << "Sparse to dense" << std::endl;
        auto dense = sparse_temp.to_matrix();
        auto coo = dense.to_coo();

        ASSERT_TRUE(coo.equals(sparse_temp));

        uint64_t dense_nnz_counter = 0;
        for(int i=0; i<dense.data.size(); ++i){
            if(dense.data[i] != 0)
                dense_nnz_counter++;
        }

        ASSERT_TRUE(dense_nnz_counter == sparse_temp.values.size());

        uint64_t nnz_counter = 0;
        for(int i=0; i<sparse_temp.values.size(); ++i){
            ASSERT_TRUE(dense(sparse_temp.rows[i], sparse_temp.cols[i]) != 0);
            nnz_counter++;
        }

        // was tested once, but it's cumbersome, so assume it works ^^
        // uint64_t n_counter = 0;
        // for(int ri=0; ri<r; ++ri){
        //     for(int ci=0; ci<c; ++ci){
        //         if(sparse_temp._nnz_locs.find({ri, ci})==sparse_temp._nnz_locs.end()){
        //             SDDMM::Types::expmt_t val = dense(ri, ci);
        //             ASSERT_TRUE(val == 0);
        //             n_counter++;
        //             // if(n_counter%10000==0) std::cout << n_counter << std::endl;
        //         }
        //     }
        // }

        // double sparsity = 1.0 - static_cast<double>(nnz_counter) / static_cast<double>(r*c);
        // std::cout << "hello world " << sparsity << " " << sparsity << " " << nnz_counter << std::endl;
    }

}

UTEST(Matrix, Sparse_Mat_Cuda_Gen) {
    {
        SDDMM::Types::vec_size_t r = 14000;
        SDDMM::Types::vec_size_t c = 14000;
        auto sparse_temp = SDDMM::Types::COO::generate_row_major_curand(r, c, 0.999f, true, 5000);
        std::cout << "Sparse to dense" << std::endl;
        auto dense = sparse_temp.to_matrix();
        auto coo = dense.to_coo();

        ASSERT_TRUE(coo.equals(sparse_temp));

        uint64_t dense_nnz_counter = 0;
        for(int i=0; i<dense.data.size(); ++i){
            if(dense.data[i] != 0)
                dense_nnz_counter++;
        }

        ASSERT_TRUE(dense_nnz_counter == sparse_temp.values.size());

        uint64_t nnz_counter = 0;
        for(int i=0; i<sparse_temp.values.size(); ++i){
            ASSERT_TRUE(dense(sparse_temp.rows[i], sparse_temp.cols[i]) != 0);
            nnz_counter++;
        }

        double sparsity = 1.0 - static_cast<double>(nnz_counter) / static_cast<double>(r*c);
        std::cout << "hello world " << std::setprecision(8) << sparsity << " " << nnz_counter << std::endl;
    }

}