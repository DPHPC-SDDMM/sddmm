#include <iostream>
#include <vector>
#include <math.h>
#include "test_helpers.hpp"
#include "../src/data_structures/matrix/matrix.h"
#include "../src/data_structures/csr/csr.h"
#include "../src/data_structures/coo/coo.h"
#include "../src/algos/naive_sddmm.cpp"
#include "../src/algos/tiled_sddmm.cpp"
#include "../src/algos/parallel_sddmm.cpp"

UTEST_MAIN();

UTEST(Matrix, TestEquals) {
    auto matrix1 = SDDMM::Types::Matrix::deterministic_gen(2, 3, {1,2,3,4,5,6});
    auto matrix2 = SDDMM::Types::Matrix::deterministic_gen(2, 3, {1,2,3,4,5,6});

    ASSERT_TRUE(matrix1 == matrix2);
}

UTEST(Matrix, TestDenseMult) {
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen(2, 3, {1,2,3,4,5,6});
        auto matrix2 = SDDMM::Types::Matrix::deterministic_gen(3, 2, {1,2,3,4,5,6});

        auto matrix3 = SDDMM::Types::Matrix::deterministic_gen(2,2, {22, 28, 49, 64});
        auto result = matrix1*matrix2;

        ASSERT_TRUE(result.n == matrix1.n && result.m == matrix2.m);
        ASSERT_TRUE(matrix3 == result);
    }
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen(1, 1, {2});
        auto matrix2 = SDDMM::Types::Matrix::deterministic_gen(1, 1, {1});

        auto matrix3 = SDDMM::Types::Matrix::deterministic_gen(1, 1, {2});
        auto result = matrix1*matrix2;

        ASSERT_TRUE(result.n == matrix1.n && result.m == matrix2.m);
        ASSERT_TRUE(matrix3 == result);
    }
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen(1, 10, {1,2,3,4,5,6,7,8,9,10});
        auto matrix2 = SDDMM::Types::Matrix::deterministic_gen(10, 1, {1,2,3,4,5,6,7,8,9,10});

        auto matrix3 = SDDMM::Types::Matrix::deterministic_gen(1, 1, {385});
        auto result = matrix1*matrix2;

        ASSERT_TRUE(result.n == matrix1.n && result.m == matrix2.m);
        ASSERT_TRUE(matrix3 == result);
    }
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen(3, 1, {1,2,3});
        auto matrix2 = SDDMM::Types::Matrix::deterministic_gen(1, 3, {1,2,3});

        auto matrix3 = SDDMM::Types::Matrix::deterministic_gen(3, 3, {1,2,3,2,4,6,3,6,9});
        auto result = matrix1*matrix2;

        ASSERT_TRUE(result.n == matrix1.n && result.m == matrix2.m);
        ASSERT_TRUE(matrix3 == result);
    }
}

UTEST(Matrix, ToCSR) {
    {
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
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
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
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
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
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
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
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
        auto matrix1 = SDDMM::Types::Matrix::deterministic_gen(0, 0, {
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
    auto dense = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
        0.5, 0.5, 0.5, 0.5,
        2.0, 2.0, 2.0, 2.0,
        2.0, 0.5, 2.0, 0.5
    });

    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9,10,11,12
        });
        auto csr_mat = temp.to_csr();

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
            0.5, 1, 1.5, 2,
            10, 12, 14, 16,
            18, 5, 22, 6 
        });
        auto exp_result = result_temp.to_csr();
        auto result = csr_mat.hadamard(dense);
        ASSERT_TRUE(result == exp_result);
    }
    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
            0, 0, 0, 0,
            5, 6, 7, 8,
            9,10,11,12
        });
        auto csr_mat = temp.to_csr();


        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
            0, 0, 0, 0,
            10, 12, 14, 16,
            18, 5, 22, 6 
        });
        auto exp_result = result_temp.to_csr();
        auto result = csr_mat.hadamard(dense);
        ASSERT_TRUE(result == exp_result);
    }
    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
            1, 2, 3, 4,
            0, 0, 0, 0,
            9,10,11,12
        });
        auto csr_mat = temp.to_csr();


        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
            0.5, 1, 1.5, 2,
            0, 0, 0, 0,
            18, 5, 22, 6 
        });
        auto exp_result = result_temp.to_csr();
        auto result = csr_mat.hadamard(dense);
        ASSERT_TRUE(result == exp_result);
    }
    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
            1, 0, 3, 0,
            5, 0, 7, 8,
            0,10, 0, 0
        });
        auto csr_mat = temp.to_csr();

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
            0.5, 0, 1.5, 0,
            10,  0, 14, 16,
             0, 5,  0, 0 
        });
        auto exp_result = result_temp.to_csr();
        auto result = csr_mat.hadamard(dense);
        ASSERT_TRUE(result == exp_result);
    }
    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        });
        auto csr_mat = temp.to_csr();

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0 
        });
        auto exp_result = result_temp.to_csr();
        auto result = csr_mat.hadamard(dense);
        ASSERT_TRUE(result == exp_result);
    }
    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen(1, 1, {
            1
        });
        auto csr_mat = temp.to_csr();

        auto dense2 = SDDMM::Types::Matrix::deterministic_gen(1, 1, {
            0.5
        });

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen(1, 1, {
            0.5 
        });
        auto exp_result = result_temp.to_csr();
        auto result = csr_mat.hadamard(dense2);
        ASSERT_TRUE(result == exp_result);
    }
}

UTEST(Matrix, Flip) {
    auto matrix1 = SDDMM::Types::Matrix::deterministic_gen(2, 3, {1,2,3,4,5,6});
    auto matrix2 = SDDMM::Types::Matrix::deterministic_gen(2, 3, {1,2,3,4,5,6});

    matrix2.to_dense_col_major();
    std::vector<SDDMM::Types::expmt_t> newVals = {1,4,2,5,3,6};
    ASSERT_TRUE(TestHelpers::compare_vectors(matrix2.data, newVals));
    ASSERT_TRUE(matrix1 == matrix2);

}

UTEST(Matrix, SDDMM_op) {
    auto X = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    });

    auto Y = SDDMM::Types::Matrix::deterministic_gen(4, 3, {
         2,  4,  6,
         8, 10, 12,
        14, 16, 18,
        20, 22, 24

    });

    // Note: for time testing purposes, do this outside of this
    // method
    Y.to_dense_col_major();

    auto inner_prod_res = SDDMM::Types::Matrix::deterministic_gen(3,3, {
        140, 160, 180, 
        316, 368, 420, 
        492, 576, 660
    });

    ASSERT_TRUE(inner_prod_res == X*Y);
    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen(3, 3, {
            0.5, 1.0, 0.5,
            1.0, 0.5, 1.0,
            0.5, 1.0, 0.5
        });
        auto csr_mat = temp.to_csr();

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen(3, 3, {
             70, 160,  90,
            316, 184, 420,
            246, 576, 330
        });
        auto exp_result = result_temp.to_csr();
        auto result = SDDMM::Algo::NaiveSDDMM(csr_mat, X, Y);
        ASSERT_TRUE(result == exp_result);
    }
    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen(3, 3, {
            0.5, 0.0, 0.5,
            0.0, 0.5, 0.0,
            0.5, 0.0, 0.5
        });
        auto csr_mat = temp.to_csr();

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen(3, 3, {
             70,   0,  90,
              0, 184,   0,
            246,   0, 330
        });
        auto exp_result = result_temp.to_csr();
        auto result = SDDMM::Algo::NaiveSDDMM(csr_mat, X, Y);
        ASSERT_TRUE(result == exp_result);
    }
    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen(3, 3, {
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        });
        auto csr_mat = temp.to_csr();

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen(3, 3, {
              0,   0,   0,
              0,   0,   0,
              0,   0,   0
        });
        auto exp_result = result_temp.to_csr();
        auto result = SDDMM::Algo::NaiveSDDMM(csr_mat, X, Y);
        ASSERT_TRUE(result == exp_result);
    }
}

UTEST(Matrix, COO_equal){
    auto matrix1 = SDDMM::Types::Matrix::deterministic_gen(2, 3, {1,2,3,4,5,6});
    auto matrix2 = SDDMM::Types::Matrix::deterministic_gen(2, 3, {1,2,3,4,5,6});
    auto matrix3 = SDDMM::Types::Matrix::deterministic_gen(2, 3, {1,2,3,4,5});

    ASSERT_TRUE(matrix1 == matrix2);
    ASSERT_TRUE(matrix1.to_coo() == matrix2.to_coo());
    ASSERT_FALSE(matrix1 == matrix3);
    ASSERT_FALSE(matrix1.to_coo() == matrix3.to_coo());
}

UTEST(Matrix, SDDMM_parallel) {
    int num_threads = 12;
    auto X = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    });

    auto Y = SDDMM::Types::Matrix::deterministic_gen(4, 3, {
         2,  4,  6,
         8, 10, 12,
        14, 16, 18,
        20, 22, 24

    });

    // Note: for time testing purposes, do this outside of this
    // method
    // Y.to_dense_col_major();

    auto inner_prod_res = SDDMM::Types::Matrix::deterministic_gen(3,3, {
        140, 160, 180, 
        316, 368, 420, 
        492, 576, 660
    });

    ASSERT_TRUE(inner_prod_res == X*Y);
    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen(3, 3, {
            0.5, 1.0, 0.5,
            1.0, 0.5, 1.0,
            0.5, 1.0, 0.5
        });
        auto coo_mat = temp.to_coo();

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen(3, 3, {
             70, 160,  90,
            316, 184, 420,
            246, 576, 330
        });
        auto exp_result = result_temp.to_coo();
        auto result = SDDMM::Algo::ParallelSDDMM(coo_mat, X, Y, num_threads);
        ASSERT_TRUE(result == exp_result);
    }
}

// UTEST(Matrix, SDDMM_giant_op) {
//     auto X = SDDMM::Types::Matrix::generate(5000, 8000);

//     auto Y = SDDMM::Types::Matrix::generate(8000, 5000);

//     // Note: for time testing purposes, do this outside of this
//     // method
//     Y.to_dense_col_major();

//     std::cout << "Finished transformation" << std::endl;

//     // Expected CSR outputs.
//     auto mat = SDDMM::Types::Matrix::generate(5000, 5000, 0.1);
//     auto csr_mat = mat.to_csr();
//     auto result = SDDMM::Algo::NaiveSDDMM(csr_mat, X, Y);

//     std::cout << "Result: " << result << std::endl;
// }

// UTEST(Matrix, SDDMM_op_zero) {
//     auto X = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
//         0,  0,  0,  0,
//         5,  6,  7,  8,
//         9, 10, 11, 12
//     });

//     auto Y = SDDMM::Types::Matrix::deterministic_gen(4, 3, {
//          2,  4,  6,
//          8, 10, 12,
//         14, 16, 18,
//         20, 22, 24

//     });

//     // Note: for time testing purposes, do this outside of this
//     // method
//     Y.to_dense_col_major();

//     auto inner_prod_res = SDDMM::Types::Matrix::deterministic_gen(3,3, {
//           0,   0,   0, 
//         316, 368, 420, 
//         492, 576, 660
//     });

//     ASSERT_TRUE(inner_prod_res == X*Y);
//     {
//         auto temp = SDDMM::Types::Matrix::deterministic_gen(3, 3, {
//             0.5, 1.0, 0.5,
//             1.0, 0.5, 1.0,
//             0.5, 1.0, 0.5
//         });
//         auto csr_mat = temp.to_csr();

//         // Expected CSR outputs.
//         auto result_temp = SDDMM::Types::Matrix::deterministic_gen(3, 3, {
//               0,   0,   0,
//             316, 184, 420,
//             246, 576, 330
//         });
//         auto exp_result = result_temp.to_csr();
//         auto result = SDDMM::Algo::NaiveSDDMM(csr_mat, X, Y);
//         ASSERT_TRUE(result == exp_result);
//     }
// }

UTEST(Matrix, SDDMM_tiled_op) {
    auto X = SDDMM::Types::Matrix::deterministic_gen(3, 4, {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    });

    auto Y = SDDMM::Types::Matrix::deterministic_gen(4, 3, {
         2,  4,  6,
         8, 10, 12,
        14, 16, 18,
        20, 22, 24

    });

    auto inner_prod_res = SDDMM::Types::Matrix::deterministic_gen(3,3, {
        140, 160, 180, 
        316, 368, 420, 
        492, 576, 660
    });

    {
        auto temp = SDDMM::Types::Matrix::deterministic_gen(3, 3, {
            0.5, 1.0, 0.5,
            1.0, 0.5, 1.0,
            0.5, 1.0, 0.5
        });
        auto csr_mat = temp.to_csr();

        // Expected CSR outputs.
        auto result_temp = SDDMM::Types::Matrix::deterministic_gen(3, 3, {
             70, 160,  90,
            316, 184, 420,
            246, 576, 330
        });
        auto exp_result = result_temp.to_csr();
        // auto result = SDDMM::Algo::NaiveSDDMM(csr_mat, X, Y);
        auto result = SDDMM::Algo::TiledSDDMM(csr_mat, X, Y, 8, 8, 8);
        ASSERT_TRUE(result == exp_result);
    }

}