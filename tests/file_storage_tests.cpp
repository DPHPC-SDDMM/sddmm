#include <iostream>
#include <vector>
#include <chrono>
#include "test_helpers.hpp"
#include "../src/data_structures/matrix/matrix.h"
#include "../src/data_structures/csr/csr.h"
#include "../src/data_structures/coo/coo.h"

UTEST_MAIN();

UTEST(FileStorage, Generator_to_file){
    std::string target_folder = ".";
    target_folder += SDDMM::Defines::path_separator;
    for(int i=1; i<100; i*=10)
    {
        auto X = SDDMM::Types::Matrix::generate_row_major(i*5, i*8, 0.0f);
        auto Y = SDDMM::Types::Matrix::generate_col_major(i*8, i*5, 0.0f);
        auto mat = SDDMM::Types::Matrix::generate_row_major(i*5, i*5, 0.1f);
        auto coo_mat = mat.to_coo();
        auto csr_mat = mat.to_csr();
        auto exp_result = coo_mat.hadamard(X*Y);

        std::string name = SDDMM::Types::COO::hadamard_to_bin_file(
            target_folder, coo_mat, 0.1f, X, 0.0f, Y, 0.0f);

        SDDMM::Types::COO out_coo;
        SDDMM::Types::CSR out_csr;
        SDDMM::Types::Matrix out_X(0,0);
        SDDMM::Types::Matrix out_Y(0,0);

        float out_sparse_sparsity;
        float out_x_sparsity;
        float out_y_sparsity;
        SDDMM::Types::COO::hadamard_from_bin_file(
            target_folder + name,
            out_coo, out_csr, out_sparse_sparsity, 
            out_X, out_x_sparsity, 
            out_Y, out_y_sparsity);

        ASSERT_TRUE(out_coo.equals(coo_mat));
        ASSERT_TRUE(out_csr.equals(csr_mat));
        ASSERT_TRUE(X == out_X);
        ASSERT_TRUE(Y == out_Y);

        std::remove((target_folder + name).c_str());
    }
}

UTEST(FileStorage, Tiny_Matrix_Write){
    std::string target_folder = ".";
    target_folder += SDDMM::Defines::path_separator;

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

    auto temp = SDDMM::Types::Matrix::deterministic_gen_row_major(3, 3, {
        0.5, 1.0, 0.5,
        1.0, 0.5, 1.0,
        0.5, 1.0, 0.5
    });
    auto csr_mat = temp.to_csr();
    auto coo_mat = temp.to_coo();

    float X_sparsity = 0.0f;
    float Y_sparsity = 0.0f;
    float sparsity = 0.1f;

    std::string name = SDDMM::Types::COO::hadamard_to_bin_file(
        target_folder, coo_mat, sparsity, X, X_sparsity, Y, Y_sparsity);

    SDDMM::Types::COO out_coo;
    SDDMM::Types::CSR out_csr;
    SDDMM::Types::Matrix out_X(0,0);
    SDDMM::Types::Matrix out_Y(0,0);

    float out_sparse_sparsity;
    float out_x_sparsity;
    float out_y_sparsity;
    SDDMM::Types::COO::hadamard_from_bin_file(
        target_folder + name,
        out_coo, out_csr, out_sparse_sparsity, 
        out_X, out_x_sparsity, 
        out_Y, out_y_sparsity);

    ASSERT_TRUE(out_coo.equals(coo_mat));
    ASSERT_TRUE(out_csr.equals(csr_mat));
    ASSERT_TRUE(X == out_X);
    ASSERT_TRUE(Y == out_Y);
    ASSERT_TRUE(out_sparse_sparsity == sparsity);
    ASSERT_TRUE(out_x_sparsity == X_sparsity);
    ASSERT_TRUE(out_y_sparsity == Y_sparsity);

    std::remove((target_folder + name).c_str());
}

UTEST(FileStorage, Giant_Matrix_Write){
    std::string target_folder = ".";
    target_folder += SDDMM::Defines::path_separator;

    std::vector<std::chrono::high_resolution_clock::time_point> ts;
    std::vector<std::string> ts_labels;

    // max for N and M: 8388608
    SDDMM::Types::vec_size_t N = 320*32; //==10240
    SDDMM::Types::vec_size_t M = 400*32; //==12800
    SDDMM::Types::vec_size_t K = 512;

    ts.push_back(std::chrono::high_resolution_clock::now());
    ts_labels.push_back("start");
    float X_sparsity = 0.11145f;
    auto X = SDDMM::Types::Matrix::generate_row_major(N, K, X_sparsity);
    ts.push_back(std::chrono::high_resolution_clock::now());
    ts_labels.push_back("generate X");

    float Y_sparsity = 0.2f;
    auto Y = SDDMM::Types::Matrix::generate_col_major(K, M, Y_sparsity);
    ts.push_back(std::chrono::high_resolution_clock::now());
    ts_labels.push_back("generate Y");

    float sparsity = 0.7f;
    auto mat = SDDMM::Types::Matrix::generate_row_major(N, M, sparsity);
    ts.push_back(std::chrono::high_resolution_clock::now());
    ts_labels.push_back("generate S mat");

    auto coo_mat = mat.to_coo();
    ts.push_back(std::chrono::high_resolution_clock::now());
    ts_labels.push_back("to coo_mat");

    auto csr_mat = mat.to_csr();
    ts.push_back(std::chrono::high_resolution_clock::now());
    ts_labels.push_back("to csr_mat");

    std::string name = SDDMM::Types::COO::hadamard_to_bin_file(
        target_folder, coo_mat, sparsity, X, X_sparsity, Y, Y_sparsity);
    ts.push_back(std::chrono::high_resolution_clock::now());
    ts_labels.push_back("to file");

    SDDMM::Types::COO out_coo;
    SDDMM::Types::CSR out_csr;
    SDDMM::Types::Matrix out_X(0,0);
    SDDMM::Types::Matrix out_Y(0,0);

    float out_sparse_sparsity;
    float out_x_sparsity;
    float out_y_sparsity;
    SDDMM::Types::COO::hadamard_from_bin_file(
        target_folder + name,
        out_coo, out_csr, out_sparse_sparsity, 
        out_X, out_x_sparsity, 
        out_Y, out_y_sparsity);
    ts.push_back(std::chrono::high_resolution_clock::now());
    ts_labels.push_back("from file");

    ASSERT_TRUE(out_coo.equals(coo_mat));
    ASSERT_TRUE(out_csr.equals(csr_mat));
    ASSERT_TRUE(X == out_X);
    ASSERT_TRUE(Y == out_Y);
    ASSERT_TRUE(out_sparse_sparsity == sparsity);
    ASSERT_TRUE(out_x_sparsity == X_sparsity);
    ASSERT_TRUE(out_y_sparsity == Y_sparsity);
    ts.push_back(std::chrono::high_resolution_clock::now());
    ts_labels.push_back("compare");

    for(int i=1; i<ts.size(); ++i){
        auto duration = std::chrono::duration_cast<SDDMM::Types::time_measure_unit>(ts[i]-ts[i-1]).count();
        std::cout << ts_labels[i] << ":\t" << static_cast<double>(duration) / 1000.0 << "ms" << std::endl;
    }

    std::remove((target_folder + name).c_str());
}
