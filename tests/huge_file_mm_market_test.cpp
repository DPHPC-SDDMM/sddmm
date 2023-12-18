#include <iostream>
#include <vector>
#include <chrono>
#include "test_helpers.hpp"
#include "../src/data_structures/matrix/matrix.h"
#include "../src/data_structures/csr/csr.h"
#include "../src/data_structures/coo/coo.h"
#include "../src/sddmm_data_gen/huge_gen.cpp"
#include "../src/experiments/benchmark_sddmm_gpu.cpp"

UTEST_MAIN();

UTEST(HugeFile, Huge_Generator){

    SDDMM::Types::vec_size_t K = 32;

    uint64_t out_size_written;
    std::string name = SDDMM::DataGenerator::huge_generator_matrix_market(
        std::string("C:/sddmm_data/"), 
        "C:/sddmm_data/data_sets/imdb/mm_market/imdb.mtx",
        //"C:\\sddmm_data\\data_sets\\patents_main\\mm_market\\patents_main.mtx",
        K, out_size_written
    );

    std::cout << "Finish create and store" << std::endl;
    std::cout << out_size_written << " bytes written to file" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    SDDMM::Types::COO out_coo;
    SDDMM::Types::CSR out_csr;
    SDDMM::Types::Matrix out_X(0,0);
    SDDMM::Types::Matrix out_Y(0,0);
    float out_sparse_sparsity;
    float out_x_sparsity;
    float out_y_sparsity;
    uint64_t out_size_read;
    SDDMM::Types::COO::hadamard_from_bin_file(
        name,
        out_coo, out_csr, out_sparse_sparsity, 
        out_X, out_x_sparsity, 
        out_Y, out_y_sparsity,
        out_size_read);
    auto stop = std::chrono::high_resolution_clock::now();

    //ASSERT_TRUE(out_coo.equals(coo_mat));
    //ASSERT_TRUE(out_csr.equals(csr_mat));
    //ASSERT_TRUE(X == out_X);
    //ASSERT_TRUE(Y == out_Y);
    //ASSERT_TRUE(out_sparse_sparsity == sparsity);
    //ASSERT_TRUE(out_x_sparsity == X_sparsity);
    //ASSERT_TRUE(out_y_sparsity == Y_sparsity);

    auto duration = std::chrono::duration_cast<SDDMM::Types::time_measure_unit>(stop - start).count();
    std::cout << "Load huge file in:\t" << static_cast<double>(duration) / 1000.0 << "ms" << std::endl;
}

//UTEST(HugeFile, TestStart) {
//    SDDMM::Experiments::GPU_SDDMMBenchmarks::benchmark_static(
//        "test", "sparsity", 3, "C://sddmm_data/small_test_data/K512/"
//    );
//}