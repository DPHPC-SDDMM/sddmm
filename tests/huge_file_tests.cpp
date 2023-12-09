#include <iostream>
#include <vector>
#include <chrono>
#include "test_helpers.hpp"
#include "../src/data_structures/matrix/matrix.h"
#include "../src/data_structures/csr/csr.h"
#include "../src/data_structures/coo/coo.h"
#include "../src/sddmm_data_gen/huge_gen.cpp"

UTEST_MAIN();

UTEST(HugeFile, Huge_Generator){
    std::string name = SDDMM::DataGenerator::huge_generator(
        std::string(".") + SDDMM::Defines::path_separator,
        512, 3000000000, 3000000000, 0.9999f
    );

    std::cout << "Finish create and store" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    SDDMM::Types::COO out_coo;
    SDDMM::Types::CSR out_csr;
    SDDMM::Types::Matrix out_X(0,0);
    SDDMM::Types::Matrix out_Y(0,0);
    float out_sparse_sparsity;
    float out_x_sparsity;
    float out_y_sparsity;
    SDDMM::Types::COO::hadamard_from_bin_file(
        name,
        out_coo, out_csr, out_sparse_sparsity, 
        out_X, out_x_sparsity, 
        out_Y, out_y_sparsity);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<SDDMM::Types::time_measure_unit>(stop - start).count();
    std::cout << "Load huge file in:\t" << static_cast<double>(duration) / 1000.0 << "ms" << std::endl;
}
