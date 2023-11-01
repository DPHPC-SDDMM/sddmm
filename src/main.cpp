#include <iostream>

#include "defines.h"

#include "libs/lib_plot.hpp"

#include "data_structures/matrix/matrix.h"
#include "data_structures/csr/csr.h"
#include "data_structures/coo/coo.h"

#include "algos/sample_algo.cpp"
#include "algos/cuda_sample.cpp"

/**
 * All algos as cpp files
*/

/**
 * This is the algo that will run
*/

int main(int argc, char** argv){
#ifdef NONE
    std::cout << SDDMM::Defines::get_title_str("NONE") << std::endl;
    std::vector<expmt_t> x = matplot::linspace(0, 2 * matplot::pi);
    std::vector<expmt_t> y = matplot::transform(x, [](auto x) { return sin(x); });

    matplot::plot(x, y, "-*");
    matplot::hold(matplot::on);
    matplot::plot(x, matplot::transform(y, [](auto y) { return -y; }), "--xr");
    matplot::plot(x, matplot::transform(x, [](auto x) { return x / matplot::pi - 1.; }), "-:gs");
    matplot::plot({1.0, 0.7, 0.4, 0.0, -0.4, -0.7, -1}, "k");

    matplot::show();
#elif CUDA_SAMPLE
    std::cout << SDDMM::Defines::get_title_str("CUDA_SAMPLE") << std::endl;
    SDDMM::Algo::SampleCudaAlgo();
#elif SAMPLE_ALGO
    std::cout << SDDMM::Defines::get_title_str("SAMPLE_ALGO") << std::endl;
    SDDMM::Defines::ErrPlotData data;
    SDDMM::Algo::SampleAlgo(data, 250, 100, 500, 100);
    SDDMM::Plots::err_plot(data);

#elif CSR_COO
    SDDMM::vec_size_t n = 5;
    SDDMM::vec_size_t m = 10;

    // generate a matrix
    auto mat_before = SDDMM::Matrix::generate(n, m, 0.2);
    std::cout << mat_before;

    // convert matrix to CSR
    auto csr_mat = mat_before.to_csr();
    std::cout << csr_mat;

    // convert CSR to COO
    auto coo_mat = csr_mat.to_coo();
    std::cout << coo_mat;

    // convert COO to matrix
    auto mat_after = coo_mat.to_matrix();
    std::cout << mat_after;

    std::cout << std::endl << "Matrices " << (mat_before == mat_after ? "do " : "do not ") << "match!";


//    auto coo_matrix = SDDMM::coo::Generate(10, 10, 0.1);
//    for (int i = 0 ; i < coo_matrix.size() ; ++i)
//    {
//        auto elmnt = coo_matrix[i];
//        std::cout << std::get<0>(elmnt) << " " << std::get<1>(elmnt) << " " << std::get<2>(elmnt) << std::endl;
//    }
//
//    auto csr_matrix = SDDMM::csr::Generate(10, 10, 0.1);
//    auto row_pointers = std::get<0>(csr_matrix);
//    auto clmn_indx = std::get<1>(csr_matrix);
//    auto values = std::get<2>(csr_matrix);
//
//    std::cout << "Printing 'row pointers': " << std::endl;
//    for (int row = 0 ; row < row_pointers.size() ; ++row)
//        std::cout << row_pointers[row] << " ";
//    std::cout << std::endl;
//
//    std::cout << "Printing 'column indices': " << std::endl;
//    for (int elmnts = 0 ; elmnts < (int) 10*10*0.1 ; ++elmnts)
//        std::cout << clmn_indx[elmnts] << " ";
//    std::cout << std::endl;
//
//    std::cout << "Printing 'values': " << std::endl;
//    for (int elmnts = 0 ; elmnts < (int) 10*10*0.1 ; ++elmnts)
//        std::cout << values[elmnts] << " ";
//    std::cout << std::endl;
#endif
    return 0;
}
