#include <iostream>

#include "defines.h"

// #include "libs/lib_plot.hpp"

// #include "data_structures/matrix/matrix.h"
// #include "data_structures/csr/csr.h"
// #include "data_structures/coo/coo.h"
#include "experiments/sddmm_benchmark.cpp"

// #include <chrono>

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
#endif

#if CUDA_SAMPLE
    std::cout << SDDMM::Defines::get_title_str("CUDA_SAMPLE") << std::endl;
    SDDMM::Algo::SampleCudaAlgo();
#endif

#if SDDMM_BENCHMARK
    // auto start = std::chrono::high_resolution_clock::now();

    // auto end = std::chrono::high_resolution_clock::now();

    // int64_t duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // std::cout << duration << std::endl;
    SDDMM::Results::ExperimentInfo info(500, 500, 800, 0.1f, 200, 32);
    SDDMM::Experiments::benchmark_sddmm(info);
#endif
    return 0;
}
