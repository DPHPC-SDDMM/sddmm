#include <iostream>
#include "defines.h"
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

#ifdef CUDA_SAMPLE
    std::cout << SDDMM::Defines::get_title_str("CUDA_SAMPLE") << std::endl;
    SDDMM::Algo::SampleCudaAlgo();
#endif

#ifdef SDDMM_BENCHMARK
    SDDMM::Results::ExperimentInfo info(
        "test_benchmark",
        500,  /* sparse_num_rows */
        500,  /* sparse_num_cols */
        800,  /* dense_inner_dim */
        0.1f, /* sparsity */
        50,  /* n_experiments_num */
        24    /* n_cpu_threads */
    );
    SDDMM::Experiments::benchmark_sddmm(info);
#endif
    return 0;
}
