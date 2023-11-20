#include <iostream>
#include "experiments/sddmm_benchmark.cpp"
#include "experiments/unrolling_benchmark.cpp"
#include "experiments/cache_benchmark.cpp"
#include "experiments/unrolling_benchmark_2.cpp"

// #include <chrono>

/**
 * All algos as cpp files
*/

#include "algos/sm-l2-sddmm/sm-l2-sddmm.cpp"

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

#ifdef SM_L2
    std::cout << SDDMM::Defines::get_title_str("SM_L2_SDDMM") << std::endl;
    SDDMM::Algo::SML2SDDMM::run();
#endif

#ifdef SDDMM_BENCHMARK
    // std::cout << sizeof(SDDMM::Types::COO::triplet) << std::endl;
    // std::cout << sizeof(SDDMM::Types::expmt_t) << std::endl;
    // std::cout << sizeof(SDDMM::Types::vec_size_t) << std::endl;
    SDDMM::Results::ExperimentInfo info(
        "parallel_sddmm",
        500,  /* sparse_num_rows */
        500,  /* sparse_num_cols */
        400,  /* dense_inner_dim */
        0.1f, /* sparsity */
        25,  /* n_experiments_num */
        8    /* n_cpu_threads */
    );
    SDDMM::Experiments::benchmark_sddmm(info);
#endif

#ifdef UNROLLING_BENCHMARK
    // needs even sizes because one of the experiments divides by 2
    SDDMM::Results::SerialExperimentInfo info(
        "Serial unrolling benchmark [Release]",
        50,   /*tile_size_row            */
        30,   /*tile_size_inner          */
        60,   /*tile_size_col            */

        // 256,   /*x_num_row                */
        // 512,   /*xy_num_inner             */
        // 384,   /*y_num_col                */

        512,   /*x_num_row                */
        1024,   /*xy_num_inner            */
        384,   /*y_num_col                */

        100 /*n_experiment_iterations  */
    );
    SDDMM::Experiments::unrolling_benchmark(info);
#endif

#ifdef TYPES_BENCHMARK
    // 8192*2048 = 16777216
    SDDMM::Results::CacheExperimentInfo types_info(
        "Types benchmarks",
        0,0,8192,2048,
        50
    );
    SDDMM::Experiments::types_benchmark(types_info);
#endif

#ifdef ARRAY_VS_VEC_BENCHMARK
    SDDMM::Results::CacheExperimentInfo cache_info(
        "Vec vs PP vs C benchmarks [Release]",
        0,0,5000,5000,
        50
    );
    SDDMM::Experiments::arr_vs_vec_vs_ptr_benchmark(cache_info);
#endif

#ifdef CACHE_BENCHMARK
    SDDMM::Results::SerialExperimentInfo info(
        "Cache experiment [Release]",
        50,   /*tile_size_row            */
        30,   /*tile_size_inner          */
        60,   /*tile_size_col            */

        512,   /*x_num_row                */
        512,   /*xy_num_inner            */
        512,   /*y_num_col                */

        50 /*n_experiment_iterations  */
    );
    SDDMM::Experiments::cache_benchmark(info);
#endif

#ifdef UNROLLING_BENCHMARK_2
    // needs even sizes because one of the experiments divides by 2
    SDDMM::Results::SerialExperimentInfo info(
        "Serial unrolling benchmark 2 [Release]",
        50,   /*tile_size_row            */
        30,   /*tile_size_inner          */
        60,   /*tile_size_col            */

        // 256,   /*x_num_row                */
        // 512,   /*xy_num_inner             */
        // 384,   /*y_num_col                */

        512,   /*x_num_row                */
        1024,   /*xy_num_inner            */
        384,   /*y_num_col                */

        100 /*n_experiment_iterations  */
    );
    SDDMM::Experiments::unrolling_benchmark_2(info);
#endif

    return 0;
}
