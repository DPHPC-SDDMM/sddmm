#include <iostream>
#include "defines.h"
#include "experiments/benchmark_sddmm_gpu.cpp"

/**
 * This is the algo that will run
*/

using namespace SDDMM;

int main(int argc, char** argv) {

    int n_warmup_iterations = 5;
    int n_experiment_iterations = 100;
    
    // #######################################
    // large tests
    int test = atoi(argv[1]);
    if (test == 1) {
        std::cout << std::endl << "Test Nr: " << 1 << std::endl;
        TEXT::Gadgets::print_colored_line(150, '#', TEXT::HIGHLIGHT_RED);
        Experiments::GPU_SDDMMBenchmarks::benchmark_static(
            "sparsity_large_K32",
            Experiments::GPU_SDDMMBenchmarks::ExperimentVariable::Sparsity,
            n_experiment_iterations,
            n_warmup_iterations,
            "C:/sddmm_data/sparsity_large_2/K32/",
            "Compare matrices with K=32 for varying sparsity on large dataset"
        );
    }
    else if (test == 2) {
        std::cout << std::endl << "Test Nr: " << 2 << std::endl;
        TEXT::Gadgets::print_colored_line(150, '#', TEXT::HIGHLIGHT_RED);
        Experiments::GPU_SDDMMBenchmarks::benchmark_static(
            "sparsity_large_K128",
            Experiments::GPU_SDDMMBenchmarks::ExperimentVariable::Sparsity,
            n_experiment_iterations,
            n_warmup_iterations,
            "C:/sddmm_data/sparsity_large_2/K128/",
            "Compare matrices with K=128 for varying sparsity on large dataset"
        );
    }
    else if (test == 3) {
        std::cout << std::endl << "Test Nr: " << 3 << std::endl;
        TEXT::Gadgets::print_colored_line(150, '#', TEXT::HIGHLIGHT_RED);
        Experiments::GPU_SDDMMBenchmarks::benchmark_static(
            "sparsity_large_K512",
            Experiments::GPU_SDDMMBenchmarks::ExperimentVariable::Sparsity,
            n_experiment_iterations,
            n_warmup_iterations,
            "C:/sddmm_data/sparsity_large_2/K512/",
            "Compare matrices with K=512 for varying sparsity on large dataset"
        );
    }

    // #######################################
    // small tests
    else if (test == 4) {
        std::cout << std::endl << "Test Nr: " << 4 << std::endl;
        TEXT::Gadgets::print_colored_line(150, '#', TEXT::HIGHLIGHT_RED);
        Experiments::GPU_SDDMMBenchmarks::benchmark_static(
            "sparsity_small_K32",
            Experiments::GPU_SDDMMBenchmarks::ExperimentVariable::Sparsity,
            n_experiment_iterations,
            n_warmup_iterations,
            "C:/sddmm_data/sparsity_small/K32/",
            "Compare matrices with K=32 for varying sparsity on small dataset"
        );
    }
    else if (test == 5) {
        std::cout << std::endl << "Test Nr: " << 5 << std::endl;
        TEXT::Gadgets::print_colored_line(150, '#', TEXT::HIGHLIGHT_RED);
        Experiments::GPU_SDDMMBenchmarks::benchmark_static(
            "sparsity_small_K128",
            Experiments::GPU_SDDMMBenchmarks::ExperimentVariable::Sparsity,
            n_experiment_iterations,
            n_warmup_iterations,
            "C:/sddmm_data/sparsity_small/K128/",
            "Compare matrices with K=128 for varying sparsity on small dataset"
        );
    }
    else if (test == 6) {
        std::cout << std::endl << "Test Nr: " << 6 << std::endl;
        TEXT::Gadgets::print_colored_line(150, '#', TEXT::HIGHLIGHT_RED);
        Experiments::GPU_SDDMMBenchmarks::benchmark_static(
            "sparsity_small_K512",
            Experiments::GPU_SDDMMBenchmarks::ExperimentVariable::Sparsity,
            n_experiment_iterations,
            n_warmup_iterations,
            "C:/sddmm_data/sparsity_small/K512/",
            "Compare matrices with K=512 for varying sparsity on small dataset"
        );
    }

    // #######################################
    // dataset tests

    else if (test == 7) {
        std::cout << std::endl << "Test Nr: " << 7 << std::endl;
        TEXT::Gadgets::print_colored_line(150, '#', TEXT::HIGHLIGHT_RED);
        Experiments::GPU_SDDMMBenchmarks::benchmark_static(
            "imdb",
            Experiments::GPU_SDDMMBenchmarks::ExperimentVariable::K,
            n_experiment_iterations,
            n_warmup_iterations,
            "C:/sddmm_data/data_sets/imdb/",
            "Compare matrices with K=[32,128,256] for IMDB data set"
        );
    }
    else if (test == 8) {
        std::cout << std::endl << "Test Nr: " << 8 << std::endl;
        TEXT::Gadgets::print_colored_line(150, '#', TEXT::HIGHLIGHT_RED);
        Experiments::GPU_SDDMMBenchmarks::benchmark_static(
            "imdb_companion",
            Experiments::GPU_SDDMMBenchmarks::ExperimentVariable::K,
            n_experiment_iterations,
            n_warmup_iterations,
            "C:/sddmm_data/data_sets/imdb_companion/",
            "Compare matrices with K=[32,128,256] for IMDB companion"
        );
    }
    else if (test == 9) {
        std::cout << std::endl << "Test Nr: " << 9 << std::endl;
        TEXT::Gadgets::print_colored_line(150, '#', TEXT::HIGHLIGHT_RED);
        Experiments::GPU_SDDMMBenchmarks::benchmark_static(
            "patents_main",
            Experiments::GPU_SDDMMBenchmarks::ExperimentVariable::K,
            n_experiment_iterations,
            n_warmup_iterations,
            "C:/sddmm_data/data_sets/patents_main/",
            "Compare matrices with K=[32,128,256] for patents_main data set"
        );
    }
    else if (test == 10) {
        std::cout << std::endl << "Test Nr: " << 10 << std::endl;
        TEXT::Gadgets::print_colored_line(150, '#', TEXT::HIGHLIGHT_RED);
        Experiments::GPU_SDDMMBenchmarks::benchmark_static(
            "patents_main_companion",
            Experiments::GPU_SDDMMBenchmarks::ExperimentVariable::K,
            n_experiment_iterations,
            n_warmup_iterations,
            "C:/sddmm_data/data_sets/patents_main_companion/",
            "Compare matrices with K=[32,128,256] for patents_main companion"
        );
    }
    else if (test == 11) {
        std::cout << std::endl << "Test Nr: " << 11 << std::endl;
        TEXT::Gadgets::print_colored_line(150, '#', TEXT::HIGHLIGHT_RED);
        Experiments::GPU_SDDMMBenchmarks::benchmark_static(
            "patents",
            Experiments::GPU_SDDMMBenchmarks::ExperimentVariable::K,
            n_experiment_iterations,
            n_warmup_iterations,
            "C:/sddmm_data/data_sets/patents/",
            "Compare matrices with K=[32,128,256] for patents data set"
        );
    }
    else if (test == 12) {
        std::cout << std::endl << "Test Nr: " << 12 << std::endl;
        TEXT::Gadgets::print_colored_line(150, '#', TEXT::HIGHLIGHT_RED);
        Experiments::GPU_SDDMMBenchmarks::benchmark_static(
            "patents_companion",
            Experiments::GPU_SDDMMBenchmarks::ExperimentVariable::K,
            n_experiment_iterations,
            n_warmup_iterations,
            "C:/sddmm_data/data_sets/patents_companion/",
            "Compare matrices with K=[32,128,256] for patents companion"
        );
    }

    return 0;
}
