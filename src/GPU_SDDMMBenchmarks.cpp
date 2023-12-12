#include <iostream>
#include "defines.h"
#include "experiments/benchmark_sddmm_gpu.cpp"

/**
 * This is the algo that will run
*/

using namespace SDDMM;

int main(int argc, char** argv) {

    if (argc != 4) {
        TEXT::Gadgets::print_colored_line(100, '=', TEXT::GREEN);
        std::cout << std::endl;

        TEXT::Gadgets::print_colored_text_line("Usage:", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 1: experiment name (in quotes if it contains spaces)", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 2: experiment variable (N,M,K,sparsity)", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 3: path to storage (in quotes if it contains spaces)", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 4: number of iterations per experiment (like ~100)", TEXT::BLUE);
    }

    std::string experiment_name = argv[1];
    std::string experiment_var = argv[2];
    std::string folder_location = argv[3];
    int n_experiment_iterations = std::atoi(argv[4]);

    if (!(experiment_var.compare("N") == 0) ||
        experiment_var.compare("M") == 0) ||
        experiment_var.compare("K") == 0) ||
        experiment_var.compare("sparsity") == 0)
    ){
        std::cout << std::endl;
        TEXT::Gadgets::print_colored_line(100, '>', TEXT::HIGHLIGHT_YELLOW);
        TEXT::Gadgets::print_colored_text_line("Error: Param 2 must be in [N,M,K,sparsity]!", TEXT::RED);
        TEXT::Gadgets::print_colored_line(100, '<', TEXT::HIGHLIGHT_YELLOW);
        std::cout << std::endl;
    }

    Experiments::GPU_SDDMMBenchmarks::benchmark_static(
        experiment_name, experiment_var, n_experiment_iterations, folder_location);

    return 0;
}
