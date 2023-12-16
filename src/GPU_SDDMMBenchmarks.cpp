#include <iostream>
#include "defines.h"
#include "experiments/benchmark_sddmm_gpu.cpp"

/**
 * This is the algo that will run
*/

using namespace SDDMM;

int main(int argc, char** argv) {

    int n_experiment_iterations = 2;
    int n_warmup_iterations = 3;
    Experiments::GPU_SDDMMBenchmarks::benchmark_static(
        "sparsity_K32",
        Experiments::GPU_SDDMMBenchmarks::ExperimentVariable::Sparsity,
        n_experiment_iterations, 
        n_warmup_iterations,
        "C:/sddmm_data/sparsity_large_2/K32/",
        "Compare matrices with K=32 for varying sparsity"
    );

    //Experiments::GPU_SDDMMBenchmarks::benchmark_static(
    //    "sparsity_K512",
    //    Experiments::GPU_SDDMMBenchmarks::ExperimentVariable::Sparsity,
    //    n_experiment_iterations,
    //    n_warmup_iterations,
    //    "C:/sddmm_data/sparsity_large_2/K512/",
    //    "Compare matrices with K=512 for varying sparsity"
    //);

    return 0;
}
