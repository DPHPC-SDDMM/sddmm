#include "../data_structures/coo/coo.h"
#include "../data_structures/csr/csr.h"
#include "../algos/cpu_sddmm/naive_sddmm.cpp"
#include "../algos/cpu_sddmm/parallel_sddmm.cpp"
#include "../algos/cuda_sddmm/cuda_sddmm.cpp"
#include "../algos/cpu_sddmm/tiled_sddmm.cpp"

namespace SDDMM {
    class Experiments {
        public:
        static void benchmark_sddmm(Results::ExperimentInfo& info){
            
            TEXT::Gadgets::print_line(100, TEXT::BRIGHT_RED);

            std::cout << TEXT::Cast::Cyan("Generate matrix") << std::endl;
            auto X = Types::Matrix::generate(info.sparse_num_row, info.dense_num_inner);
            auto Y = Types::Matrix::generate(info.dense_num_inner, info.sparse_num_col);
            auto sparse_mat = Types::Matrix::generate(info.sparse_num_row, info.sparse_num_col, info.sparsity);

            std::cout << TEXT::Cast::Cyan("Matrix to coo") << std::endl;
            auto coo_mat = sparse_mat.to_coo();
            std::cout << TEXT::Cast::Cyan("Matrix to csr") << std::endl;
            auto csr_mat = sparse_mat.to_csr();

            std::cout << TEXT::Cast::Cyan("Start measurements") << std::endl;
            // =========================================

            Results::ExperimentData parallel_sddmm;
            parallel_sddmm.label = "parallel (CPU)";
            std::cout << TEXT::Cast::Cyan("..(1/5)..") << "parallel_sddmm ..." << std::endl;
            TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

            for(auto x=0; x<info.n_experiment_iterations; ++x)
            {
                omp_set_num_threads(32);
                auto result = SDDMM::Algo::parallel_sddmm(coo_mat, X, Y, info.n_cpu_threads, &parallel_sddmm);
                TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);
            }
            // =========================================

            Results::ExperimentData naive_sddmm_coo;
            naive_sddmm_coo.label = "naive (COO,CPU)";
            std::cout << TEXT::Cast::Cyan("..(2/5)..") << "naive_sddmm(COO) ..." << std::endl;
            TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

            for(auto x=0; x<info.n_experiment_iterations; ++x)
            {
                auto result = SDDMM::Algo::naive_sddmm(coo_mat, X, Y, &naive_sddmm_coo);
                TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);
            }
            // =========================================

            Results::ExperimentData naive_sddmm_csr;
            naive_sddmm_csr.label = "naive (CSR,CPU)";
            std::cout << TEXT::Cast::Cyan("..(3/5)..") << "parallel_sddmm(CSR) ..." << std::endl;
            for(auto x=0; x<info.n_experiment_iterations; ++x)
            {
                auto result = SDDMM::Algo::naive_sddmm(csr_mat, X, Y, &naive_sddmm_csr);
                TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);
            }
            // =========================================

            Results::ExperimentData cuda_sddmm;
            cuda_sddmm.label = "cuda";
            std::cout << TEXT::Cast::Cyan("..(4/5)..") << "cuda_tiled_sddmm ..." << std::endl;
            TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

            for(auto x=0; x<info.n_experiment_iterations; ++x)
            {
                auto result = SDDMM::Algo::cuda_tiled_sddmm(coo_mat, X, Y, &cuda_sddmm);
                TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);
            }
            // =========================================

            Results::ExperimentData tiled_sddmm;
            tiled_sddmm.label = "tiled (CPU)";
            std::cout << TEXT::Cast::Cyan("..(5/5)..") << "tiled_sddmm ..." << std::endl;
            TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

            for(auto x=0; x<info.n_experiment_iterations; ++x)
            {
                auto result = SDDMM::Algo::tiled_sddmm(csr_mat, X, Y, 128, 128, 128, &tiled_sddmm);
                TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);
            }
            // =========================================

            Results::to_file(info, {
                parallel_sddmm,
                naive_sddmm_coo,
                naive_sddmm_csr,
                cuda_sddmm,
                tiled_sddmm
            });
        }
    };
}