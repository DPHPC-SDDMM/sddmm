#include "../data_structures/coo/coo.h"
#include "../data_structures/csr/csr.h"
#include "../algos/naive_sddmm.cpp"
#include "../algos/parallel_sddmm.cpp"
#include "../algos/cuda_sddmm.cpp"
#include "../algos/tiled_sddmm.cpp"

namespace SDDMM {
    namespace EXPERIMENTS {
        void benchmark_sddmm(Types::vec_size_t sparse_s_row, Types::vec_size_t sparse_s_col, Types::vec_size_t dense_s_inner){
            std::cout << "Generate matrix" << std::endl;
            auto X = Types::Matrix::generate(sparse_s_row, dense_s_inner);
            auto Y = Types::Matrix::generate(dense_s_inner, sparse_s_col);
            auto sparse_mat = Types::Matrix::generate(sparse_s_row, sparse_s_col, 0.1);

            std::cout << "Matrix to coo" << std::endl;
            auto coo_mat = sparse_mat.to_coo();
            auto csr_mat = sparse_mat.to_csr();

            std::cout << "Start measurements" << std::endl;
            Defines::ExperimentData parallel_sddmm;
            {
                omp_set_num_threads(32);
                auto result = SDDMM::Algo::parallel_sddmm(coo_mat, X, Y, 32, &parallel_sddmm);
            }

            Defines::ExperimentData naive_sddmm_coo;
            {
                auto result = SDDMM::Algo::naive_sddmm(coo_mat, X, Y, &naive_sddmm_coo);
            }

            Defines::ExperimentData naive_sddmm_csr;
            {
                auto result = SDDMM::Algo::naive_sddmm(csr_mat, X, Y, &naive_sddmm_csr);
            }

            Defines::ExperimentData cuda_sddmm;
            {
                auto result = SDDMM::Algo::cuda_tiled_sddmm(coo_mat, X, Y, &cuda_sddmm);
            }

            Defines::ExperimentData tiled_sddmm;
            {
                auto result = SDDMM::Algo::tiled_sddmm(csr_mat, X, Y, 128, 128, 128);
            }
        }
    }
}