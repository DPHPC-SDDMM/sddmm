#include "../data_structures/coo/coo.h"
#include "../data_structures/csr/csr.h"
#include "../data_structures/matrix/matrix.h"
#include "../algos/cuda_sddmm/cuda_sddmm.cpp"
#include "../algos/cpu_sddmm/parallel_cpu_sddmm.cpp"
#include "../algos/cpu_sddmm/tiled_sddmm.cpp"
#include "../results.h"

namespace SDDMM {
    namespace Experiments {
        class ExperimentTestData {
            public:
            enum class TestSubject {
                CPU_Baseline,
                GPU_Baseline,
                CPU_Parallel
            };

            static Results::ExperimentData run(
                TestSubject subject,
                int cur_exp, 
                int tot_exp,
                int n_experiment_iterations,
                Types::COO& coo_mat, 
                Types::CSR& csr_mat,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::expmt_t& total
            ){
                Results::ExperimentData data;
                switch(subject){
                    case TestSubject::GPU_Baseline:
                    data.label = "GPU Baseline";
                    break;
                    case TestSubject::CPU_Parallel:
                    data.label = "CPU Parallel";
                    break;
                    case TestSubject::CPU_Baseline:
                    data.label = "CPU Baseline";
                    break;
                }
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, n_experiment_iterations);

                Types::vec_size_t n_max = n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, n_experiment_iterations);
                        
                    switch(subject){
                        case TestSubject::GPU_Baseline:
                        total += SDDMM::Algo::cuda_sddmm(coo_mat, X, Y, &data).values[0];
                        break;
                        case TestSubject::CPU_Parallel:
                        total += SDDMM::Algo::parallel_sddmm(coo_mat, X, Y, 32, &data).values[0];
                        break;
                        case TestSubject::CPU_Baseline:
                        total += SDDMM::Algo::tiled_sddmm(csr_mat, X, Y, 8, 8, 8, &data).values[0];
                    }
                }

                return data;
            }

            static void benchmark_static(std::string experiment_name, int n_experiment_iterations) {

                TEXT::Gadgets::print_colored_line(100, '=', TEXT::BRIGHT_RED);

                std::cout << TEXT::Cast::Cyan("Start measurements") << std::endl;
                // ===================================================================

                std::vector<ExperimentTestData::TestSubject> subject = {
                    ExperimentTestData::TestSubject::CPU_Baseline,
                    ExperimentTestData::TestSubject::CPU_Parallel,
                    ExperimentTestData::TestSubject::GPU_Baseline
                };

                Types::vec_size_t N = 350;
                Types::vec_size_t M = 300;
                Types::vec_size_t K = 128;

                float sparsity = 0.1f;
                // run all tests
                for (int n = 0; n < 7; ++n) {

                    std::vector<Types::expmt_t> total(subject.size(), 0.0);
                    std::vector<Results::ExperimentData> results;

                    std::cout << TEXT::Cast::Cyan("Generate matrix") << std::endl;
                    auto X = Types::Matrix::generate_row_major(N, K, 0.0);
                    auto Y = Types::Matrix::generate_col_major(K, M, 0.0);
                    auto sparse_mat = Types::Matrix::generate_row_major(N, M, sparsity);

                    std::cout << TEXT::Cast::Cyan("Matrix to coo") << std::endl;
                    auto coo_mat = sparse_mat.to_coo();
                    auto csr_mat = sparse_mat.to_csr();

                    for (int i = 0; i < subject.size(); ++i) {
                        results.push_back(ExperimentTestData::run(subject[i], i + 1, subject.size(), n_experiment_iterations, coo_mat, csr_mat, X, Y, total[i]));
                    }

                    std::stringstream info;
                    info << "[INFO]\n"
                        << "experiment_name " << experiment_name << "\n"
                        << "variable " << "sparsity" << "\n"
                        << "N " << N << "\n"
                        << "M " << M << "\n"
                        << "K " << K << "\n"
                        << "sparsity " << sparsity << "\n";
                    info << "[/INFO]";

                    std::stringstream str;
                    str << "iters-" << n_experiment_iterations;

                    // ===================================================================
                    std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
                    Results::to_file(experiment_name, str.str(), info.str(), results, "../../results/stat_test_data/");

                    sparsity += 0.1f;
                }
            }
        };
    };
}