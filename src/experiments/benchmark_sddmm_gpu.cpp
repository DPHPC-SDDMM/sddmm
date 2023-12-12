#include "../data_structures/coo/coo.h"
#include "../data_structures/csr/csr.h"
#include "../algos/cuda_sddmm/cuda_sddmm.cpp"
#include "../algos/sm-l2-sddmm/sm-l2-gpu.cuh"
#include "../algos/cusparse_sddmm/cusparse_1.cpp"
#include "../results.h"

namespace SDDMM {
    namespace Experiments {
        class GPU_SDDMMBenchmarks {
            public:
            enum class TestSubject {
                Non_Tiled_Baseline,
                cuSPARSE,
                Sm_L2
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
                    case TestSubject::Non_Tiled_Baseline:
                    data.label = "non_tiled Baseline";
                    break;
                    case TestSubject::cuSPARSE:
                    data.label = "cuSPARSE";
                    break;
                    case TestSubject::Sm_L2:
                    data.label = "sm_l2";
                    break;
                }
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, n_experiment_iterations);

                if(subject == TestSubject::Sm_L2){
                    Types::vec_size_t n_max = n_experiment_iterations+1;

                    for(Types::vec_size_t n=0; n<n_max; ++n){
                        TEXT::Gadgets::print_progress(n, n_experiment_iterations);
                        // total += SDDMM::Algo::parallel_sddmm_slow(coo_mat, X, Y, info.n_cpu_threads, &data).values[0];
                    }
                }
                else{
                    Types::vec_size_t n_max = n_experiment_iterations+1;
                    for(Types::vec_size_t n=0; n<n_max; ++n){
                        TEXT::Gadgets::print_progress(n, n_experiment_iterations);
                        
                        switch(subject){
                            case TestSubject::Non_Tiled_Baseline:
                            total += SDDMM::Algo::cuda_sddmm(coo_mat, X, Y, &data).values[0];
                            break;
                            case TestSubject::cuSPARSE:
                            total += SDDMM::Algo::cuSPARSE_SDDMM(csr_mat, X, Y, &data).values[0];
                            break;
                        }
                    }
                }

                return data;
            }
        };

        void benchmark_static(std::string experiment_name, int n_experiment_iterations, std::vector<std::string> mat_files){
            
            TEXT::Gadgets::print_colored_line(100, '=', TEXT::BRIGHT_RED);

            std::cout << TEXT::Cast::Cyan("Start measurements") << std::endl;
            // ===================================================================

            std::vector<GPU_SDDMMBenchmarks::TestSubject> subject = {
                GPU_SDDMMBenchmarks::TestSubject::Non_Tiled_Baseline,
                GPU_SDDMMBenchmarks::TestSubject::cuSPARSE,
                GPU_SDDMMBenchmarks::TestSubject::Sm_L2
            };  

            // run all tests
            std::vector<Types::expmt_t> total(subject.size(), 0.0);
            std::vector<Results::ExperimentData> results;
            
            for (auto& name : mat_files) {
                Types::COO coo_mat;
                Types::CSR csr_mat;
                Types::Matrix X(0, 0);
                Types::Matrix Y(0, 0);
                float out_sparse_sparsity;
                float X_sparsity;
                float Y_sparsity;
                Types::COO::hadamard_from_bin_file(
                    name,
                    coo_mat, csr_mat, out_sparse_sparsity,
                    X, X_sparsity,
                    Y, Y_sparsity);

                for (int i = 0; i < subject.size(); ++i) {
                    results.push_back(GPU_SDDMMBenchmarks::run(subject[i], i + 1, subject.size(), n_experiment_iterations, coo_mat, csr_mat, X, Y, total[i]));
                }
            }

            std::stringstream info;
            info << "[INFO]\n"
                 << "experiment_name " << experiment_name << "\n"
                 << "n_experiment_iterations " << n_experiment_iterations << "\n";
            int c = 1;
            for (auto& name : mat_files) {
                info << "file_" << c << " " << name << "\n";
                c++;
            }
            info << "[/INFO]";

            std::stringstream str;
            str << "iters-" << n_experiment_iterations;

            // ===================================================================
            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(experiment_name, str.str(), info.str(), results);
        }
    };
}