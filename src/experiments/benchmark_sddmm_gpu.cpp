#include "../data_structures/coo/coo.h"
#include "../data_structures/csr/csr.h"
#include "../algos/cuda_sddmm/cuda_sddmm.cpp"
#include "../algos/sm-l2-sddmm/sm-l2-gpu.cuh"
#include "../results.h"

namespace SDDMM {
    namespace Experiments {
        class GPU_SDDMMBenchmarks {
            public:
            enum class TestSubject {
                Non_Tiled_Baseline,
                Tiled_Baseline,
                Sm_L2
            };

            static Results::ExperimentData run(
                TestSubject subject,
                int cur_exp, 
                int tot_exp,
                Results::ExperimentInfo& info,
                Types::COO& coo_mat, 
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::expmt_t& total
            ){
                Results::ExperimentData data;
                switch(subject){
                    case TestSubject::Non_Tiled_Baseline:
                    data.label = "non_tiled Baseline";
                    break;
                    case TestSubject::Tiled_Baseline:
                    data.label = "tiled Baseline";
                    break;
                    case TestSubject::Sm_L2:
                    data.label = "sm_l2";
                    break;
                }
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                if(subject == TestSubject::Sm_L2){
                    Types::vec_size_t n_max = info.n_experiment_iterations+1;

                    for(Types::vec_size_t n=0; n<n_max; ++n){
                        TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                        // total += SDDMM::Algo::parallel_sddmm_slow(coo_mat, X, Y, info.n_cpu_threads, &data).values[0];
                    }
                }
                else{
                    Types::vec_size_t n_max = info.n_experiment_iterations+1;
                    for(Types::vec_size_t n=0; n<n_max; ++n){
                        TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                        
                        switch(subject){
                            case TestSubject::Non_Tiled_Baseline:
                            // total += SDDMM::Algo::parallel_sddmm_git(coo_mat, X, Y, info.n_cpu_threads, &data);
                            break;
                            case TestSubject::Tiled_Baseline:
                            // total += SDDMM::Algo::parallel_sddmm(coo_mat, X, Y, info.n_cpu_threads, &data).values[0];
                            break;
                        }
                    }
                }

                return data;
            }
        };

        void benchmark_static(Results::ExperimentInfo& info){
            
            TEXT::Gadgets::print_colored_line(100, '=', TEXT::BRIGHT_RED);

            std::cout << TEXT::Cast::Cyan("Generate matrix") << std::endl;
            auto X = Types::Matrix::generate_row_major(128, 32, 0.0);
            auto Y = Types::Matrix::generate_col_major(32, 256, 0.0);
            auto sparse_mat = Types::Matrix::generate_row_major(128, 256, info.sparsity);

            std::cout << TEXT::Cast::Cyan("Matrix to coo") << std::endl;
            auto coo_mat = sparse_mat.to_coo();

            std::cout << TEXT::Cast::Cyan("Start measurements") << std::endl;
            // ===================================================================

            std::vector<GPU_SDDMMBenchmarks::TestSubject> subject = {
                GPU_SDDMMBenchmarks::TestSubject::Non_Tiled_Baseline,
                // GPU_SDDMMBenchmarks::TestSubject::Tiled_Baseline,
                GPU_SDDMMBenchmarks::TestSubject::Sm_L2
            };  

            // run all tests
            std::vector<Types::expmt_t> total(subject.size(), 0.0);
            std::vector<Results::ExperimentData> results;
            int i=1;
            for(int i=0; i<subject.size(); ++i){
                results.push_back(GPU_SDDMMBenchmarks::run(subject[i], i+1, subject.size(), info, coo_mat, X, Y, total[i]));
            }

            // ===================================================================
            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(info.experiment_name, info.to_string(), info.to_info(), results);
        }
    };
}