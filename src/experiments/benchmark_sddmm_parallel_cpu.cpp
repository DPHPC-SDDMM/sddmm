#include "../data_structures/coo/coo.h"
#include "../data_structures/csr/csr.h"
#include "../algos/cpu_sddmm/parallel_cpu_sddmm.cpp"
#include "../results.h"

namespace SDDMM {
    namespace Experiments {
        class CPU_SDDMMBenchmarks {
            public:
            enum class TestSubject {
                Baseline,
                Slow,
                Fast
            };

            static Results::ExperimentData parallel_cpu_sddmm(
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
                    case TestSubject::Baseline:
                    data.label = "parallel_cpu_sddmm Baseline [T" + std::to_string(info.n_cpu_threads) + "]";
                    break;
                    case TestSubject::Fast:
                    data.label = "parallel_cpu_sddmm Fast [T" + std::to_string(info.n_cpu_threads) + "]";
                    break;
                    case TestSubject::Slow:
                    data.label = "parallel_cpu_sddmm Slow [T" + std::to_string(info.n_cpu_threads) + "]";
                    break;
                }
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                omp_set_dynamic(0);
                omp_set_num_threads(info.n_cpu_threads);
                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    switch(subject){
                        case TestSubject::Baseline:
                        total += SDDMM::Algo::parallel_sddmm_git(coo_mat, X, Y, info.n_cpu_threads, &data);
                        break;
                        case TestSubject::Fast:
                        total += SDDMM::Algo::parallel_sddmm(coo_mat, X, Y, info.n_cpu_threads, &data).values[0];
                        break;
                        case TestSubject::Slow:
                        total += SDDMM::Algo::parallel_sddmm_slow(coo_mat, X, Y, info.n_cpu_threads, &data).values[0];
                        break;
                    }
                }

                return data;
            }
        };

        void benchmark_parallel_cpu_sddmm(Results::ExperimentInfo& info){
            
            TEXT::Gadgets::print_colored_line(100, '=', TEXT::BRIGHT_RED);

            std::cout << TEXT::Cast::Cyan("Generate matrix") << std::endl;
            auto X = Types::Matrix::generate_row_major(info.sparse_num_row, info.dense_num_inner, 0.0);
            auto Y = Types::Matrix::generate_col_major(info.dense_num_inner, info.sparse_num_col, 0.0);
            auto sparse_mat = Types::Matrix::generate_row_major(info.sparse_num_row, info.sparse_num_col, info.sparsity);

            std::cout << TEXT::Cast::Cyan("Matrix to coo") << std::endl;
            auto coo_mat = sparse_mat.to_coo();

            std::cout << TEXT::Cast::Cyan("Start measurements") << std::endl;
            // ===================================================================

            std::vector<CPU_SDDMMBenchmarks::TestSubject> subject = {
                CPU_SDDMMBenchmarks::TestSubject::Baseline,
                CPU_SDDMMBenchmarks::TestSubject::Slow,
                CPU_SDDMMBenchmarks::TestSubject::Fast
            };  

            // run all tests
            std::vector<Types::expmt_t> total(subject.size(), 0.0);
            std::vector<Results::ExperimentData> results;
            int i=1;
            for(int i=0; i<subject.size(); ++i){
                results.push_back(CPU_SDDMMBenchmarks::parallel_cpu_sddmm(subject[i], i+1, subject.size(), info, coo_mat, X, Y, total[i]));
            }

            // ===================================================================
            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(info.experiment_name, info.to_string(), info.to_info(), results);
        }
    };
}