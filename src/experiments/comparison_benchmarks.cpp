#include <chrono>
#include <cstddef>
#include <iostream>
#ifdef __AVX2__
    // needs add_compile_options(-mavx2)
    #include <immintrin.h>
#endif
#include <limits>
#include "../results.h"
#include "../defines.h"
#include "../data_structures/matrix/matrix.h"
#include "../data_structures/coo/coo.h"
#include "../data_structures/csr/csr.h"

#include "../algos/cpu_sddmm/naive_sddmm.cpp"
#include "../algos/cpu_sddmm/parallel_sddmm.cpp"

#include "../algos/cuda_sddmm/cuda_sddmm.cpp"

// #define CHECK

namespace SDDMM {
    namespace Experiments {
        class ComparisonExperiments {
            private:

            static std::string get_cur(int cur_exp, int tot_exp){
                return std::string("..(") 
                     + std::to_string(cur_exp) 
                     + std::string("/") 
                     + std::to_string(tot_exp) 
                     + std::string(")..");
            }

            public:
            static Results::ExperimentData vary_naive(std::string what, int cur_exp, int tot_exp, Results::ExperimentInfo& info, int param, Types::Matrix A, Types::Matrix B, Types::Matrix S){
                Results::ExperimentData data;
                data.label = "vary_" + what + "_naive [" + std::to_string(param) + "]";
                
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    Types::COO s = S.to_coo();

                    auto start = std::chrono::high_resolution_clock::now();

                    Algo::naive_sddmm(s, A, B);
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }
                }

                return data;
            }

            static Results::ExperimentData vary_parallel(std::string what, int cur_exp, int tot_exp, Results::ExperimentInfo& info, int param, Types::Matrix A, Types::Matrix B, Types::Matrix S){
                Results::ExperimentData data;
                data.label = "vary_" + what + "_parallel [" + std::to_string(param) + "]";
                
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    Types::COO s = S.to_coo();

                    auto start = std::chrono::high_resolution_clock::now();

                    Algo::parallel_sddmm(s, A, B, info.n_cpu_threads);
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }
                }

                return data;
            }

            static Results::ExperimentData vary_cuda(std::string what, int cur_exp, int tot_exp, Results::ExperimentInfo& info, int param, Types::Matrix A, Types::Matrix B, Types::Matrix S){
                Results::ExperimentData data;
                data.label = "vary_" + what + "_cuda [" + std::to_string(param) + "]";
                
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    Types::COO s = S.to_coo();

                    auto start = std::chrono::high_resolution_clock::now();

                    Algo::cuda_tiled_sddmm(s, A, B);
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }
                }

                return data;
            }

        };
        
        void sparsity_benchmark(Results::ExperimentInfo& info){
            std::vector<Results::ExperimentData> results;
            
            // For now range hardcoded, add to new experiment info struct?
            for (float sparsity = 0.99; sparsity >= 0.90; sparsity -= 0.01) {
                Types::Matrix S = Types::Matrix::generate(info.dense_num_inner, info.sparse_num_col, sparsity);
                Types::Matrix A = Types::Matrix::generate(info.sparse_num_row, info.dense_num_inner, 0);
                Types::Matrix B = Types::Matrix::generate(info.dense_num_inner, info.sparse_num_col, 0);
                std::cout << "Finished matrix generation for sparsity: " << sparsity << std::endl;
                results.push_back(ComparisonExperiments::vary_naive("sparsity", 1, 3, info, 100 * sparsity, S, A, B));
                results.push_back(ComparisonExperiments::vary_parallel("sparsity", 2, 3, info, 100 * sparsity, S, A, B));
                results.push_back(ComparisonExperiments::vary_cuda("sparsity", 3, 3, info, 100 * sparsity, S, A, B));
            }

            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(info.experiment_name, info.to_string(), info.to_info(), results);
        }

        void size_benchmark(Results::ExperimentInfo& info){
            std::vector<Results::ExperimentData> results;
            
            // For now range hardcoded, add to new experiment info struct?
            for (size_t n = 1E2; n < 1E10; n *= 10) {
                Types::Matrix S = Types::Matrix::generate(n, n, info.sparsity);
                Types::Matrix A = Types::Matrix::generate(n, n, 0);
                Types::Matrix B = Types::Matrix::generate(n, n, 0);
                std::cout << "Finished matrix generation for size: " << n << std::endl;
                results.push_back(ComparisonExperiments::vary_naive("size", 1, 3, info, n, S, A, B));
                results.push_back(ComparisonExperiments::vary_parallel("size", 2, 3, info, n, S, A, B));
                results.push_back(ComparisonExperiments::vary_cuda("size", 3, 3, info, n, S, A, B));
            }

            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(info.experiment_name, info.to_string(), info.to_info(), results);
        }
    }
}