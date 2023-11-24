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
#include "../algos/cpu_sddmm/parallel_cpu_sddmm.cpp"
#include "../algos/cuda_sddmm/cuda_sddmm.cpp"
#include "../algos/sm-l2-sddmm/sm-l2-sddmm.cpp"

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
            static Results::ExperimentData vary_naive(std::string what, int cur_exp, int tot_exp, Results::ExperimentInfo& info, int param, Types::Matrix A, Types::Matrix B, Types::Matrix S, Types::expmt_t& total){
                Results::ExperimentData data;
                data.label = "naive [" + std::to_string(param) + "]";
                
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    Types::COO s = S.to_coo();

                    auto start = std::chrono::high_resolution_clock::now();

                    total += Algo::naive_sddmm(s, A, B).values[0];
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }
                }

                return data;
            }

            static Results::ExperimentData vary_parallel(std::string what, int cur_exp, int tot_exp, Results::ExperimentInfo& info, int param, Types::Matrix A, Types::Matrix B, Types::Matrix S, Types::expmt_t& total){
                Results::ExperimentData data;
                data.label = "parallel [" + std::to_string(param) + "]";
                
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    Types::COO s = S.to_coo();

                    auto start = std::chrono::high_resolution_clock::now();

                    total += Algo::parallel_sddmm(s, A, B, info.n_cpu_threads, &data).values[0];
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }
                }

                return data;
            }

            static Results::ExperimentData vary_cuda(std::string what, int cur_exp, int tot_exp, Results::ExperimentInfo& info, int param, Types::Matrix A, Types::Matrix B, Types::Matrix S, Types::expmt_t& total){
                Results::ExperimentData data;
                data.label = "cuda [" + std::to_string(param) + "]";
                
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    Types::COO s = S.to_coo();

                    auto start = std::chrono::high_resolution_clock::now();

                    total += Algo::cuda_tiled_sddmm(s, A, B).values[0];
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }
                }

                return data;
            }

            static Results::ExperimentData vary_sl2(std::string what, int cur_exp, int tot_exp, Results::ExperimentInfo& info, int param, Types::Matrix A, Types::Matrix B, Types::Matrix S, Types::expmt_t& total, float sparsity){
                Results::ExperimentData data;
                data.label = "sl2 [" + std::to_string(param) + "]";
                
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    Types::COO s = S.to_coo();

                    auto start = std::chrono::high_resolution_clock::now();

                    total += Algo::SML2SDDMM::run(s, sparsity, A, B, false);
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }
                }

                return data;
            }

            static Results::ExperimentData vary_sl2_2(std::string what, int cur_exp, int tot_exp, Results::ExperimentInfo& info, int param, Types::Matrix A, Types::Matrix B, Types::Matrix S, Types::expmt_t& total, float sparsity){
                Results::ExperimentData data;
                data.label = "sl2 n.p. [" + std::to_string(param) + "]";
                
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    Types::COO s = S.to_coo();

                    // auto start = std::chrono::high_resolution_clock::now();

                    auto duration = Algo::SML2SDDMM::run(s, sparsity, A, B, false);
                    
                    // auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(duration);
                    }
                }

                return data;
            }

        };
        
        void sparsity_benchmark(Results::ExperimentInfo& info){
            std::vector<Results::ExperimentData> results;
            
            Types::expmt_t total_1 = 0;
            Types::expmt_t total_2 = 0;
            Types::expmt_t total_3 = 0;
            Types::expmt_t total_4 = 0;
            Types::expmt_t total_5 = 0;

            // For now range hardcoded, add to new experiment info struct?
            for (float sparsity = 0.9; sparsity >= 0.1; sparsity -= 0.2) {
                Types::Matrix S = Types::Matrix::generate(info.dense_num_inner, info.sparse_num_col, sparsity);
                Types::Matrix A = Types::Matrix::generate(info.sparse_num_row, info.dense_num_inner, 0);
                Types::Matrix B = Types::Matrix::generate(info.dense_num_inner, info.sparse_num_col, 0);
                std::cout << "Finished matrix generation for sparsity: " << sparsity << std::endl;
                results.push_back(ComparisonExperiments::vary_naive("sparsity", 1, 3, info, 100 * sparsity, S, A, B, total_1));
                results.push_back(ComparisonExperiments::vary_parallel("sparsity", 2, 3, info, 100 * sparsity, S, A, B, total_2));
                results.push_back(ComparisonExperiments::vary_cuda("sparsity", 3, 3, info, 100 * sparsity, S, A, B, total_3));
                results.push_back(ComparisonExperiments::vary_sl2("sparsity", 3, 3, info, 100 * sparsity, S, A, B, total_4, sparsity));
                results.push_back(ComparisonExperiments::vary_sl2_2("sparsity", 3, 3, info, 100 * sparsity, S, A, B, total_5, sparsity));
            }

            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(info.experiment_name, info.to_string(), info.to_info(), results);
        }

        void size_benchmark(Results::ExperimentInfo& info){
            std::vector<Results::ExperimentData> results;
            
            Types::expmt_t total_1 = 0;
            Types::expmt_t total_2 = 0;
            Types::expmt_t total_3 = 0;
            Types::expmt_t total_4 = 0;
            Types::expmt_t total_5 = 0;

            std::vector<Types::vec_size_t> sizes = {10, 100, 1000};
            // For now range hardcoded, add to new experiment info struct?
            for (Types::vec_size_t n : sizes) {
                Types::Matrix S = Types::Matrix::generate(n, n, info.sparsity);
                Types::Matrix A = Types::Matrix::generate(n, n, 0);
                Types::Matrix B = Types::Matrix::generate(n, n, 0);
                std::cout << "Finished matrix generation for size: " << n << std::endl;
                results.push_back(ComparisonExperiments::vary_naive("size", 1, 3, info, n, S, A, B, total_1));
                results.push_back(ComparisonExperiments::vary_parallel("size", 2, 3, info, n, S, A, B, total_2));
                results.push_back(ComparisonExperiments::vary_cuda("size", 3, 3, info, n, S, A, B, total_3));
                results.push_back(ComparisonExperiments::vary_sl2("sparsity", 3, 3, info, n, S, A, B, total_4, info.sparsity));
                results.push_back(ComparisonExperiments::vary_sl2_2("sparsity", 3, 3, info, n, S, A, B, total_5, info.sparsity));
            }

            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(info.experiment_name, info.to_string(), info.to_info(), results);
        }
    }
}