#include "../data_structures/coo/coo.h"
#include "../data_structures/csr/csr.h"
#include "../algos/cpu_sddmm/naive_sddmm.cpp"
#include "../algos/cpu_sddmm/parallel_sddmm.cpp"
#include "../algos/cuda_sddmm/cuda_sddmm.cpp"
#include "../algos/cpu_sddmm/tiled_sddmm.cpp"
#include "../results.h"

namespace SDDMM {
    namespace Experiments {
        class SDDMMBenchmarks {
            public:
            static Results::ExperimentData parallel_sddmm(
                int cur_exp, 
                int tot_exp, 
                Results::ExperimentInfo& info,
                Types::COO& coo_mat, 
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::expmt_t& total
            ){
                Results::ExperimentData data;
                data.label = "parallel_sddmm [T" + std::to_string(info.n_cpu_threads) + "]";
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                omp_set_dynamic(0);
                omp_set_num_threads(info.n_cpu_threads);
                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    total += SDDMM::Algo::parallel_sddmm(coo_mat, X, Y, info.n_cpu_threads, &data).values[0];
                }

                return data;
            }

            static Results::ExperimentData parallel_sddmm_git(
                int cur_exp, 
                int tot_exp, 
                Results::ExperimentInfo& info,
                Types::COO& coo_mat, 
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::expmt_t& total
            ){
                Results::ExperimentData data;
                data.label = "parallel_sddmm_p [T" + std::to_string(info.n_cpu_threads) + "] git version";
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                omp_set_dynamic(0);
                omp_set_num_threads(info.n_cpu_threads);
                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    total += SDDMM::Algo::parallel_sddmm_blub(coo_mat, X, Y, info.n_cpu_threads, &data);
                }

                return data;
            }

            static Results::ExperimentData parallel_sddmm_close_to_git(
                int cur_exp, 
                int tot_exp, 
                Results::ExperimentInfo& info,
                Types::COO& coo_mat, 
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::expmt_t& total
            ){
                Results::ExperimentData data;
                data.label = "parallel_sddmm_g [T" + std::to_string(info.n_cpu_threads) + "] close to git, only algo";
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                omp_set_dynamic(0);
                omp_set_num_threads(info.n_cpu_threads);
                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    total += SDDMM::Algo::parallel_sddmm_close_to_git(coo_mat, X, Y, info.n_cpu_threads, &data).values[0];
                }

                return data;
            }

            static Results::ExperimentData parallel_sddmm_causal_search_1(
                int cur_exp, 
                int tot_exp, 
                Results::ExperimentInfo& info,
                Types::COO& coo_mat, 
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::expmt_t& total
            ){
                Results::ExperimentData data;
                data.label = "parallel_sddmm_cs1 [T" + std::to_string(info.n_cpu_threads) + "] include copy in time";
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                omp_set_dynamic(0);
                omp_set_num_threads(info.n_cpu_threads);
                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    total += SDDMM::Algo::parallel_sddmm_causal_search_1(coo_mat, X, Y, info.n_cpu_threads, &data).values[0];
                }

                return data;
            }

            static Results::ExperimentData parallel_sddmm_causal_search_2(
                int cur_exp, 
                int tot_exp, 
                Results::ExperimentInfo& info,
                Types::COO& coo_mat, 
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::expmt_t& total
            ){
                Results::ExperimentData data;
                data.label = "parallel_sddmm_cs2 [T" + std::to_string(info.n_cpu_threads) + "] filter zero values";
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                omp_set_dynamic(0);
                omp_set_num_threads(info.n_cpu_threads);
                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    total += SDDMM::Algo::parallel_sddmm_causal_search_2(coo_mat, X, Y, info.n_cpu_threads, &data).values[0];
                }

                return data;
            }

            static Results::ExperimentData parallel_sddmm_causal_search_3(
                int cur_exp, 
                int tot_exp, 
                Results::ExperimentInfo& info,
                Types::COO& coo_mat, 
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::expmt_t& total
            ){
                Results::ExperimentData data;
                data.label = "parallel_sddmm_cs3 [T" + std::to_string(info.n_cpu_threads) + "] no parallel for";
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                omp_set_dynamic(0);
                omp_set_num_threads(info.n_cpu_threads);
                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    total += SDDMM::Algo::parallel_sddmm_causal_search_3(coo_mat, X, Y, info.n_cpu_threads, &data).values[0];
                }

                return data;
            }

            static Results::ExperimentData parallel_sddmm_causal_search_4(
                int cur_exp, 
                int tot_exp, 
                Results::ExperimentInfo& info,
                Types::COO& coo_mat, 
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::expmt_t& total
            ){
                Results::ExperimentData data;
                data.label = "parallel_sddmm_cs4 [T" + std::to_string(info.n_cpu_threads) + "] git with res mat creation";
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                omp_set_dynamic(0);
                omp_set_num_threads(info.n_cpu_threads);
                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    total += SDDMM::Algo::parallel_sddmm_causal_search_4(coo_mat, X, Y, info.n_cpu_threads, &data).values[0];
                }

                return data;
            }

            static Results::ExperimentData parallel_sddmm_causal_search_5(
                int cur_exp, 
                int tot_exp, 
                Results::ExperimentInfo& info,
                Types::COO& coo_mat, 
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::expmt_t& total
            ){
                Results::ExperimentData data;
                data.label = "parallel_sddmm_cs5 [T" + std::to_string(info.n_cpu_threads) + "] include everything, create res last";
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                omp_set_dynamic(0);
                omp_set_num_threads(info.n_cpu_threads);
                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    total += SDDMM::Algo::parallel_sddmm_causal_search_5(coo_mat, X, Y, info.n_cpu_threads, &data).values[0];
                }

                return data;
            }

            static Results::ExperimentData parallel_sddmm_causal_search_6(
                int cur_exp, 
                int tot_exp, 
                Results::ExperimentInfo& info,
                Types::COO& coo_mat, 
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::expmt_t& total
            ){
                Results::ExperimentData data;
                data.label = "parallel_sddmm_cs6 [T" + std::to_string(info.n_cpu_threads) + "] include everything, 4 byte vals";
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                omp_set_dynamic(0);
                omp_set_num_threads(info.n_cpu_threads);
                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    total += SDDMM::Algo::parallel_sddmm_causal_search_5(coo_mat, X, Y, info.n_cpu_threads, &data).values[0];
                }

                return data;
            }

            static Results::ExperimentData parallel_sddmm_causal_search_7(
                int cur_exp, 
                int tot_exp, 
                Results::ExperimentInfo& info,
                Types::COO& coo_mat, 
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::expmt_t& total
            ){
                Results::ExperimentData data;
                data.label = "parallel_sddmm_cs7 [T" + std::to_string(info.n_cpu_threads) + "] include everything, 8 byte vals";
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                omp_set_dynamic(0);
                omp_set_num_threads(info.n_cpu_threads);
                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    
                    total += SDDMM::Algo::parallel_sddmm_causal_search_7(coo_mat, X, Y, info.n_cpu_threads, &data).values[0];
                }

                return data;
            }

            static Results::ExperimentData naive_sddmm(
                int cur_exp, 
                int tot_exp, 
                Results::ExperimentInfo& info,
                Types::COO& coo_mat, 
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::expmt_t& total
            ){
                Results::ExperimentData data;
                data.label = "naive_sddmm";
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    // auto start = std::chrono::high_resolution_clock::now();
                    
                    total += SDDMM::Algo::naive_sddmm(coo_mat, X, Y, &data).values[0];
                    
                    // auto end = std::chrono::high_resolution_clock::now();
                    // if(n > 0){
                    //     // discard warmup
                    //     data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    // }
                }

                return data;
            }

            static Results::ExperimentData cuda_tiled_sddmm(
                int cur_exp, 
                int tot_exp, 
                Results::ExperimentInfo& info,
                Types::COO& coo_mat, 
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::expmt_t& total
            ){
                Results::ExperimentData data;
                data.label = "cuda_tiled_sddmm";
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    // auto start = std::chrono::high_resolution_clock::now();
                    
                    total += SDDMM::Algo::cuda_tiled_sddmm(coo_mat, X, Y, &data).values[0];
                    
                    // auto end = std::chrono::high_resolution_clock::now();
                    // if(n > 0){
                    //     // discard warmup
                    //     data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    // }
                }

                return data;
            }
        };

        void benchmark_sddmm(Results::ExperimentInfo& info){
            
            TEXT::Gadgets::print_colored_line(100, '=', TEXT::BRIGHT_RED);

            std::cout << TEXT::Cast::Cyan("Generate matrix") << std::endl;
            auto X = Types::Matrix::generate(info.sparse_num_row, info.dense_num_inner, 0.0);
            auto Y = Types::Matrix::generate(info.dense_num_inner, info.sparse_num_col, 0.0);
            auto sparse_mat = Types::Matrix::generate(info.sparse_num_row, info.sparse_num_col, info.sparsity);

            std::cout << TEXT::Cast::Cyan("Matrix to coo") << std::endl;
            auto coo_mat = sparse_mat.to_coo();
            std::cout << TEXT::Cast::Cyan("Matrix to csr") << std::endl;
            auto csr_mat = sparse_mat.to_csr();

            std::cout << TEXT::Cast::Cyan("Start measurements") << std::endl;
            // =========================================

            // Results::ExperimentData parallel_sddmm;
            // parallel_sddmm.label = "parallel (CPU)";
            // std::cout << TEXT::Cast::Cyan("..(1/5)..") << "parallel_sddmm ..." << std::endl;
            // TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

            // omp_set_dynamic(0);
            // for(auto x=0; x<info.n_experiment_iterations; ++x)
            // {
            //     omp_set_num_threads(info.n_cpu_threads);
            //     auto result = SDDMM::Algo::parallel_sddmm(coo_mat, X, Y, info.n_cpu_threads, &parallel_sddmm);
            //     TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);
            // }
            // // =========================================

            // Results::ExperimentData parallel_sddmm2;
            // parallel_sddmm2.label = "parallel 2 (CPU)";
            // std::cout << TEXT::Cast::Cyan("..(1/5)..") << "parallel_2_sddmm ..." << std::endl;
            // TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

            // omp_set_dynamic(0);
            // for(auto x=0; x<info.n_experiment_iterations; ++x)
            // {
            //     omp_set_num_threads(1);
            //     auto result = SDDMM::Algo::parallel_sddmm(coo_mat, X, Y, 1, &parallel_sddmm2);
            //     TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);
            // }
            // // =========================================

            // Results::ExperimentData parallel_sddmm_3;
            // parallel_sddmm_3.label = "parallel crap (CPU)";
            // std::cout << TEXT::Cast::Cyan("..(1/5)..") << "parallel_sddmm_3 ..." << std::endl;
            // TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

            // omp_set_dynamic(0);
            // for(auto x=0; x<info.n_experiment_iterations; ++x)
            // {
            //     omp_set_num_threads(info.n_cpu_threads);
            //     auto result = SDDMM::Algo::parallel_sddmm(coo_mat, X, Y, info.n_cpu_threads, &parallel_sddmm_3);
            //     TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);
            // }
            // // =========================================

            // Results::ExperimentData naive_sddmm_coo;
            // naive_sddmm_coo.label = "naive (COO,CPU)";
            // std::cout << TEXT::Cast::Cyan("..(2/5)..") << "naive_sddmm(COO) ..." << std::endl;
            // TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

            // for(auto x=0; x<info.n_experiment_iterations; ++x)
            // {
            //     auto result = SDDMM::Algo::naive_sddmm(coo_mat, X, Y, &naive_sddmm_coo);
            //     TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);
            // }
            // // =========================================

            // Results::ExperimentData naive_sddmm_csr;
            // naive_sddmm_csr.label = "naive (CSR,CPU)";
            // std::cout << TEXT::Cast::Cyan("..(3/5)..") << "naive_sddmm(CSR) ..." << std::endl;
            // for(auto x=0; x<info.n_experiment_iterations; ++x)
            // {
            //     auto result = SDDMM::Algo::naive_sddmm(csr_mat, X, Y, &naive_sddmm_csr);
            //     TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);
            // }
            // // =========================================

            // Results::ExperimentData cuda_sddmm;
            // cuda_sddmm.label = "cuda";
            // std::cout << TEXT::Cast::Cyan("..(4/5)..") << "cuda_tiled_sddmm ..." << std::endl;
            // TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

            // for(auto x=0; x<info.n_experiment_iterations; ++x)
            // {
            //     auto result = SDDMM::Algo::cuda_tiled_sddmm(coo_mat, X, Y, &cuda_sddmm);
            //     TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);
            // }
            // // =========================================

            // Results::ExperimentData tiled_sddmm;
            // tiled_sddmm.label = "tiled (CPU)";
            // std::cout << TEXT::Cast::Cyan("..(5/5)..") << "tiled_sddmm ..." << std::endl;
            // TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

            // for(auto x=0; x<info.n_experiment_iterations; ++x)
            // {
            //     auto result = SDDMM::Algo::tiled_sddmm(csr_mat, X, Y, 128, 128, 128, &tiled_sddmm);
            //     TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);
            // }
            // // =========================================

            std::vector<Types::expmt_t> total = {0,0,0,0,0,0,0};
            int i=1;
            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(info.experiment_name, info.to_string(), info.to_info(), {
                // SDDMMBenchmarks::parallel_sddmm_git(i++, total.size(), info,coo_mat, X, Y, total[i]),
                // SDDMMBenchmarks::parallel_sddmm_causal_search_4(i++, total.size(), info, coo_mat, X, Y, total[i]),
                // SDDMMBenchmarks::parallel_sddmm_close_to_git(i++, total.size(), info, coo_mat, X, Y, total[i]),
                // SDDMMBenchmarks::parallel_sddmm_causal_search_1(i++, total.size(), info, coo_mat, X, Y, total[i]),
                SDDMMBenchmarks::parallel_sddmm_causal_search_2(i++, total.size(), info, coo_mat, X, Y, total[i]),
                // SDDMMBenchmarks::parallel_sddmm_causal_search_5(i++, total.size(), info, coo_mat, X, Y, total[i]),
                SDDMMBenchmarks::parallel_sddmm_causal_search_6(i++, total.size(), info, coo_mat, X, Y, total[i]),
                SDDMMBenchmarks::parallel_sddmm_causal_search_7(i++, total.size(), info, coo_mat, X, Y, total[i]),
            });
        }
    };
}