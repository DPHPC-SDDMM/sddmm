#include <chrono>
#include <iostream>
#ifdef __AVX2__
    // needs add_compile_options(-mavx2)
    #include <immintrin.h>
#endif
#include <limits>
#include "../results.h"
#include "../defines.h"
#include "../data_structures/matrix/matrix.h"
#include "../algos/cuda_mat_mult/cuda_tiled_mat_mult.cpp"

// #define CHECK

namespace SDDMM {
    namespace Experiments {
        class GpuCacheExperiments {
            public:
            static Results::ExperimentData cuda_mat_mul(
                int cur_exp, 
                int tot_exp,
                Types::vec_size_t n_experiment_iterations,
                Types::Matrix& X, 
                Types::Matrix& Y,
                Types::vec_size_t ts,
                Types::expmt_t& total
            ){
                Results::ExperimentData data;
                data.label = "ts" + std::to_string(ts);
                
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, n_experiment_iterations);

                Types::vec_size_t n_max = n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, n_experiment_iterations);
                    if(n>0){
                        total += SDDMM::Algo::cuda_tiled_mat_mult(X, Y, ts, &data).data[0];
                    }
                    else{
                        auto res = SDDMM::Algo::cuda_tiled_mat_mult(X, Y, ts, nullptr);
                        auto expected = X*Y;
                        assert(res == expected);
                        total += res.data[0];
                    }
                }

                return data;
            }
        };

        void benchmark_gpu_cache(std::string experiment_name, Types::vec_size_t n_experiment_iterations){
            // generate data

            Types::vec_size_t x_num_row = 128;
            Types::vec_size_t xy_num_inner = 128;
            Types::vec_size_t y_num_col = 128;

            Types::Matrix X = Types::Matrix::generate_row_major(x_num_row, xy_num_inner, 0.0);
            Types::Matrix Y = Types::Matrix::generate_row_major(xy_num_inner, y_num_col, 0.0);

            std::cout << "Finished matrix generation" << std::endl;

            std::vector<Results::ExperimentData> results;
            Types::vec_size_t ts = 2;
            Types::vec_size_t i=1;
            Types::vec_size_t tot = 0;
            while(ts <= xy_num_inner){
                tot++;
                ts = 2*ts;
            }
            ts = 4;
            Types::vec_size_t ts_min = ts;
            Types::expmt_t* total = new Types::expmt_t[tot];
            while(ts <= xy_num_inner){
                results.push_back(GpuCacheExperiments::cuda_mat_mul(
                    i, tot, n_experiment_iterations, X, Y, ts, total[i-1])
                );
                i++;
                ts = 2*ts;
            }
            Types::vec_size_t ts_max = ts;

            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            std::stringstream to_string;
            to_string << "_xrNum" << x_num_row
                      << "_xyiNum" << xy_num_inner
                      << "_ycNum" << y_num_col
                      << "_nIt" << n_experiment_iterations;
            std::stringstream to_info;
            to_info << "[INFO]\n"
                    << "experiment_name " << experiment_name << "\n"
                    << "tile_size [" << ts_min << "," << ts_max << "]" << "\n"
                    << "xr_num " << x_num_row << "\n"
                    << "xyi_num " << xy_num_inner << "\n"
                    << "yc_num " << y_num_col << "\n"
                    << "n_experiment_iterations " << n_experiment_iterations << "\n"
                    << "[/INFO]";

            Results::to_file(experiment_name, to_string.str(), to_info.str(), results);
        }
        
    }
}