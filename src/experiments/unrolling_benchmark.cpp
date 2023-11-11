#include <chrono>
#include <iostream>
#include "../results.h"
#include "../defines.h"
#include "../data_structures/matrix/matrix.h"

namespace SDDMM {
    namespace Experiments {
        class UnrollingExperiments {
            public:
            static Results::ExperimentData naive(
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData simple_loop;
                simple_loop.label = "Simple 2D loop";
                std::cout << TEXT::Cast::Cyan("..(1/5)..") << "simple 2d loop ..." << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Types::Matrix r1(info.x_num_row, info.y_num_col);
                    for(auto r=0; r<info.x_num_row; ++r){
                        for(auto c=0; c<info.y_num_col; ++c){
                            for(auto i=0; i<info.xy_num_inner; ++i){
                                r1.data[r*info.y_num_col + c] += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                            }
                        }
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    simple_loop.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);
                }

                return simple_loop;
            }

            static Results::ExperimentData precalc_mult_loop(
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData precalc_mult_loop;
                precalc_mult_loop.label = "Precalc access ind";
                std::cout << TEXT::Cast::Cyan("..(2/5)..") << "precalc access ind ..." << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Types::Matrix r1(info.x_num_row, info.y_num_col);
                    for(auto r=0; r<info.x_num_row; ++r){
                        for(auto c=0; c<info.y_num_col; ++c){
                            // precalculate the access index for the target
                            Types::vec_size_t txy = r*info.y_num_col + c;
                            for(auto i=0; i<info.xy_num_inner; ++i){
                                r1.data[txy] += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                            }
                        }
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    precalc_mult_loop.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);
                }

                return precalc_mult_loop;
            }

            static Results::ExperimentData add_mult_loop(
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData add_mult_loop;
                add_mult_loop.label = "Add access ind";
                std::cout << TEXT::Cast::Cyan("..(3/5)..") << "add access ind ..." << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Types::Matrix r1(info.x_num_row, info.y_num_col);
                    Types::vec_size_t ni = 0;
                    for(auto r=0; r<info.x_num_row; ++r){
                        for(auto c=0; c<info.y_num_col; ++c){
                            // precalculate the access index for the target
                            Types::vec_size_t xyi = ni+c;
                            for(auto i=0; i<info.xy_num_inner; ++i){
                                r1.data[xyi] += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                            }
                        }
                        ni += info.y_num_col;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    add_mult_loop.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);
                }

                return add_mult_loop;
            }

            static Results::ExperimentData accum_var(
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData accum_var;
                accum_var.label = "Accum in loc var";
                std::cout << TEXT::Cast::Cyan("..(4/5)..") << "add access ind ..." << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Types::Matrix r1(info.x_num_row, info.y_num_col);
                    for(auto r=0; r<info.x_num_row; ++r){
                        for(auto c=0; c<info.y_num_col; ++c){
                            // precalculate the access index for the target
                            Types::vec_size_t txy = r*info.y_num_col + c;
                            Types::expmt_t var = 0;
                            for(auto i=0; i<info.xy_num_inner; ++i){
                                var += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                            }
                            r1.data[txy] = var;
                        }
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    accum_var.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);
                }

                return accum_var;
            }
        };
        
        void unrolling_benchmark(Results::SerialExperimentInfo& info){
            // generate data
            auto X = Types::Matrix::generate(info.x_num_row, info.xy_num_inner);
            auto Y = Types::Matrix::generate(info.xy_num_inner, info.y_num_col);

            std::vector<Results::ExperimentData> results {
                UnrollingExperiments::naive(X, Y, info),
                UnrollingExperiments::precalc_mult_loop(X, Y, info),
                UnrollingExperiments::add_mult_loop(X, Y, info)
            };

            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(info.experiment_name, info.to_string(), info.to_info(), results);
        }
    }
}