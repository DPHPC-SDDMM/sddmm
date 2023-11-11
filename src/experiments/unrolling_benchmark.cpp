#include <chrono>
#include <iostream>
#include "../results.h"
#include "../defines.h"
#include "../data_structures/matrix/matrix.h"

namespace SDDMM {
    namespace Experiments {
        class UnrollingExperiments {
            private:
            static std::string get_cur(int cur_exp, int tot_exp){
                return std::string("..(") 
                     + std::to_string(cur_exp) 
                     + std::string("/") 
                     + std::to_string(tot_exp) 
                     + std::string(")..");
            }
            public:
            static Results::ExperimentData naive(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Simple 2D loop";
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
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
                    data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);

                    if(x == 0){
                        assert(r1 == exp_res);
                    }
                }

                return data;
            }

            static Results::ExperimentData precalc_mult_loop(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Precalc access ind";
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
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
                    data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);

                    if(x == 0){
                        assert(r1 == exp_res);
                    }
                }

                return data;
            }

            static Results::ExperimentData add_mult_loop(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Add access ind";
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
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
                    data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);

                    if(x == 0){
                        assert(r1 == exp_res);
                    }
                }

                return data;
            }

            static Results::ExperimentData accum_var(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Accum in loc var";
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
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
                    data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);

                    if(x == 0){
                        assert(r1 == exp_res);
                    }
                }

                return data;
            }

            static Results::ExperimentData add_and_loc_acc_mult_loop(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Add access ind and local acc var";
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
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
                            Types::expmt_t var = 0;
                            for(auto i=0; i<info.xy_num_inner; ++i){
                                var += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                            }
                            r1.data[xyi] = var;
                        }
                        ni += info.y_num_col;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);

                    if(x == 0){
                        assert(r1 == exp_res);
                    }
                }

                return data;
            }

            static Results::ExperimentData add_and_loc_acc_unrol_inner_loop_mult_loop_2(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Add access ind, loc acc, inner loop unroll (2x)";
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Types::Matrix r1(info.x_num_row, info.y_num_col);
                    Types::vec_size_t ni = 0;
                    const Types::vec_size_t j = 2;
                    const Types::vec_size_t s = info.xy_num_inner-j;
                    for(auto r=0; r<info.x_num_row; ++r){
                        for(auto c=0; c<info.y_num_col; ++c){
                            // precalculate the access index for the target
                            Types::vec_size_t xyi = ni+c;
                            Types::expmt_t var = 0;
                            Types::vec_size_t i;
                            for(i=0; i<s; i+=j){
                                var += X.data[r*info.xy_num_inner + i+0] * Y.data[(i+0)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+1] * Y.data[(i+1)*info.y_num_col + c];
                            }
                            while(i<info.xy_num_inner){
                                var += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                                i++;
                            }
                            r1.data[xyi] = var;
                        }
                        ni += info.y_num_col;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);

                    if(x == 0){
                        assert(r1 == exp_res);
                    }
                }

                return data;
            }

            static Results::ExperimentData add_and_loc_acc_unrol_inner_loop_mult_loop_4(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Add access ind, loc acc, inner loop unroll (4x)";
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Types::Matrix r1(info.x_num_row, info.y_num_col);
                    Types::vec_size_t ni = 0;
                    const Types::vec_size_t j = 4;
                    const Types::vec_size_t s = info.xy_num_inner-j;
                    for(auto r=0; r<info.x_num_row; ++r){
                        for(auto c=0; c<info.y_num_col; ++c){
                            // precalculate the access index for the target
                            Types::vec_size_t xyi = ni+c;
                            Types::expmt_t var = 0;
                            Types::vec_size_t i;
                            for(i=0; i<s; i+=j){
                                var += X.data[r*info.xy_num_inner + i+0] * Y.data[(i+0)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+1] * Y.data[(i+1)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+2] * Y.data[(i+2)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+3] * Y.data[(i+3)*info.y_num_col + c];
                            }
                            while(i<info.xy_num_inner){
                                var += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                                i++;
                            }
                            r1.data[xyi] = var;
                        }
                        ni += info.y_num_col;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);

                    if(x == 0){
                        assert(r1 == exp_res);
                    }
                }

                return data;
            }

            static Results::ExperimentData add_and_loc_acc_unrol_inner_loop_mult_loop_8(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Add access ind, loc acc, inner loop unroll (8x)";
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Types::Matrix r1(info.x_num_row, info.y_num_col);
                    Types::vec_size_t ni = 0;
                    const Types::vec_size_t j = 8;
                    const Types::vec_size_t s = info.xy_num_inner-j;
                    for(auto r=0; r<info.x_num_row; ++r){
                        for(auto c=0; c<info.y_num_col; ++c){
                            // precalculate the access index for the target
                            Types::vec_size_t xyi = ni+c;
                            Types::expmt_t var = 0;
                            Types::vec_size_t i;
                            for(i=0; i<s; i+=j){
                                var += X.data[r*info.xy_num_inner + i+0] * Y.data[(i+0)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+1] * Y.data[(i+1)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+2] * Y.data[(i+2)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+3] * Y.data[(i+3)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+4] * Y.data[(i+4)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+5] * Y.data[(i+5)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+6] * Y.data[(i+6)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+7] * Y.data[(i+7)*info.y_num_col + c];
                            }
                            while(i<info.xy_num_inner){
                                var += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                                i++;
                            }
                            r1.data[xyi] = var;
                        }
                        ni += info.y_num_col;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);

                    if(x == 0){
                        assert(r1 == exp_res);
                    }
                }

                return data;
            }

            static Results::ExperimentData add_and_loc_acc_unrol_inner_loop_mult_loop_16(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Add access ind, loc acc, inner loop unroll (16x)";
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Types::Matrix r1(info.x_num_row, info.y_num_col);
                    Types::vec_size_t ni = 0;
                    const Types::vec_size_t j = 16;
                    const Types::vec_size_t s = info.xy_num_inner-j;
                    for(auto r=0; r<info.x_num_row; ++r){
                        for(auto c=0; c<info.y_num_col; ++c){
                            // precalculate the access index for the target
                            Types::vec_size_t xyi = ni+c;
                            Types::expmt_t var = 0;
                            Types::vec_size_t i;
                            for(i=0; i<s; i+=j){
                                var += X.data[r*info.xy_num_inner + i+0] * Y.data[(i+0)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+1] * Y.data[(i+1)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+2] * Y.data[(i+2)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+3] * Y.data[(i+3)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+4] * Y.data[(i+4)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+5] * Y.data[(i+5)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+6] * Y.data[(i+6)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+7] * Y.data[(i+7)*info.y_num_col + c];

                                var += X.data[r*info.xy_num_inner + i+ 8] * Y.data[(i+ 8)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+ 9] * Y.data[(i+ 9)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+10] * Y.data[(i+10)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+11] * Y.data[(i+11)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+12] * Y.data[(i+12)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+13] * Y.data[(i+13)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+14] * Y.data[(i+14)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+15] * Y.data[(i+15)*info.y_num_col + c];
                            }
                            while(i<info.xy_num_inner){
                                var += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                                i++;
                            }
                            r1.data[xyi] = var;
                        }
                        ni += info.y_num_col;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);

                    if(x == 0){
                        assert(r1 == exp_res);
                    }
                }

                return data;
            }

            static Results::ExperimentData add_and_loc_acc_unrol_inner_loop_mult_loop_32(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Add access ind, loc acc, inner loop unroll (32x)";
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Types::Matrix r1(info.x_num_row, info.y_num_col);
                    Types::vec_size_t ni = 0;
                    const Types::vec_size_t j = 32;
                    const Types::vec_size_t s = info.xy_num_inner-j;
                    for(auto r=0; r<info.x_num_row; ++r){
                        for(auto c=0; c<info.y_num_col; ++c){
                            // precalculate the access index for the target
                            Types::vec_size_t xyi = ni+c;
                            Types::expmt_t var = 0;
                            Types::vec_size_t i;
                            for(i=0; i<s; i+=j){
                                var += X.data[r*info.xy_num_inner + i+0] * Y.data[(i+0)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+1] * Y.data[(i+1)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+2] * Y.data[(i+2)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+3] * Y.data[(i+3)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+4] * Y.data[(i+4)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+5] * Y.data[(i+5)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+6] * Y.data[(i+6)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+7] * Y.data[(i+7)*info.y_num_col + c];

                                var += X.data[r*info.xy_num_inner + i+ 8] * Y.data[(i+ 8)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+ 9] * Y.data[(i+ 9)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+10] * Y.data[(i+10)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+11] * Y.data[(i+11)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+12] * Y.data[(i+12)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+13] * Y.data[(i+13)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+14] * Y.data[(i+14)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+15] * Y.data[(i+15)*info.y_num_col + c];

                                var += X.data[r*info.xy_num_inner + i+16] * Y.data[(i+16)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+17] * Y.data[(i+17)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+18] * Y.data[(i+18)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+19] * Y.data[(i+19)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+20] * Y.data[(i+20)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+21] * Y.data[(i+21)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+22] * Y.data[(i+22)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+23] * Y.data[(i+23)*info.y_num_col + c];

                                var += X.data[r*info.xy_num_inner + i+24] * Y.data[(i+24)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+25] * Y.data[(i+25)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+26] * Y.data[(i+26)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+27] * Y.data[(i+27)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+28] * Y.data[(i+28)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+29] * Y.data[(i+29)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+30] * Y.data[(i+30)*info.y_num_col + c];
                                var += X.data[r*info.xy_num_inner + i+31] * Y.data[(i+31)*info.y_num_col + c];
                            }
                            while(i<info.xy_num_inner){
                                var += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                                i++;
                            }
                            r1.data[xyi] = var;
                        }
                        ni += info.y_num_col;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);

                    if(x == 0){
                        assert(r1 == exp_res);
                    }
                }

                return data;
            }

            static Results::ExperimentData add_and_loc_acc_unrol_inner_loop_mult_loop_4_reassoc(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Add access ind, loc acc, inner loop unroll (4x), reassoc";
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Types::Matrix r1(info.x_num_row, info.y_num_col);
                    Types::vec_size_t ni = 0;
                    const Types::vec_size_t j = 4;
                    const Types::vec_size_t s = info.xy_num_inner-j;
                    for(auto r=0; r<info.x_num_row; ++r){
                        for(auto c=0; c<info.y_num_col; ++c){
                            // precalculate the access index for the target
                            Types::vec_size_t xyi = ni+c;
                            Types::expmt_t var = 0;
                            Types::vec_size_t i;
                            for(i=0; i<s; i+=j){
                                var += X.data[r*info.xy_num_inner + i+0] * Y.data[(i+0)*info.y_num_col + c]
                                       + (X.data[r*info.xy_num_inner + i+1] * Y.data[(i+1)*info.y_num_col + c]
                                          + (X.data[r*info.xy_num_inner + i+2] * Y.data[(i+2)*info.y_num_col + c]
                                             + (X.data[r*info.xy_num_inner + i+3] * Y.data[(i+3)*info.y_num_col + c])));
                            }
                            while(i<info.xy_num_inner){
                                var += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                                i++;
                            }
                            r1.data[xyi] = var;
                        }
                        ni += info.y_num_col;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);

                    if(x == 0){
                        // NOTE!!: This test will fail if we use high precision 1e-6 because
                        // floats are not associative and for this to work, we can't care about that!!!
                        // Performance measures will still work though...
                        #ifdef USE_LOW_PRECISION
                        assert(r1 == exp_res);
                        #endif
                    }
                }

                return data;
            }

            static Results::ExperimentData add_and_loc_acc_unrol_inner_loop_mult_loop_8_reassoc(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Add access ind, loc acc, inner loop unroll (8x), reassociation";
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Types::Matrix r1(info.x_num_row, info.y_num_col);
                    Types::vec_size_t ni = 0;
                    const Types::vec_size_t j = 8;
                    const Types::vec_size_t s = info.xy_num_inner-j;
                    for(auto r=0; r<info.x_num_row; ++r){
                        for(auto c=0; c<info.y_num_col; ++c){
                            // precalculate the access index for the target
                            Types::vec_size_t xyi = ni+c;
                            Types::expmt_t var = 0;
                            Types::vec_size_t i;
                            for(i=0; i<s; i+=j){
                                var += X.data[r*info.xy_num_inner + i+0] * Y.data[(i+0)*info.y_num_col + c] 
                                       + (X.data[r*info.xy_num_inner + i+1] * Y.data[(i+1)*info.y_num_col + c]
                                          + (X.data[r*info.xy_num_inner + i+2] * Y.data[(i+2)*info.y_num_col + c] 
                                             + (X.data[r*info.xy_num_inner + i+3] * Y.data[(i+3)*info.y_num_col + c] 
                                                + (X.data[r*info.xy_num_inner + i+4] * Y.data[(i+4)*info.y_num_col + c] 
                                                   + (X.data[r*info.xy_num_inner + i+5] * Y.data[(i+5)*info.y_num_col + c] 
                                                      + (X.data[r*info.xy_num_inner + i+6] * Y.data[(i+6)*info.y_num_col + c] 
                                                         + (X.data[r*info.xy_num_inner + i+7] * Y.data[(i+7)*info.y_num_col + c])))))));
                            }
                            while(i<info.xy_num_inner){
                                var += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                                i++;
                            }
                            r1.data[xyi] = var;
                        }
                        ni += info.y_num_col;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);

                    if(x == 0){
                        // NOTE!!: This test will fail if we use high precision 1e-6 because
                        // floats are not associative and for this to work, we can't care about that!!!
                        // Performance measures will still work though...
                        #ifdef USE_LOW_PRECISION
                        assert(r1 == exp_res);
                        #endif
                    }
                }

                return data;
            }

            static Results::ExperimentData add_and_loc_acc_unrol_inner_loop_mult_loop_2_sep_acc(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Add access ind, loc acc, inner loop unroll (2x), separate accumulators";
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    // Types::vec_size_t x_num_row = info.x_num_row/2;
                    // Types::vec_size_t y_num_col = info.y_num_col/2;

                    // Types::Matrix r1_eq(x_num_row, y_num_col);
                    // Types::Matrix r1_odd(x_num_row, y_num_col);
                    Types::Matrix r1(info.x_num_row, info.y_num_col);

                    Types::vec_size_t ni = 0;
                    const Types::vec_size_t j = 2;
                    const Types::vec_size_t s = info.xy_num_inner-j;
                    for(auto r=0; r<info.x_num_row; ++r){
                        for(auto c=0; c<info.y_num_col; ++c){
                            // precalculate the access index for the target
                            Types::vec_size_t xyi = ni+c;
                            Types::expmt_t var_1 = 0;
                            Types::expmt_t var_2 = 0;
                            Types::vec_size_t i;
                            for(i=0; i<s; i+=j){
                                var_1 += X.data[r*info.xy_num_inner + i+0] * Y.data[(i+0)*info.y_num_col + c];
                                var_2 += X.data[r*info.xy_num_inner + i+1] * Y.data[(i+1)*info.y_num_col + c];
                            }
                            while(i<info.xy_num_inner){
                                var_1 += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                                i++;
                            }
                            r1.data[xyi] = var_1 + var_2;
                        }
                        ni += info.y_num_col;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);

                    if(x == 0){
                        assert(r1 == exp_res);
                    }
                }

                return data;
            }

            static Results::ExperimentData add_and_loc_acc_unrol_inner_loop_mult_loop_4_sep_acc(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Add access ind, loc acc, inner loop unroll (4x), separate accumulators";
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    // Types::vec_size_t x_num_row = info.x_num_row/2;
                    // Types::vec_size_t y_num_col = info.y_num_col/2;

                    // Types::Matrix r1_eq(x_num_row, y_num_col);
                    // Types::Matrix r1_odd(x_num_row, y_num_col);
                    Types::Matrix r1(info.x_num_row, info.y_num_col);

                    Types::vec_size_t ni = 0;
                    const Types::vec_size_t j = 4;
                    const Types::vec_size_t s = info.xy_num_inner-j;
                    for(auto r=0; r<info.x_num_row; ++r){
                        for(auto c=0; c<info.y_num_col; ++c){
                            // precalculate the access index for the target
                            Types::vec_size_t xyi = ni+c;
                            Types::expmt_t var_1 = 0;
                            Types::expmt_t var_2 = 0;
                            Types::expmt_t var_3 = 0;
                            Types::expmt_t var_4 = 0;
                            Types::vec_size_t i;
                            for(i=0; i<s; i+=j){
                                var_1 += X.data[r*info.xy_num_inner + i+0] * Y.data[(i+0)*info.y_num_col + c];
                                var_2 += X.data[r*info.xy_num_inner + i+1] * Y.data[(i+1)*info.y_num_col + c];
                                var_3 += X.data[r*info.xy_num_inner + i+2] * Y.data[(i+2)*info.y_num_col + c];
                                var_4 += X.data[r*info.xy_num_inner + i+3] * Y.data[(i+3)*info.y_num_col + c];
                            }
                            while(i<info.xy_num_inner){
                                var_1 += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                                i++;
                            }
                            r1.data[xyi] = var_1 + var_2 + var_3 + var_4;
                        }
                        ni += info.y_num_col;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);

                    if(x == 0){
                        assert(r1 == exp_res);
                    }
                }

                return data;
            }

            static Results::ExperimentData add_and_loc_acc_unrol_inner_loop_mult_loop_8_sep_acc(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Add access ind, loc acc, inner loop unroll (8x), separate accumulators";
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    // Types::vec_size_t x_num_row = info.x_num_row/2;
                    // Types::vec_size_t y_num_col = info.y_num_col/2;

                    // Types::Matrix r1_eq(x_num_row, y_num_col);
                    // Types::Matrix r1_odd(x_num_row, y_num_col);
                    Types::Matrix r1(info.x_num_row, info.y_num_col);

                    Types::vec_size_t ni = 0;
                    const Types::vec_size_t j = 8;
                    const Types::vec_size_t s = info.xy_num_inner-j;
                    for(auto r=0; r<info.x_num_row; ++r){
                        for(auto c=0; c<info.y_num_col; ++c){
                            // precalculate the access index for the target
                            Types::vec_size_t xyi = ni+c;
                            Types::expmt_t var_1 = 0;
                            Types::expmt_t var_2 = 0;
                            Types::expmt_t var_3 = 0;
                            Types::expmt_t var_4 = 0;
                            Types::expmt_t var_5 = 0;
                            Types::expmt_t var_6 = 0;
                            Types::expmt_t var_7 = 0;
                            Types::expmt_t var_8 = 0;
                            Types::vec_size_t i;
                            for(i=0; i<s; i+=j){
                                var_1 += X.data[r*info.xy_num_inner + i+0] * Y.data[(i+0)*info.y_num_col + c];
                                var_2 += X.data[r*info.xy_num_inner + i+1] * Y.data[(i+1)*info.y_num_col + c];
                                var_3 += X.data[r*info.xy_num_inner + i+2] * Y.data[(i+2)*info.y_num_col + c];
                                var_4 += X.data[r*info.xy_num_inner + i+3] * Y.data[(i+3)*info.y_num_col + c];
                                var_5 += X.data[r*info.xy_num_inner + i+4] * Y.data[(i+4)*info.y_num_col + c];
                                var_6 += X.data[r*info.xy_num_inner + i+5] * Y.data[(i+5)*info.y_num_col + c];
                                var_7 += X.data[r*info.xy_num_inner + i+6] * Y.data[(i+6)*info.y_num_col + c];
                                var_8 += X.data[r*info.xy_num_inner + i+7] * Y.data[(i+7)*info.y_num_col + c];
                            }
                            while(i<info.xy_num_inner){
                                var_1 += X.data[r*info.xy_num_inner + i] * Y.data[i*info.y_num_col + c];
                                i++;
                            }
                            r1.data[xyi] = var_1 + var_2 + var_3 + var_4 + var_5 + var_6 + var_7 + var_8;
                        }
                        ni += info.y_num_col;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);

                    if(x == 0){
                        assert(r1 == exp_res);
                    }
                }

                return data;
            }
        };
        
        void unrolling_benchmark(Results::SerialExperimentInfo& info){
            // generate data
            auto X = Types::Matrix::generate(info.x_num_row, info.xy_num_inner, 0.0);
            auto Y = Types::Matrix::generate(info.xy_num_inner, info.y_num_col, 0.0);
            auto res = X*Y;

            int tot = 13;
            std::vector<Results::ExperimentData> results {
                UnrollingExperiments::naive(1, tot, X, Y, res, info),
                UnrollingExperiments::precalc_mult_loop(2, tot, X, Y, res, info),
                UnrollingExperiments::add_mult_loop(3, tot, X, Y, res, info),
                UnrollingExperiments::accum_var(4, tot, X, Y, res, info),
                UnrollingExperiments::add_and_loc_acc_mult_loop(5, tot, X, Y, res, info),
                UnrollingExperiments::add_and_loc_acc_unrol_inner_loop_mult_loop_4(6, tot, X, Y, res, info),
                UnrollingExperiments::add_and_loc_acc_unrol_inner_loop_mult_loop_8(7, tot, X, Y, res, info),
                UnrollingExperiments::add_and_loc_acc_unrol_inner_loop_mult_loop_16(8, tot, X, Y, res, info),
                UnrollingExperiments::add_and_loc_acc_unrol_inner_loop_mult_loop_32(9, tot, X, Y, res, info),
                UnrollingExperiments::add_and_loc_acc_unrol_inner_loop_mult_loop_4_reassoc(9, tot, X, Y, res, info),
                UnrollingExperiments::add_and_loc_acc_unrol_inner_loop_mult_loop_8_reassoc(10, tot, X, Y, res, info),
                UnrollingExperiments::add_and_loc_acc_unrol_inner_loop_mult_loop_2_sep_acc(11, tot, X, Y, res, info),
                UnrollingExperiments::add_and_loc_acc_unrol_inner_loop_mult_loop_4_sep_acc(12, tot, X, Y, res, info),
                UnrollingExperiments::add_and_loc_acc_unrol_inner_loop_mult_loop_8_sep_acc(13, tot, X, Y, res, info)
            };

            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(info.experiment_name, info.to_string(), info.to_info(), results);
        }
    }
}