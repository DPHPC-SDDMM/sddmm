#include <chrono>
#include <iostream>
#ifdef __AVX2__
    // needs add_compile_options(-mavx2)
    #include <immintrin.h>
#endif
#include "../results.h"
#include "../defines.h"
#include "../data_structures/matrix/matrix.h"

// #define CHECK

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

#ifdef CHECK
                    if(x == 0){
                        assert(r1 == exp_res);
                    }
#endif
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

#ifdef CHECK
                    if(x == 0){
                        assert(r1 == exp_res);
                    }
#endif
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

#ifdef CHECK
                    if(x == 0){
                        assert(r1 == exp_res);
                    }
#endif
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

#ifdef CHECK
                    if(x == 0){
                        assert(r1 == exp_res);
                    }
#endif
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

#ifdef CHECK
                    if(x == 0){
                        assert(r1 == exp_res);
                    }
#endif
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

#ifdef CHECK
                    if(x == 0){
                        assert(r1 == exp_res);
                    }
#endif
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

#ifdef CHECK
                    if(x == 0){
                        assert(r1 == exp_res);
                    }
#endif
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

#ifdef CHECK
                    if(x == 0){
                        assert(r1 == exp_res);
                    }
#endif
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

#ifdef CHECK
                    if(x == 0){
                        assert(r1 == exp_res);
                    }
#endif
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

#ifdef CHECK
                    if(x == 0){
                        assert(r1 == exp_res);
                    }
#endif
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

#ifdef CHECK
                    if(x == 0){
                        // NOTE!!: This test will fail if we use high precision 1e-6 because
                        // floats are not associative and for this to work, we can't care about that!!!
                        // Performance measures will still work though...
                        #ifdef USE_LOW_PRECISION
                        assert(r1 == exp_res);
                        #endif
                    }
#endif
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

#ifdef CHECK
                    if(x == 0){
                        assert(r1 == exp_res);
                    }
#endif
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

#ifdef CHECK
                    if(x == 0){
                        assert(r1 == exp_res);
                    }
#endif
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

#ifdef CHECK
                    if(x == 0){
                        assert(r1 == exp_res);
                    }
#endif
                }

                return data;
            }

            static Results::ExperimentData add_and_loc_acc_prec_in_loop_unrol_inner_loop_mult_loop_8_sep_acc(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Add access ind, precalc inner loop, loc acc, inner loop unroll (8x), separate accumulators";
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
                                Types::vec_size_t i0 = i+0;
                                Types::vec_size_t i1 = i+1;
                                Types::vec_size_t i2 = i+2;
                                Types::vec_size_t i3 = i+3;
                                Types::vec_size_t i4 = i+4;
                                Types::vec_size_t i5 = i+5;
                                Types::vec_size_t i6 = i+6;
                                Types::vec_size_t i7 = i+7;
                                var_1 += X.data[r*info.xy_num_inner + i0] * Y.data[i0*info.y_num_col + c];
                                var_2 += X.data[r*info.xy_num_inner + i1] * Y.data[i1*info.y_num_col + c];
                                var_3 += X.data[r*info.xy_num_inner + i2] * Y.data[i2*info.y_num_col + c];
                                var_4 += X.data[r*info.xy_num_inner + i3] * Y.data[i3*info.y_num_col + c];
                                var_5 += X.data[r*info.xy_num_inner + i4] * Y.data[i4*info.y_num_col + c];
                                var_6 += X.data[r*info.xy_num_inner + i5] * Y.data[i5*info.y_num_col + c];
                                var_7 += X.data[r*info.xy_num_inner + i6] * Y.data[i6*info.y_num_col + c];
                                var_8 += X.data[r*info.xy_num_inner + i7] * Y.data[i7*info.y_num_col + c];
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

#ifdef CHECK
                    if(x == 0){
                        assert(r1 == exp_res);
                    }
#endif
                }

                return data;
            }

            static Results::ExperimentData add_vectorized(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info
            ) {
                Results::ExperimentData data;
                data.label = "Add Vectorized m256";
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Types::Matrix r1(info.x_num_row, info.y_num_col);
                    Types::vec_size_t ni = 0;
                    const Types::vec_size_t j = 8;
                    const Types::vec_size_t st = info.xy_num_inner;
                    const Types::vec_size_t s = st;
                    Types::expmt_t cache[8];
                    for(auto r=0; r<info.x_num_row; ++r){
                        for(auto c=0; c<info.y_num_col; ++c){
                            // precalculate the access index for the target
                            Types::vec_size_t xyi = ni+c;
                            Types::vec_size_t i;
                            __m256 var = _mm256_setzero_ps();
                            for(i=0; i<s; i+=j){
                                Types::vec_size_t i7 = i+7;
                                Types::vec_size_t i6 = i+6;
                                Types::vec_size_t i5 = i+5;
                                Types::vec_size_t i4 = i+4;
                                Types::vec_size_t i3 = i+3;
                                Types::vec_size_t i2 = i+2;
                                Types::vec_size_t i1 = i+1;
                                Types::vec_size_t i0 = i+0;

                                bool st7 = st > i7;
                                bool st6 = st > i6;
                                bool st5 = st > i5;
                                bool st4 = st > i4;
                                bool st3 = st > i3;
                                bool st2 = st > i2;
                                bool st1 = st > i1;
                                bool st0 = st > i0;

                                __m256 x = _mm256_set_ps(
                                    st7 ? X.data[r*info.xy_num_inner + i7] : 0,
                                    st6 ? X.data[r*info.xy_num_inner + i6] : 0,
                                    st5 ? X.data[r*info.xy_num_inner + i5] : 0,
                                    st4 ? X.data[r*info.xy_num_inner + i4] : 0,
                                    st3 ? X.data[r*info.xy_num_inner + i3] : 0,
                                    st2 ? X.data[r*info.xy_num_inner + i2] : 0,
                                    st1 ? X.data[r*info.xy_num_inner + i1] : 0,
                                    st0 ? X.data[r*info.xy_num_inner + i0] : 0
                                );
                                __m256 y = _mm256_set_ps(
                                    st7 ? Y.data[i7*info.y_num_col + c] : 0,
                                    st6 ? Y.data[i6*info.y_num_col + c] : 0,
                                    st5 ? Y.data[i5*info.y_num_col + c] : 0,
                                    st4 ? Y.data[i4*info.y_num_col + c] : 0,
                                    st3 ? Y.data[i3*info.y_num_col + c] : 0,
                                    st2 ? Y.data[i2*info.y_num_col + c] : 0,
                                    st1 ? Y.data[i1*info.y_num_col + c] : 0,
                                    st0 ? Y.data[i0*info.y_num_col + c] : 0
                                );
                                var = _mm256_add_ps(var, _mm256_mul_ps(x,y));
                            }
                            _mm256_storeu_ps(&cache[0], var);
                            Types::expmt_t acc = cache[0] + cache[1] + cache[2] + cache[3] + cache[4] + cache[5] + cache[6] + cache[7];
                            r1.data[xyi] += acc;
                        }
                        ni += info.y_num_col;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    TEXT::Gadgets::print_progress(x+1, info.n_experiment_iterations);

#ifdef CHECK
                    if(x == 0){
                        assert(r1 == exp_res);
                    }
#endif
                }

                return data;
            }
        };
        
        void unrolling_benchmark(Results::SerialExperimentInfo& info){
            // generate data
            auto X = Types::Matrix::generate(info.x_num_row, info.xy_num_inner, 0.0);
            auto Y = Types::Matrix::generate(info.xy_num_inner, info.y_num_col, 0.0);
            auto res = X*Y;

            int tot = 16;
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
                UnrollingExperiments::add_and_loc_acc_unrol_inner_loop_mult_loop_4_reassoc(10, tot, X, Y, res, info),
                UnrollingExperiments::add_and_loc_acc_unrol_inner_loop_mult_loop_8_reassoc(11, tot, X, Y, res, info),
                UnrollingExperiments::add_and_loc_acc_unrol_inner_loop_mult_loop_2_sep_acc(12, tot, X, Y, res, info),
                UnrollingExperiments::add_and_loc_acc_unrol_inner_loop_mult_loop_4_sep_acc(13, tot, X, Y, res, info),
                UnrollingExperiments::add_and_loc_acc_unrol_inner_loop_mult_loop_8_sep_acc(14, tot, X, Y, res, info),
                UnrollingExperiments::add_and_loc_acc_prec_in_loop_unrol_inner_loop_mult_loop_8_sep_acc(15, tot, X, Y, res, info),
                UnrollingExperiments::add_vectorized(16, tot, X, Y, res, info)
            };

            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(info.experiment_name, info.to_string(), info.to_info(), results);
        }
    }
}