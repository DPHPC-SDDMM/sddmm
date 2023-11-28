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
        class UnrollingExperiments2 {
            private:
            static std::string get_cur(int cur_exp, int tot_exp){
                return std::string("..(") 
                     + std::to_string(cur_exp) 
                     + std::string("/") 
                     + std::to_string(tot_exp) 
                     + std::string(")..");
            }

            static std::string get_dims(Types::Matrix& X, Types::Matrix& Y){
                std::string res = "";
                res += "[N=" + std::to_string(X.n) + ",";
                res += "K=" + std::to_string(X.m) + ",";
                res += "M=" + std::to_string(Y.n) + "]";
                return res;
            }

            public:

            static Results::ExperimentData unroll_4(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info,
                Types::expmt_t& total
            ) {
                Results::ExperimentData data;
                data.label = "Unroll (4x), separate accums, collecting 'while' " + get_dims(X,Y);
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Types::Matrix r1(info.x_num_row, info.y_num_col);
                    Types::vec_size_t ni = 0;
                    const Types::vec_size_t j = 4;
                    const Types::vec_size_t s = info.xy_num_inner;
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
                    total += r1(0,0);
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

            static Results::ExperimentData unroll_4_improve_1(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info,
                Types::expmt_t& total
            ) {
                Results::ExperimentData data;
                data.label  = "Unroll (4x), separate accums, zero padding " + get_dims(X,Y);
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Types::Matrix r1(info.x_num_row, info.y_num_col);
                    Types::vec_size_t ni = 0;
                    const Types::vec_size_t j = 4;
                    const Types::vec_size_t s = info.xy_num_inner;
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
                                const Types::vec_size_t i0 = i+0;
                                const Types::vec_size_t i1 = i+1;
                                const Types::vec_size_t i2 = i+2;
                                const Types::vec_size_t i3 = i+3;

                                bool t0 = i0 < s;
                                bool t1 = i1 < s;
                                bool t2 = i2 < s;
                                bool t3 = i3 < s;

                                var_1 += t0 ? X.data[r*info.xy_num_inner + i0] * Y.data[i0*info.y_num_col + c] : 0;
                                var_2 += t1 ? X.data[r*info.xy_num_inner + i1] * Y.data[i1*info.y_num_col + c] : 0;
                                var_3 += t2 ? X.data[r*info.xy_num_inner + i2] * Y.data[i2*info.y_num_col + c] : 0;
                                var_4 += t3 ? X.data[r*info.xy_num_inner + i3] * Y.data[i3*info.y_num_col + c] : 0;
                            }

                            r1.data[xyi] = var_1 + var_2 + var_3 + var_4;
                        }
                        ni += info.y_num_col;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    total += r1(0,0);
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

            static Results::ExperimentData unroll_8(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info,
                Types::expmt_t& total
            ) {
                Results::ExperimentData data;
                data.label = "Unroll (8x), separate accums, collecting 'while' " + get_dims(X,Y);
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Types::Matrix r1(info.x_num_row, info.y_num_col);
                    Types::vec_size_t ni = 0;
                    const Types::vec_size_t j = 8;
                    const Types::vec_size_t s = info.xy_num_inner;
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
                    total += r1(0,0);
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

            static Results::ExperimentData unroll_8_improve_1(
                int cur_exp,
                int tot_exp,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::Matrix& exp_res,
                Results::SerialExperimentInfo& info,
                Types::expmt_t& total
            ) {
                Results::ExperimentData data;
                data.label  = "Unroll (8x), separate accums, zero padding " + get_dims(X,Y);
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);
                for(auto x=0; x<info.n_experiment_iterations; ++x)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Types::Matrix r1(info.x_num_row, info.y_num_col);
                    Types::vec_size_t ni = 0;
                    const Types::vec_size_t j = 8;
                    const Types::vec_size_t s = info.xy_num_inner;
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
                                const Types::vec_size_t i0 = i+0;
                                const Types::vec_size_t i1 = i+1;
                                const Types::vec_size_t i2 = i+2;
                                const Types::vec_size_t i3 = i+3;
                                const Types::vec_size_t i4 = i+4;
                                const Types::vec_size_t i5 = i+5;
                                const Types::vec_size_t i6 = i+6;
                                const Types::vec_size_t i7 = i+7;

                                bool t0 = i0 < s;
                                bool t1 = i1 < s;
                                bool t2 = i2 < s;
                                bool t3 = i3 < s;
                                bool t4 = i4 < s;
                                bool t5 = i5 < s;
                                bool t6 = i6 < s;
                                bool t7 = i7 < s;

                                var_1 += t0 ? X.data[r*info.xy_num_inner + i0] * Y.data[i0*info.y_num_col + c] : 0;
                                var_2 += t1 ? X.data[r*info.xy_num_inner + i1] * Y.data[i1*info.y_num_col + c] : 0;
                                var_3 += t2 ? X.data[r*info.xy_num_inner + i2] * Y.data[i2*info.y_num_col + c] : 0;
                                var_4 += t3 ? X.data[r*info.xy_num_inner + i3] * Y.data[i3*info.y_num_col + c] : 0;
                                var_5 += t0 ? X.data[r*info.xy_num_inner + i4] * Y.data[i4*info.y_num_col + c] : 0;
                                var_6 += t1 ? X.data[r*info.xy_num_inner + i5] * Y.data[i5*info.y_num_col + c] : 0;
                                var_7 += t2 ? X.data[r*info.xy_num_inner + i6] * Y.data[i6*info.y_num_col + c] : 0;
                                var_8 += t3 ? X.data[r*info.xy_num_inner + i7] * Y.data[i7*info.y_num_col + c] : 0;
                            }

                            r1.data[xyi] = var_1 + var_2 + var_3 + var_4 + var_5 + var_6 + var_7 + var_8;
                        }
                        ni += info.y_num_col;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    total += r1(0,0);
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
        
        void unrolling_benchmark_2(Results::SerialExperimentInfo& info){
            // generate data
            auto X_fit = Types::Matrix::generate_row_major(info.x_num_row, info.xy_num_inner, 0.0);
            auto Y_fit = Types::Matrix::generate_col_major(info.xy_num_inner, info.y_num_col, 0.0);
            auto res_fit = X_fit*Y_fit;

            // construct something that is not divisible by 4 to test the while loop impact
            Types::vec_size_t x_num_row =  info.x_num_row;
            Types::vec_size_t xy_num_inner = info.xy_num_inner;
            Types::vec_size_t y_num_col =  info.y_num_col;
            while(x_num_row % 4 == 0) x_num_row--;
            while(xy_num_inner % 4 == 0) xy_num_inner--;
            while(y_num_col % 4 == 0) y_num_col--;
            Results::SerialExperimentInfo info2(
                info.experiment_name,
                info.tile_size_row,
                info.tile_size_inner,
                info.tile_size_col,
                x_num_row,
                xy_num_inner,
                y_num_col,
                info.n_experiment_iterations
            );

            auto X_prime = Types::Matrix::generate_row_major(info2.x_num_row, info2.xy_num_inner, 0.0);
            auto Y_prime = Types::Matrix::generate_row_major(info2.xy_num_inner, info2.y_num_col, 0.0);
            auto res_prime = X_prime*Y_prime;

            Types::expmt_t total_1 = 0;
            Types::expmt_t total_2 = 0;
            Types::expmt_t total_3 = 0;
            Types::expmt_t total_4 = 0;
            Types::expmt_t total_5 = 0;
            Types::expmt_t total_6 = 0;
            Types::expmt_t total_7 = 0;
            Types::expmt_t total_8 = 0;

            int tot = 8;
            std::vector<Results::ExperimentData> results {
                UnrollingExperiments2::unroll_4(1, tot, X_fit, Y_fit, res_fit, info, total_1),
                UnrollingExperiments2::unroll_4(2, tot, X_prime, Y_prime, res_prime, info2, total_2),
                UnrollingExperiments2::unroll_8(3, tot, X_fit, Y_fit, res_fit, info, total_3),
                UnrollingExperiments2::unroll_8(4, tot, X_prime, Y_prime, res_prime, info2, total_4),
                UnrollingExperiments2::unroll_4_improve_1(5, tot, X_fit, Y_fit, res_fit, info, total_5),
                UnrollingExperiments2::unroll_4_improve_1(6, tot, X_prime, Y_prime, res_prime, info2, total_6),
                UnrollingExperiments2::unroll_8_improve_1(7, tot, X_fit, Y_fit, res_fit, info, total_7),
                UnrollingExperiments2::unroll_8_improve_1(8, tot, X_prime, Y_prime, res_prime, info2, total_8)
            };

            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(info.experiment_name, info.to_string(), info.to_info(), results);
        }
    }
}