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

// #define CHECK

namespace SDDMM {
    namespace Experiments {
        class CacheExperiments {
            private:
            static std::string get_cur(int cur_exp, int tot_exp){
                return std::string("..(") 
                     + std::to_string(cur_exp) 
                     + std::string("/") 
                     + std::to_string(tot_exp) 
                     + std::string(")..");
            }

            template<typename T>
            static std::vector<std::vector<T>> init_vec_2d(Types::vec_size_t n_row, Types::vec_size_t n_col){
                std::vector<std::vector<T>> vec;
                // std::vector<std::vector<T>> test(n_row, std::vector<T>(n_col, 1))
                for(Types::vec_size_t i=0; i<n_row; ++i){
                    vec.push_back(std::vector<T>(n_col, 1));
                }
                return vec;
            }

            template<typename T>
            static std::vector<T> init_vec_1d(Types::vec_size_t n_row, Types::vec_size_t n_col){
                std::vector<T> vec(n_row*n_col, static_cast<T>(1.0));
                return vec;
            }

            template<typename T>
            static T** init_arr_pp(Types::vec_size_t n_row, Types::vec_size_t n_col){
                T** arr = new T*[n_row];
                for(Types::vec_size_t i=0; i<n_row; ++i){
                    arr[i] = new T[n_col];
                    for(Types::vec_size_t j=0; j<n_col; ++j){
                        arr[i][j] = 1;
                    }
                }
                return arr;
            }

            template<typename T>
            static void free_arr_pp(T** arr, Types::vec_size_t n_row, Types::vec_size_t n_col){
                for(Types::vec_size_t i=0; i<n_row; ++i){
                    delete[] arr[i];
                }
                delete[] arr;
            }

            template<typename T>
            static T* init_arr_c(Types::vec_size_t n_row, Types::vec_size_t n_col){
                Types::vec_size_t s = n_col*n_row;
                T* arrc = new T[s];
                for(Types::vec_size_t i=0; i<s; ++i){
                    arrc[i] = 1;
                }
                return arrc;
            }

            template<typename T>
            static void free_arr_c(T* arr){
                delete[] arr;
            }

            template<typename T>
            static T sum_arr_2d(T** arr, Types::vec_size_t r_num, Types::vec_size_t c_num){
                T sum = static_cast<T>(0);
                for(Types::vec_size_t r=0; r<r_num; ++r){
                    for(Types::vec_size_t c=0; c<c_num; ++c){
                        sum += arr[r][c];
                    }
                }
                return sum;
            }

            template<typename T>
            static T sum_arr_2d_cols(T** arr, Types::vec_size_t r_num, Types::vec_size_t c_num){
                T sum = static_cast<T>(0);
                for(Types::vec_size_t c=0; c<c_num; ++c){
                    for(Types::vec_size_t r=0; r<r_num; ++r){
                        sum += arr[r][c];
                    }
                }
                return sum;
            }

            template<typename T>
            static T sum_arr_1d(T* arr, Types::vec_size_t r_num, Types::vec_size_t c_num){
                T sum = static_cast<T>(0);
                for(Types::vec_size_t r=0; r<r_num; ++r){
                    for(Types::vec_size_t c=0; c<c_num; ++c){
                        sum += arr[r*c_num + c];
                    }
                }
                return sum;
            }

            template<typename T>
            static T sum_arr_1d_cols(T* arr, Types::vec_size_t r_num, Types::vec_size_t c_num){
                T sum = static_cast<T>(0);
                for(Types::vec_size_t c=0; c<c_num; ++c){
                    for(Types::vec_size_t r=0; r<r_num; ++r){
                        sum += arr[r*c_num + c];
                    }
                }
                return sum;
            }

            template<typename T>
            static T sum_vec_1d(std::vector<T> vec, Types::vec_size_t r_num, Types::vec_size_t c_num){
                T sum = static_cast<T>(0);
                for(Types::vec_size_t r=0; r<r_num; ++r){
                    for(Types::vec_size_t c=0; c<c_num; ++c){
                        sum += vec[r*c_num + c];
                    }
                }
                return sum;
            }

            template<typename T>
            static T sum_vec_1d_cols(std::vector<T> vec, Types::vec_size_t r_num, Types::vec_size_t c_num){
                T sum = static_cast<T>(0);
                for(Types::vec_size_t c=0; c<c_num; ++c){
                    for(Types::vec_size_t r=0; r<r_num; ++r){
                        sum += vec[r*c_num + c];
                    }
                }
                return sum;
            }

            template<typename T>
            static T sum_vec_2d(std::vector<std::vector<T>> vec_2d, Types::vec_size_t r_num, Types::vec_size_t c_num){
                T sum = static_cast<T>(0);
                for(Types::vec_size_t r=0; r<r_num; ++r){
                    for(Types::vec_size_t c=0; c<c_num; ++c){
                        sum += vec_2d[r][c];
                    }
                }
                return sum;
            }

            template<typename T>
            static T sum_vec_2d_cols(std::vector<std::vector<T>> vec_2d, Types::vec_size_t r_num, Types::vec_size_t c_num){
                T sum = static_cast<T>(0);
                for(Types::vec_size_t c=0; c<c_num; ++c){
                    for(Types::vec_size_t r=0; r<r_num; ++r){
                        sum += vec_2d[r][c];
                    }
                }
                return sum;
            }

            public:
            template<typename T>
            static Results::ExperimentData sum_array_rows_vec_2d(int cur_exp, int tot_exp, Results::CacheExperimentInfo& info, Types::vec_size_t r_num, Types::vec_size_t c_num, T& total){
                r_num = info.r_num == 0 ? r_num : info.r_num;
                c_num = info.c_num == 0 ? c_num : info.c_num;
                std::vector<std::vector<T>> vec = init_vec_2d<T>(r_num, c_num);

                Results::ExperimentData data;
                data.label = "sum_array_rows_vec_2d [" + std::string(typeid(T).name()) + "]";
                
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    total += sum_vec_2d(vec, r_num, c_num);
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }

                    if(n == 0){
                        T expected = r_num*c_num;
                        assert(total == expected);
                    }
                }

                return data;
            }

            template<typename T>
            static Results::ExperimentData sum_array_rows_pp(int cur_exp, int tot_exp, Results::CacheExperimentInfo& info, Types::vec_size_t r_num, Types::vec_size_t c_num, T& total){
                r_num = info.r_num == 0 ? r_num : info.r_num;
                c_num = info.c_num == 0 ? c_num : info.c_num;
                T** arr = init_arr_pp<T>(r_num, c_num);

                Results::ExperimentData data;
                data.label = "sum_array_rows_pp [" + std::string(typeid(T).name()) + "]";
                
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    total += sum_arr_2d(arr, r_num, c_num);
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }

                    if(n == 0){
                        T expected = r_num*c_num;
                        assert(total == expected);
                    }
                }

                free_arr_pp(arr, r_num, c_num);
                return data;
            }

            template<typename T>
            static Results::ExperimentData sum_array_rows_c(int cur_exp, int tot_exp, Results::CacheExperimentInfo& info, Types::vec_size_t r_num, Types::vec_size_t c_num, T& total){
                r_num = info.r_num == 0 ? r_num : info.r_num;
                c_num = info.c_num == 0 ? c_num : info.c_num;
                T* arr = init_arr_c<T>(r_num, c_num);

                Results::ExperimentData data;
                data.label = "sum_array_rows_c [" + std::string(typeid(T).name()) + "]";
                
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    total += sum_arr_1d(arr, r_num, c_num);

                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }

                    if(n == 0){
                        T expected = r_num*c_num;
                        assert(total == expected);
                    }
                }

                free_arr_c(arr);
                return data;
            }

            template<typename T>
            static Results::ExperimentData sum_array_cols_vec_2d(int cur_exp, int tot_exp, Results::CacheExperimentInfo& info, Types::vec_size_t r_num, Types::vec_size_t c_num, T& total){
                r_num = info.r_num == 0 ? r_num : info.r_num;
                c_num = info.c_num == 0 ? c_num : info.c_num;
                std::vector<std::vector<T>> vec = init_vec_2d<T>(info.r_num, info.c_num);

                Results::ExperimentData data;
                data.label = "sum_array_cols_vec_2d [" + std::string(typeid(T).name()) + "]";

                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    total += sum_vec_2d_cols(vec, r_num, c_num);

                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }

                    if(n == 0){
                        T expected = r_num*c_num;
                        assert(total == expected);
                    }
                }

                return data;
            }

            template<typename T>
            static Results::ExperimentData sum_array_cols_pp(int cur_exp, int tot_exp, Results::CacheExperimentInfo& info, Types::vec_size_t r_num, Types::vec_size_t c_num, T& total){
                r_num = info.r_num == 0 ? r_num : info.r_num;
                c_num = info.c_num == 0 ? c_num : info.c_num;
                T** arr = init_arr_pp<T>(r_num, c_num);

                Results::ExperimentData data;
                data.label = "sum_array_cols_pp [" + std::string(typeid(T).name()) + "]";
                
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    total += sum_arr_2d_cols(arr, r_num, c_num);
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }

                    if(n == 0){
                        T expected = r_num*c_num;
                        assert(total == expected);
                    }
                }

                free_arr_pp(arr, r_num, c_num);
                return data;
            }

            template<typename T>
            static Results::ExperimentData sum_array_cols_c(int cur_exp, int tot_exp, Results::CacheExperimentInfo& info, Types::vec_size_t r_num, Types::vec_size_t c_num, T& total){
                r_num = info.r_num == 0 ? r_num : info.r_num;
                c_num = info.c_num == 0 ? c_num : info.c_num;
                T* arr = init_arr_c<T>(r_num, c_num);

                Results::ExperimentData data;
                data.label = "sum_array_cols_c [" + std::string(typeid(T).name()) + "]";
                
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    total += sum_arr_1d_cols(arr, r_num, c_num);
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }

                    if(n == 0){
                        T expected = r_num*c_num;
                        assert(total == expected);
                    }
                }

                free_arr_c(arr);
                return data;
            }
        

            template<typename T>
            static Results::ExperimentData sum_array_rows_vec_1d(int cur_exp, int tot_exp, Results::CacheExperimentInfo& info, Types::vec_size_t r_num, Types::vec_size_t c_num, T& total){
                r_num = info.r_num == 0 ? r_num : info.r_num;
                c_num = info.c_num == 0 ? c_num : info.c_num;
                std::vector<T> vec = init_vec_1d<T>(r_num, c_num);

                Results::ExperimentData data;
                data.label = "sum_array_rows_vec_1d [" + std::string(typeid(T).name()) + "]";
                
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    total += sum_vec_1d(vec, r_num, c_num);
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }

                    if(n == 0){
                        T expected = r_num*c_num;
                        assert(total == expected);
                    }
                }

                return data;
            }

            template<typename T>
            static Results::ExperimentData sum_array_cols_vec_1d(int cur_exp, int tot_exp, Results::CacheExperimentInfo& info, Types::vec_size_t r_num, Types::vec_size_t c_num, T& total){
                r_num = info.r_num == 0 ? r_num : info.r_num;
                c_num = info.c_num == 0 ? c_num : info.c_num;
                std::vector<T> vec = init_vec_1d<T>(r_num, c_num);

                Results::ExperimentData data;
                data.label = "sum_array_cols_vec_1d [" + std::string(typeid(T).name()) + "]";
                
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    total += sum_vec_1d_cols(vec, r_num, c_num);
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }

                    if(n == 0){
                        T expected = r_num*c_num;
                        assert(total == expected);
                    }
                }

                return data;
            }
        };
        
        void types_benchmark(Results::CacheExperimentInfo& info){
            int total_1 = 0;
            float total_2 = 0;
            double total_3 = 0;
            int total_4 = 0;
            float total_5 = 0;
            double total_6 = 0;
            std::vector<Results::ExperimentData> results {
                CacheExperiments::sum_array_rows_vec_1d<int>(1, 6, info, 0, 0, total_1),
                CacheExperiments::sum_array_rows_vec_1d<float>(2, 6, info, 0, 0, total_2),
                CacheExperiments::sum_array_rows_vec_1d<double>(3, 6, info, 0, 0, total_3),
                CacheExperiments::sum_array_cols_vec_1d<int>(4, 6, info, 0, 0, total_4),
                CacheExperiments::sum_array_cols_vec_1d<float>(5, 6, info, 0, 0, total_5),
                CacheExperiments::sum_array_cols_vec_1d<double>(6, 6, info, 0, 0, total_6)
            };

            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(info.experiment_name, info.to_string(), info.to_info(), results);
        }

        int64_t total_1 = 0;
        int64_t total_2 = 0;
        int64_t total_3 = 0;
        int64_t total_4 = 0;
        int64_t total_5 = 0;
        int64_t total_6 = 0;
        int64_t total_7 = 0;
        int64_t total_8 = 0;
        void arr_vs_vec_vs_ptr_benchmark(Results::CacheExperimentInfo& info){
            std::vector<Results::ExperimentData> results {
                CacheExperiments::sum_array_rows_vec_1d<int64_t>(1, 6, info, 0, 0, total_1),
                CacheExperiments::sum_array_rows_vec_2d<int64_t>(1, 6, info, 0, 0, total_2),
                CacheExperiments::sum_array_rows_pp<int64_t>(2, 6, info, 0, 0, total_3),
                CacheExperiments::sum_array_rows_c<int64_t>(3, 6, info, 0, 0, total_4),
                CacheExperiments::sum_array_cols_vec_1d<int64_t>(4, 6, info, 0, 0, total_5),
                CacheExperiments::sum_array_cols_vec_2d<int64_t>(4, 6, info, 0, 0, total_6),
                CacheExperiments::sum_array_cols_pp<int64_t>(5, 6, info, 0, 0, total_7),
                CacheExperiments::sum_array_cols_c<int64_t>(6, 6, info, 0, 0, total_8)
            };

            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(info.experiment_name, info.to_string(), info.to_info(), results);
        }

        void cache_benchmark(Results::CacheExperimentInfo& info){
            // generate data

            // int tot = 1;
            // std::vector<Results::ExperimentData> results {
            //     CacheExperiments::sum_array_cols_vec<int>(1, 6, info, 100, 100),
            //     CacheExperiments::sum_array_cols_vec<int>(2, 6, info, 100, 100)
            // };

            // std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            // Results::to_file(info.experiment_name, info.to_string(), info.to_info(), results);
        }
    }
}