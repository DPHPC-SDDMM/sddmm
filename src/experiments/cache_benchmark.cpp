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
            static std::vector<std::vector<T>> init_vec(Types::vec_size_t n_row, Types::vec_size_t n_col){
                std::vector<std::vector<T>> vec;
                for(Types::vec_size_t i=0; i<n_row; ++i){
                    vec.push_back(std::vector<T>(n_col, 1));
                }
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

            public:
            template<typename T>
            static Results::ExperimentData sum_array_rows_vec(int cur_exp, int tot_exp, Results::CacheExperimentInfo& info, Types::vec_size_t r_num, Types::vec_size_t c_num){
                r_num = info.r_num == 0 ? r_num : info.r_num;
                c_num = info.c_num == 0 ? c_num : info.c_num;
                std::vector<std::vector<T>> vec = init_vec<T>(r_num, c_num);

                Results::ExperimentData data;
                data.label = "sum_array_rows_vec [" + std::string(typeid(T).name()) + "]";
                
                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    auto start = std::chrono::high_resolution_clock::now();
                    T sum = static_cast<T>(0);
                    for(Types::vec_size_t r=0; r<r_num; ++r){
                        for(Types::vec_size_t c=0; c<c_num; ++c){
                            sum += vec[r][c];
                        }
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }

                    if(n == 0){
                        T expected = r_num*c_num;
                        assert(sum == expected);
                    }
                }

                return data;
            }

            template<typename T>
            static Results::ExperimentData sum_array_rows_pp(int cur_exp, int tot_exp, Results::CacheExperimentInfo& info, Types::vec_size_t r_num, Types::vec_size_t c_num){
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
                    T sum = static_cast<T>(0);
                    for(Types::vec_size_t r=0; r<r_num; ++r){
                        for(Types::vec_size_t c=0; c<c_num; ++c){
                            sum += arr[r][c];
                        }
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }

                    if(n == 0){
                        T expected = r_num*c_num;
                        assert(sum == expected);
                    }
                }

                free_arr_pp(arr, r_num, c_num);
                return data;
            }

            template<typename T>
            static Results::ExperimentData sum_array_rows_c(int cur_exp, int tot_exp, Results::CacheExperimentInfo& info, Types::vec_size_t r_num, Types::vec_size_t c_num){
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
                    T sum = static_cast<T>(0);
                    for(Types::vec_size_t r=0; r<r_num; ++r){
                        for(Types::vec_size_t c=0; c<c_num; ++c){
                            sum += arr[r*c_num + c];
                        }
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }

                    if(n == 0){
                        T expected = r_num*c_num;
                        assert(sum == expected);
                    }
                }

                free_arr_c(arr);
                return data;
            }

            template<typename T>
            static Results::ExperimentData sum_array_cols_vec(int cur_exp, int tot_exp, Results::CacheExperimentInfo& info, Types::vec_size_t r_num, Types::vec_size_t c_num){
                r_num = info.r_num == 0 ? r_num : info.r_num;
                c_num = info.c_num == 0 ? c_num : info.c_num;
                std::vector<std::vector<T>> vec = init_vec<T>(info.r_num, info.c_num);

                Results::ExperimentData data;
                data.label = "sum_array_cols_vec [" + std::string(typeid(T).name()) + "]";

                std::cout << TEXT::Cast::Cyan(get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, info.n_experiment_iterations);

                Types::vec_size_t n_max = info.n_experiment_iterations+1;
                for(Types::vec_size_t n=0; n<n_max; ++n){
                    TEXT::Gadgets::print_progress(n, info.n_experiment_iterations);
                    auto start = std::chrono::high_resolution_clock::now();
                    T sum = 0;
                    for(Types::vec_size_t c=0; c<c_num; ++c){
                        for(Types::vec_size_t r=0; r<r_num; ++r){
                            sum += vec[r][c];
                        }
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }

                    if(n == 0){
                        assert(sum == r_num*c_num);
                    }
                }

                return data;
            }

            template<typename T>
            static Results::ExperimentData sum_array_cols_pp(int cur_exp, int tot_exp, Results::CacheExperimentInfo& info, Types::vec_size_t r_num, Types::vec_size_t c_num){
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
                    T sum = static_cast<T>(0);
                    for(Types::vec_size_t c=0; c<c_num; ++c){
                        for(Types::vec_size_t r=0; r<r_num; ++r){
                            sum += arr[r][c];
                        }
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }

                    if(n == 0){
                        T expected = r_num*c_num;
                        assert(sum == expected);
                    }
                }

                free_arr_pp(arr, r_num, c_num);
                return data;
            }

            template<typename T>
            static Results::ExperimentData sum_array_cols_c(int cur_exp, int tot_exp, Results::CacheExperimentInfo& info, Types::vec_size_t r_num, Types::vec_size_t c_num){
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
                    T sum = static_cast<T>(0);
                    for(Types::vec_size_t c=0; c<c_num; ++c){
                        for(Types::vec_size_t r=0; r<r_num; ++r){
                            sum += arr[r*c_num + c];
                        }
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    if(n > 0){
                        // discard warmup
                        data.durations.push_back(std::chrono::duration_cast<Types::time_measure_unit>(end - start).count());
                    }

                    if(n == 0){
                        T expected = r_num*c_num;
                        assert(sum == expected);
                    }
                }

                free_arr_c(arr);
                return data;
            }
        };
        
        void types_benchmark(Results::CacheExperimentInfo& info){
            std::vector<Results::ExperimentData> results {
                CacheExperiments::sum_array_rows_vec<int>(1, 6, info, 0, 0),
                CacheExperiments::sum_array_rows_vec<float>(2, 6, info, 0, 0),
                CacheExperiments::sum_array_rows_vec<double>(3, 6, info, 0, 0),
                CacheExperiments::sum_array_cols_vec<int>(4, 6, info, 0, 0),
                CacheExperiments::sum_array_cols_vec<float>(5, 6, info, 0, 0),
                CacheExperiments::sum_array_cols_vec<double>(6, 6, info, 0, 0)
            };

            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(info.experiment_name, info.to_string(), info.to_info(), results);
        }

        void arr_vs_vec_vs_ptr_benchmark(Results::CacheExperimentInfo& info){
            std::vector<Results::ExperimentData> results {
                CacheExperiments::sum_array_rows_vec<int64_t>(1, 6, info, 0, 0),
                CacheExperiments::sum_array_rows_pp<int64_t>(2, 6, info, 0, 0),
                CacheExperiments::sum_array_rows_c<int64_t>(3, 6, info, 0, 0),
                CacheExperiments::sum_array_cols_vec<int64_t>(4, 6, info, 0, 0),
                CacheExperiments::sum_array_cols_pp<int64_t>(5, 6, info, 0, 0),
                CacheExperiments::sum_array_cols_c<int64_t>(6, 6, info, 0, 0)
            };

            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(info.experiment_name, info.to_string(), info.to_info(), results);
        }

        void cache_benchmark(Results::CacheExperimentInfo& info){
            // generate data

            int tot = 1;
            std::vector<Results::ExperimentData> results {
                CacheExperiments::sum_array_cols_vec<int>(1, 6, info, 100, 100),
                CacheExperiments::sum_array_cols_vec<int>(2, 6, info, 100, 100)
            };

            std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
            Results::to_file(info.experiment_name, info.to_string(), info.to_info(), results);
        }
    }
}