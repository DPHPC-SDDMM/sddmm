#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <functional>

#include "defines.h"

namespace SDDMM{
    class Results {
    public:
        struct ExperimentData {
            std::string label;
            std::vector<Types::time_duration_unit> durations;
        };

        struct EmptyExperimentInfo {
            const std::string experiment_name;
            const Types::vec_size_t n_experiment_iterations;

            EmptyExperimentInfo(
                std::string experiment_name,
                Types::vec_size_t n_experiment_iterations
            ) 
            : 
                n_experiment_iterations(n_experiment_iterations),
                experiment_name(experiment_name) 
            {}

            std::string to_string(){
                std::stringstream s;
                  s << "_nIt-" << n_experiment_iterations;
                
                return s.str();
            }

            std::string to_info() {
                std::stringstream s;
                s << "[INFO]\n"
                  << "experiment_name " << experiment_name << "\n"
                  << "n_experiment_iterations " << n_experiment_iterations << "\n"
                  << "[/INFO]";
                
                return s.str();
            }
        };

        struct CacheExperimentInfo {
            const std::string experiment_name;               
            const Types::vec_size_t r_tile_size;
            const Types::vec_size_t c_tile_size;
            const Types::vec_size_t r_num;
            const Types::vec_size_t c_num;
            const Types::vec_size_t n_experiment_iterations;

            CacheExperimentInfo(
                std::string experiment_name,
                Types::vec_size_t r_tile_size,
                Types::vec_size_t c_tile_size,
                Types::vec_size_t r_num,
                Types::vec_size_t c_num,
                Types::vec_size_t n_experiment_iterations
            )
            :
                experiment_name(experiment_name),
                r_tile_size(r_tile_size),
                c_tile_size(c_tile_size),
                r_num(r_num),
                c_num(c_num),
                n_experiment_iterations(n_experiment_iterations)
            {}

            std::string to_string() {
                std::stringstream s;
                s << "rTS" << r_tile_size
                  << "_cTS" << c_tile_size
                  << "_rNum" << r_num
                  << "_cNum" << c_num
                  << "_nIt" << n_experiment_iterations;
                
                return s.str();
            }

            std::string to_info() {
                std::stringstream s;
                s << "[INFO]\n"
                  << "experiment_name " << experiment_name << "\n"
                  << "r_tile_size " << r_tile_size << "\n"
                  << "c_tile_size " << c_tile_size << "\n"
                  << "r_num " << r_num << "\n"
                  << "c_num " << c_num << "\n"
                  << "n_experiment_iterations " << n_experiment_iterations << "\n"
                  << "[/INFO]";
                
                return s.str();
            }
        };

        /**
         * A data structure which includes the parameterization of a serial experiment.
         * An "experiment" in this context is a process/program which includes
         * some degree of randomness and as such requires to be executed multiple times,
         * so as to collect data, which in turn will be utilized to draw conclusions.
         * Its characterization as "serial" is derived from the fact that all iterations of the program run on a *single CPU core*.
         * 
         * The struct's attributes are:
         * - `std::string experiment_name`
         * - `const Types::vec_size_t tile_size_row`
         * - `const Types::vec_size_t tile_size_row`
         * - `const Types::vec_size_t tile_size_col`
         * - `const Types::vec_size_t x_num_row`
         * - `const Types::vec_size_t xy_num_inner`
         * - `const Types::vec_size_t y_num_row`
         * 
         * The struct's functions are:
         * - `std::string to_string`
         * - `std::string to_info`
        */
        struct SerialExperimentInfo {
            std::string experiment_name; // An identifier, which is used when this struct is "stringified"
            const Types::vec_size_t tile_size_row; // The number of rows which each tile (of the sparse matrix) include, a.k.a the "tile height".
            const Types::vec_size_t tile_size_inner; // The number of rows/columns of the dense matrices will be used in the computation of a single tile.
            const Types::vec_size_t tile_size_col; // The number of columns which each tile (of the sparse matrix) include, a.k.a the "tile width".
            const Types::vec_size_t x_num_row; // The number of rows of the dense matrix X, which is the left-hand-side (LHS) of the dense matrix multiplication XY.
            const Types::vec_size_t xy_num_inner; // The number of columns of the dense matrix X, which (by the definition of matrix multiplication) is identical to the number of rows of matrix Y.
            const Types::vec_size_t y_num_col; // The number of rows of the dense matrix Y, which is the right-hand-side (RHS) of the dense matrix multiplication XY.
            const Types::vec_size_t n_experiment_iterations; // The number of repetitions that the program will be executed.

            SerialExperimentInfo(
                std::string experiment_name,
                Types::vec_size_t tile_size_row,
                Types::vec_size_t tile_size_inner,
                Types::vec_size_t tile_size_col,
                Types::vec_size_t x_num_row,
                Types::vec_size_t xy_num_inner,
                Types::vec_size_t y_num_col,
                Types::vec_size_t n_experiment_iterations
            )
            :
                experiment_name(experiment_name),
                tile_size_row(tile_size_row),
                tile_size_inner(tile_size_inner),
                tile_size_col(tile_size_col),
                x_num_row(x_num_row),
                xy_num_inner(xy_num_inner),
                y_num_col(y_num_col),
                n_experiment_iterations(n_experiment_iterations)
            {}

            /**
             * @return String of the format
             * "tsR-<tile_size_row>_tsI-<tile_size_inner>_tsC-<tle_size_col>_dsr-<x_num_row>_dsI-<xy_num_inner>_dsC-<y_num_col>_n_it-<n_experiment_iterations>"
            */
            std::string to_string() {
                std::stringstream s;
                s << "tsR-" << tile_size_row
                  << "_tsI-" << tile_size_inner
                  << "_tsC-" << tile_size_col
                  << "_dsR-" << x_num_row
                  << "_dsI-" << xy_num_inner
                  << "_dsC-" << y_num_col
                  << "_nIt-" << n_experiment_iterations;
                
                return s.str();
            }

            /**
             * @return A nicely formatted string.
             * @note This function is usually used when writing the [INFO] section of an output file.
            */
            std::string to_info() {
                std::stringstream s;
                s << "[INFO]\n"
                  << "experiment_name " << experiment_name << "\n"
                  << "tile_size_row " << tile_size_row << "\n"
                  << "tile_size_inner " << tile_size_inner << "\n"
                  << "tile_size_col " << tile_size_col << "\n"
                  << "x_num_row " << x_num_row << "\n"
                  << "xy_num_inner " << xy_num_inner << "\n"
                  << "y_num_col " << y_num_col << "\n"
                  << "n_experiment_iterations " << n_experiment_iterations << "\n"
                  << "[/INFO]";
                
                return s.str();
            }
        };


        /**
         * A data structure which includes the parameterization of a (potentially) parallel experiment.
         * An "experiment" in this context is a process/program which includes
         * some degree of randomness and as such requires to be executed multiple times,
         * so as to collect data, which in turn will be utilized to draw conclusions.
         * Its characterization as "parallel" is derived from the fact that the iterations of the program run on *multiple CPU core*.
         * 
         * The struct's attributes are:
         * - `std::string experiment_name`
         * - `const Types::vec_size_t sparse_num_row`
         * - `const Types::vec_size_t sparse_num_col`
         * - `const Types::vec_size_t dense_num_inner`
         * - `float sparsity`
         * - `const Types::vec_size_t n_experiment_iterations`
         * - `const Types::vec_size_t n_cpu_threads`
         * 
         * The struct's functions are:
         * - `std::string to_string`
         * - `std::string to_info`
         * 
         * @note In order for the sparse and dense matrix to be valid, it is impled that:
         * @note `sparse_num_row`: are the number of rows of matrix `X` in the subsequent `X*Y` dense-with-dense matrix product.
         * @note `sparse_num_col` are the number of columns of matrix `Y` in the subsequent `X*Y` dense-with-dense matrix product.
        */
        struct ExperimentInfo {
            ExperimentInfo(
                std::string experiment_name,
                Types::vec_size_t sparse_num_row,
                Types::vec_size_t sparse_num_col,
                Types::vec_size_t dense_num_inner,
                float sparsity,
                Types::vec_size_t n_experiment_iterations,
                Types::vec_size_t n_cpu_threads
            ) :
                experiment_name(experiment_name),
                sparse_num_row(sparse_num_row),
                sparse_num_col(sparse_num_col),
                dense_num_inner(dense_num_inner),
                sparsity(sparsity),
                n_experiment_iterations(n_experiment_iterations),
                n_cpu_threads(n_cpu_threads)
            {}

            const std::string experiment_name;

            // ([sparse_num_row x dense_num_inner] * [dense_num_inner x sparse_num_col])..hadamard..([sparse_num_row x sparse_num_col])
            const Types::vec_size_t sparse_num_row;
            const Types::vec_size_t sparse_num_col; 
            const Types::vec_size_t dense_num_inner;
            // sparsity of the sparse matrix
            const float sparsity;
            // number of iterations per experiment part
            const Types::vec_size_t n_experiment_iterations;
            // number of threads for cpu side multithreading
            const Types::vec_size_t n_cpu_threads;

            /**
             * @return String of the format
             * "[NxK, KxM]Had[NxM]N<sparse_num_row>_M<sparse_num_col>_K<dense_num_inner>_sparsity-<sparsity>_iters-<n_experiment_iterations>_cpu-t<n_cpu_threads>"
            */
            std::string to_string(){
                std::stringstream s;
                s << "[NxK,KxM]Had[NxM]" 
                 << "N" << sparse_num_row 
                 << "_M" << sparse_num_col 
                 << "_K" << dense_num_inner
                 << "_sparsity-" << sparsity
                 << "_iters-" << n_experiment_iterations
                 << "_cpu-t-" << n_cpu_threads;

                return s.str();
            }

            /**
             * @return A nicely formatted string.
             * @note This function is usually used when writing the [INFO] section of an output file.
            */
            std::string to_info(){
                std::stringstream s;
                s << "[INFO]\n"
                 << "experiment_name " << experiment_name << "\n"
                 << "sparse_num_row " << sparse_num_row << "\n"
                 << "sparse_num_col " << sparse_num_col << "\n"
                 << "dense_num_inner " << dense_num_inner << "\n"
                 << "sparsity " << sparsity << "\n"
                 << "n_experiment_iterations " << n_experiment_iterations << "\n"
                 << "n_cpu_threads " << n_cpu_threads << "\n"
                 << "[/INFO]";

                return s.str();
            }
        };

        static std::string to_file(
            const std::string experiment_name,
            const std::string desc_string, 
            const std::string info_string, 
            const std::vector<ExperimentData>& data){
            for(auto d : data){
                assert(d.durations.size() > 0 && "All ExperimentData structs must contain result data");
            }

            auto created_at = std::chrono::system_clock::now();
            auto created_at_t = std::chrono::system_clock::to_time_t(created_at);
            std::string time = std::string(std::ctime(&created_at_t));
            // std::replace(time.begin(), time.end(), ' ', '_');
            std::replace(time.begin(), time.end(), ':', '-');
            time = time.substr(0, time.size()-1);

            std::stringstream name;
            name << "../../results/" << experiment_name << "__"
                 << desc_string
                 << "_[" << time << "]"
                 << ".txt";

            std::ofstream output_file;
            output_file.open(name.str());
            output_file << info_string << "\n";
            output_file << "[DATA]\n" ;
            for(auto d : data){
                output_file << "[L] " << d.label << "\n";
                size_t s = d.durations.size()-1;
                output_file << "[D] ";
                for(size_t i=0; i<s; ++i){
                    output_file << std::setprecision(12) << d.durations.at(i) << " ";
                }
                output_file << std::setprecision(12) << d.durations.at(s) << "\n";
            }
            output_file << "[/DATA]\n" ;

            output_file.close();
            return name.str();
        }
    };
}