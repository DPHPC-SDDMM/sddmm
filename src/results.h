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

#include "defines.h"

namespace SDDMM{
    class Results {
    public:
        struct ExperimentData {
            std::string label;
            std::vector<Types::time_duration_unit> durations;
        };

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

            std::string to_string(){
                std::stringstream s;
                s << "<NxK,KxM>Had<NxM>" 
                 << "N" << sparse_num_row 
                 << "_M" << sparse_num_col 
                 << "_K" << dense_num_inner
                 << "_sparsity-" << sparsity
                 << "_iters-" << n_experiment_iterations
                 << "_cpu-t-" << n_cpu_threads;

                return s.str();
            }

            std::string to_info(){
                std::stringstream s;
                s << "[INFO]\n"
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

        static std::string to_file(ExperimentInfo& info, const std::vector<ExperimentData>& data){
            for(auto d : data){
                assert(d.durations.size() > 0 && "All ExperimentData structs must contain result data");
            }

            auto created_at = std::chrono::system_clock::now();
            auto created_at_t = std::chrono::system_clock::to_time_t(created_at);
            std::string time = std::string(std::ctime(&created_at_t));
            std::replace(time.begin(), time.end(), ' ', '_');
            time = time.substr(0, time.size()-1);

            std::stringstream name;
            name << "../../results/" << info.experiment_name
                 << info.to_string()
                 << "_[" << time << "]"
                 << ".txt";

            std::ofstream output_file;
            output_file.open(name.str());
            output_file << info.to_info() << "\n";
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