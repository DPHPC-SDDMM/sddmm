#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include "../defines.h"
#include "../data_structures/matrix/matrix.h"
#include "../data_structures/csr/csr.h"
#include "../data_structures/coo/coo.h"

namespace SDDMM {
    class DataGenerator {
    public:
        struct Sparsity {
            float S_sparsity;
            float X_sparsity;
            float Y_sparsity;
        };

        /**
        * @brief Loads an existing matrix market file and generates two fitting uniformly distributed dense matrices A and B for
        * SDDMM (A @ B).Hadamard(matrix market file)
        *
        * @param target_folder: folder where result will be stored (must end with a path separator, make sure to use the correct one for the operating system)
        * @param mm_source_file: source file for the matrix market sparse matrix (Note: make sure to use the correct path separators for the operating system)
        * @param K: inner dimension that will be used for the two dense matrices A and B (N and M will be the ones from the loaded matrix market file)
        * @out out_size_written: size in bytes written to the binary file
        * @param skip: [true/false] if true, ask-if-sizes-are-ok is skipped (can be used for generation scripts)
        * @returns string of the absolute path where the binary file was stored
        *
        * @warning Dimensionality of matrices are expected to match each operation used, i.e.
        *  1) If X in R^{N x K}, then Y must be in R^{K x M}
        *  2) A_sparse must be in R^{N x K}
        *
        * @sa
        * - [COO matrix format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO))
        * - [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
        */
        static std::string huge_generator_matrix_market(
            std::string target_folder,
            std::string mm_source_file,
            Types::vec_size_t K,
            uint64_t& out_size_written,
            bool skip=false
        ) {
            if (!(target_folder.back() == '/' || target_folder.back() == '\\')) {
                std::string msg = std::string("target_folder path must end with a valid path-separator (") +
                    Defines::path_separator +
                    std::string(" on current OS)!");
                TEXT::Gadgets::print_colored_text_line(msg, TEXT::RED);
                throw new std::runtime_error(msg);
            }

            uint64_t out_size_read;
            Types::COO mm_matrix = Types::COO::read_matrix_market_file(mm_source_file.c_str(), out_size_read, true);

            Types::vec_size_t N = mm_matrix.n;
            Types::vec_size_t M = mm_matrix.m;
            double nnz = mm_matrix.values.size();
            double x_mb = static_cast<double>(N) * static_cast<double>(K) / 1024.0 / 1024 * sizeof(Types::expmt_t);
            double y_mb = static_cast<double>(M) * static_cast<double>(K) / 1024.0 / 1024 * sizeof(Types::expmt_t);
            double s_mb = nnz / 1024.0 / 1024 * (sizeof(Types::expmt_t) + 2 * sizeof(Types::vec_size_t));
            double S_sparsity = 1.0 - 1.0 / (N * M) * nnz;

            TEXT::Gadgets::print_colored_text_line(
                std::string("Generating\n") +
                std::string("...dense X:  [") + std::to_string(N) + std::string(" x ") + std::to_string(K) + std::string("], ") + std::to_string(x_mb) + std::string("MB\n") +
                std::string("...dense Y:  [") + std::to_string(K) + std::string(" x ") + std::to_string(M) + std::string("], ") + std::to_string(y_mb) + std::string("MB\n") +
                std::string("...sparse S: [") + std::to_string(N) + std::string(" x ") + std::to_string(M) + std::string("] with sparsity ")
                + std::to_string(S_sparsity) + std::string(", approx ") + std::to_string(nnz) + std::string(" nnz values, ") + std::to_string(s_mb) + std::string("MB\n") +
                std::string("total required size: ") + std::to_string(x_mb + y_mb + s_mb) + std::string("MB\n"),
                TEXT::BRIGHT_MAGENTA
            );

            TEXT::Gadgets::print_colored_line(100, '=', TEXT::RED);
            TEXT::Gadgets::print_colored_text_line("Proceed? [y/n]", TEXT::RED);
            if (!skip) {
                char ans = 'n';
                std::cin >> ans;
                if (!(ans == 'y')) {
                    return "";
                }
            }

            return huge_generator_dense_gen(
                target_folder, mm_matrix, K, S_sparsity, 0.0f, 0.0f, out_size_written
            );
        }

        /**
        * @brief Takes existing sparse matrix in COO format and generates two fitting uniformly distributed dense matrices A and B for
        * SDDMM (A @ B).Hadamard(matrix market file)
        *
        * @param target_folder: folder where result will be stored (must end with a path separator, make sure to use the correct one for the operating system)
        * @param mm_coo_matrix: existing COO formated sparse matrix
        * @param K: inner dimension that will be used for the two dense matrices A and B (N and M will be the ones from the loaded matrix market file)
        * @param S_sparsity: proportion of zeros in sparse matrix (0 == no zeros, 1.0 all zeros)
        * @param X_sparsity: required proportion of zeros in dense matrix X (0 == no zeros, 1.0 all zeros)
        * @param Y_sparsity: required proportion of zeros in dense matrix Y (0 == no zeros, 1.0 all zeros)
        * @out out_size_written: size in bytes written to the binary file
        * @param skip: [true/false] if true, ask-if-sizes-are-ok is skipped (can be used for generation scripts)
        * @returns string of the absolute path where the binary file was stored
        *
        * @warning Dimensionality of matrices are expected to match each operation used, i.e.
        *  1) If X in R^{N x K}, then Y must be in R^{K x M}
        *  2) A_sparse must be in R^{N x K}
        *
        * @sa
        * - [COO matrix format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO))
        * - [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
        */
        static std::string huge_generator_dense_gen(
            std::string target_folder,
            Types::COO& mm_coo_matrix,
            Types::vec_size_t K,
            float S_sparsity, float X_sparsity, float Y_sparsity, uint64_t& out_size_written,
            bool skip = false
        ) {
            if (!(target_folder.back() == '/' || target_folder.back() == '\\')) {
                std::string msg = std::string("target_folder path must end with a valid path-separator (") +
                    Defines::path_separator +
                    std::string(" on current OS)!");
                TEXT::Gadgets::print_colored_text_line(msg, TEXT::RED);
                throw new std::runtime_error(msg);
            }

            std::vector<std::chrono::high_resolution_clock::time_point> ts;
            std::vector<std::string> ts_labels;

            ts.push_back(std::chrono::high_resolution_clock::now());
            ts_labels.push_back("start");
            auto X = SDDMM::Types::Matrix::generate_row_major(mm_coo_matrix.n, K, X_sparsity, -1.f, 1.f, true, 40000);
            ts.push_back(std::chrono::high_resolution_clock::now());
            ts_labels.push_back("generate X");

            auto Y = SDDMM::Types::Matrix::generate_col_major(K, mm_coo_matrix.m, Y_sparsity, -1.f, 1.f, true, 40000);
            ts.push_back(std::chrono::high_resolution_clock::now());
            ts_labels.push_back("generate Y");

            TEXT::Gadgets::print_colored_text_line("Write to file:", TEXT::BRIGHT_CYAN);
            std::string name = SDDMM::Types::COO::hadamard_to_bin_file(
                target_folder, mm_coo_matrix, S_sparsity, X, X_sparsity, Y, Y_sparsity, out_size_written);
            ts.push_back(std::chrono::high_resolution_clock::now());
            ts_labels.push_back("to file");

            TEXT::Gadgets::print_colored_text_line("Time results:", TEXT::BRIGHT_CYAN);
            for (int i = 1; i < ts.size(); ++i) {
                auto duration = std::chrono::duration_cast<SDDMM::Types::time_measure_unit>(ts[i] - ts[i - 1]).count();
                TEXT::Gadgets::print_colored_text_line(ts_labels[i] + ":\t" + std::to_string(static_cast<double>(duration) / 1000.0) + std::string("ms"), TEXT::BRIGHT_BLUE);
            }

            return name;

            // SDDMM::Types::COO out_coo;
            // SDDMM::Types::CSR out_csr;
            // SDDMM::Types::Matrix out_X(0,0);
            // SDDMM::Types::Matrix out_Y(0,0);
            // float out_sparse_sparsity;
            // float out_x_sparsity;
            // float out_y_sparsity;
            // SDDMM::Types::COO::hadamard_from_bin_file(
            //     target_folder + name,
            //     out_coo, out_csr, out_sparse_sparsity, 
            //     out_X, out_x_sparsity, 
            //     out_Y, out_y_sparsity);
        }

        /**
        * @brief Generates uniformly distributed sparse matrix and two fitting dense matrices A and B for
        * SDDMM (A @ B).Hadamard(matrix market file). It takes an inner dimension and required sizes in bytes for the two dense matrices. The goal is to
        * create matrices with dimensions N and M as large as possible because the goal is to create matrices that are large.
        *
        * @param target_folder: folder where result will be stored (must end with a path separator, make sure to use the correct one for the operating system)
        * @param K: inner dimension that will be used for the two dense matrices A and B (N and M will be the ones from the loaded matrix market file)
        * @param sizeof_X_in_byte: required size of matrix X in bytes (Note: the real size will be the best possible, smaller size to fit the dimensions)
        * @param sizeof_Y_in_byte: required size of matrix Y in bytes (Note: the real size will be the best possible, smaller size to fit the dimensions)
        * @param S_sparsity: proportion of zeros in sparse matrix (0 == no zeros, 1.0 all zeros)
        * @param K_row: inner dimension that will be used to calculate N and M of the dense matrices 
        *               (allow to create series of matrices with the same N and M but different sparsities)
        * @param skip: [true/false] if true, ask-if-sizes-are-ok is skipped (can be used for generation scripts)
        * @out out_size_written: size in bytes written to the binary file
        * @returns string of the absolute path where the binary file was stored
        *
        * @warning Dimensionality of matrices are expected to match each operation used, i.e.
        *  1) If X in R^{N x K}, then Y must be in R^{K x M}
        *  2) A_sparse must be in R^{N x K}
        *
        * @sa
        * - [COO matrix format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO))
        * - [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
        */
        static std::string huge_generator(
            std::string target_folder,
            Types::vec_size_t K,
            Types::vec_size_t sizeof_X_in_byte,
            Types::vec_size_t sizeof_Y_in_byte,
            float S_sparsity,
            Types::vec_size_t K_row,
            bool skip,
            uint64_t& out_size_written
        ){
            if(!(target_folder.back() == '/' || target_folder.back() == '\\')){
                std::string msg = std::string("target_folder path must end with a valid path-separator (") +
                    Defines::path_separator +
                    std::string(" on current OS)!");
                TEXT::Gadgets::print_colored_text_line(msg, TEXT::RED);
                throw std::runtime_error(msg);
            }

            Types::vec_size_t N = sizeof_X_in_byte / sizeof(Types::expmt_t) / K_row;
            Types::vec_size_t M = sizeof_Y_in_byte / sizeof(Types::expmt_t) / K_row;
            double nnz = static_cast<double>(N)*static_cast<double>(M)*(1.0 - S_sparsity);
            double x_mb = static_cast<double>(N)*static_cast<double>(K)/1024.0/1024*sizeof(Types::expmt_t);
            double y_mb = static_cast<double>(M)*static_cast<double>(K)/1024.0/1024*sizeof(Types::expmt_t);
            double s_mb = nnz/1024.0/1024*(sizeof(Types::expmt_t)+2*sizeof(Types::vec_size_t));

            TEXT::Gadgets::print_colored_text_line(
                std::string("Generating\n") +
                std::string("...dense X:  [") + std::to_string(N) + std::string(" x ") + std::to_string(K) + std::string("], ") + std::to_string(x_mb) + std::string("MB\n") +
                std::string("...dense Y:  [") + std::to_string(K) + std::string(" x ") + std::to_string(M) + std::string("], ") + std::to_string(y_mb) + std::string("MB\n") +
                std::string("...sparse S: [") + std::to_string(N) + std::string(" x ") + std::to_string(M) + std::string("] with sparsity ") 
                        + std::to_string(S_sparsity) + std::string(", approx ") + std::to_string(nnz) + std::string(" nnz values, ") + std::to_string(s_mb) + std::string("MB\n") +
                std::string("total required size: ") + std::to_string(x_mb + y_mb + s_mb) + std::string("MB\n") +
                std::string("using K_row: ") + std::to_string(K_row),
                TEXT::BRIGHT_MAGENTA
            );

            if (Types::COO::no_filter_condition(S_sparsity, N, M)) {
                std::cout << std::endl;
                TEXT::Gadgets::print_colored_line(100, '>', TEXT::HIGHLIGHT_YELLOW);
                TEXT::Gadgets::print_colored_text_line("WARNING: infeasible to filter coordinates of sparse matrix!", TEXT::RED);
                TEXT::Gadgets::print_colored_line(100, '<', TEXT::HIGHLIGHT_YELLOW);
                std::cout << std::endl;
                return "";
            }

            TEXT::Gadgets::print_colored_line(100, '=', TEXT::RED);
            TEXT::Gadgets::print_colored_text_line("Proceed? [y/n]", TEXT::RED);
            if (!skip) {
                char ans = 'n';
                std::cin >> ans;
                if (!(ans == 'y')) {
                    return "";
                }
            }
            
            return huge_generator_gen(
                target_folder, N, M, K, S_sparsity, 0.0f, 0.0f, out_size_written
            );
        }

        /**
        * @brief Generates uniformly distributed sparse matrix and two fitting dense matrices A and B for
        * SDDMM (A @ B).Hadamard(matrix market file). It takes an inner dimension and the outer dimension of A and B. The goal is to create random matrices
        * with specific dimensions to create a companion matrix for an existing sparse matrix in matrix market format.
        *
        * @param target_folder: folder where result will be stored (must end with a path separator, make sure to use the correct one for the operating system)
        * @param K: inner dimension that will be used for the two dense matrices A and B (N and M will be the ones from the loaded matrix market file)
        * @param N: outer dimension of dense matrix A
        * @param M: outer dimension of dense matrix B
        * @param S_sparsity: proportion of zeros in sparse matrix (0 == no zeros, 1.0 all zeros)
        * @param skip: [true/false] if true, ask-if-sizes-are-ok is skipped (can be used for generation scripts)
        * @out out_size_written: size in bytes written to the binary file
        * @returns string of the absolute path where the binary file was stored
        *
        * @warning Dimensionality of matrices are expected to match each operation used, i.e.
        *  1) If X in R^{N x K}, then Y must be in R^{K x M}
        *  2) A_sparse must be in R^{N x K}
        *
        * @sa
        * - [COO matrix format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO))
        * - [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
        */
        static std::string huge_generator_companion(
            std::string target_folder,
            Types::vec_size_t K,
            Types::vec_size_t N,
            Types::vec_size_t M,
            float S_sparsity,
            bool skip,
            uint64_t& out_size_written
        ) {
            if (!(target_folder.back() == '/' || target_folder.back() == '\\')) {
                std::string msg = std::string("target_folder path must end with a valid path-separator (") +
                    Defines::path_separator +
                    std::string(" on current OS)!");
                TEXT::Gadgets::print_colored_text_line(msg, TEXT::RED);
                throw std::runtime_error(msg);
            }

            double nnz = static_cast<double>(N) * static_cast<double>(M) * (1.0 - S_sparsity);
            double x_mb = static_cast<double>(N) * static_cast<double>(K) / 1024.0 / 1024 * sizeof(Types::expmt_t);
            double y_mb = static_cast<double>(M) * static_cast<double>(K) / 1024.0 / 1024 * sizeof(Types::expmt_t);
            double s_mb = nnz / 1024.0 / 1024 * (sizeof(Types::expmt_t) + 2 * sizeof(Types::vec_size_t));

            TEXT::Gadgets::print_colored_text_line(
                std::string("Generating\n") +
                std::string("...dense X:  [") + std::to_string(N) + std::string(" x ") + std::to_string(K) + std::string("], ") + std::to_string(x_mb) + std::string("MB\n") +
                std::string("...dense Y:  [") + std::to_string(K) + std::string(" x ") + std::to_string(M) + std::string("], ") + std::to_string(y_mb) + std::string("MB\n") +
                std::string("...sparse S: [") + std::to_string(N) + std::string(" x ") + std::to_string(M) + std::string("] with sparsity ")
                + std::to_string(S_sparsity) + std::string(", approx ") + std::to_string(nnz) + std::string(" nnz values, ") + std::to_string(s_mb) + std::string("MB\n") +
                std::string("total required size: ") + std::to_string(x_mb + y_mb + s_mb) + std::string("MB\n"),
                TEXT::BRIGHT_MAGENTA
            );

            if (Types::COO::no_filter_condition(S_sparsity, N, M)) {
                std::cout << std::endl;
                TEXT::Gadgets::print_colored_line(100, '>', TEXT::HIGHLIGHT_YELLOW);
                TEXT::Gadgets::print_colored_text_line("WARNING: infeasible to filter coordinates of sparse matrix!", TEXT::RED);
                TEXT::Gadgets::print_colored_line(100, '<', TEXT::HIGHLIGHT_YELLOW);
                std::cout << std::endl;
                return "";
            }

            TEXT::Gadgets::print_colored_line(100, '=', TEXT::RED);
            TEXT::Gadgets::print_colored_text_line("Proceed? [y/n]", TEXT::RED);
            if (!skip) {
                char ans = 'n';
                std::cin >> ans;
                if (!(ans == 'y')) {
                    return "";
                }
            }

            return huge_generator_gen(
                target_folder, N, M, K, S_sparsity, 0.0f, 0.0f, out_size_written
            );
        }

        /**
        * @brief Generates uniformly distributed sparse matrix and two fitting dense matrices A and B for
        * SDDMM (X @ Y).Hadamard(matrix market file). It takes an inner dimension and the outer dimension of X and Y. The goal is to create random matrices
        * with specific dimensions to create a companion matrix for an existing sparse matrix in matrix market format.
        *
        * @param target_folder: folder where result will be stored (must end with a path separator, make sure to use the correct one for the operating system)
        * @param K: inner dimension that will be used for the two dense matrices A and B (N and M will be the ones from the loaded matrix market file)
        * @param N: outer dimension of dense matrix A
        * @param M: outer dimension of dense matrix B
        * @param S_sparsity: proportion of zeros in sparse matrix (0 == no zeros, 1.0 all zeros)
        * @param X_sparsity: proportion of zeros in dense matrix X (0 == no zeros, 1.0 all zeros)
        * @param Y_sparsity: proportion of zeros in dense matrix Y (0 == no zeros, 1.0 all zeros)
        * @out out_size_written: size in bytes written to the binary file
        * @returns string of the absolute path where the binary file was stored
        *
        * @warning Dimensionality of matrices are expected to match each operation used, i.e.
        *  1) If X in R^{N x K}, then Y must be in R^{K x M}
        *  2) A_sparse must be in R^{N x K}
        *
        * @sa
        * - [COO matrix format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO))
        * - [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
        */
        static std::string huge_generator_gen(
            std::string target_folder, 
            Types::vec_size_t N, Types::vec_size_t M, Types::vec_size_t K,
            float S_sparsity, float X_sparsity, float Y_sparsity, uint64_t& out_size_written
        ){
            if(!(target_folder.back() == '/' || target_folder.back() == '\\')){
                throw std::runtime_error(
                    std::string("target_folder path must end with a valid path-separator (") + 
                    Defines::path_separator + 
                    std::string(" on current OS)!")
                );
            }

            std::vector<std::chrono::high_resolution_clock::time_point> ts;
            std::vector<std::string> ts_labels;

            ts.push_back(std::chrono::high_resolution_clock::now());
            ts_labels.push_back("start");
            auto X = SDDMM::Types::Matrix::generate_row_major(N, K, X_sparsity, -1.f, 1.f, true, 40000);
            ts.push_back(std::chrono::high_resolution_clock::now());
            ts_labels.push_back("generate X");

            auto Y = SDDMM::Types::Matrix::generate_col_major(K, M, Y_sparsity, -1.f, 1.f, true, 40000);
            ts.push_back(std::chrono::high_resolution_clock::now());
            ts_labels.push_back("generate Y");

            auto coo_mat = SDDMM::Types::COO::generate_row_major_curand(N, M, S_sparsity, true, 40000);
            ts.push_back(std::chrono::high_resolution_clock::now());
            ts_labels.push_back("generate S mat");

            TEXT::Gadgets::print_colored_text_line("Write to file:", TEXT::BRIGHT_CYAN);
            std::string name = SDDMM::Types::COO::hadamard_to_bin_file(
                target_folder, coo_mat, S_sparsity, X, X_sparsity, Y, Y_sparsity, out_size_written);
            ts.push_back(std::chrono::high_resolution_clock::now());
            ts_labels.push_back("to file");

            TEXT::Gadgets::print_colored_text_line("Time results:", TEXT::BRIGHT_CYAN);
            for(int i=1; i<ts.size(); ++i){
                auto duration = std::chrono::duration_cast<SDDMM::Types::time_measure_unit>(ts[i]-ts[i-1]).count();
                TEXT::Gadgets::print_colored_text_line(ts_labels[i] + ":\t" + std::to_string(static_cast<double>(duration) / 1000.0) + std::string("ms"), TEXT::BRIGHT_BLUE);
            }

            return name;

            // SDDMM::Types::COO out_coo;
            // SDDMM::Types::CSR out_csr;
            // SDDMM::Types::Matrix out_X(0,0);
            // SDDMM::Types::Matrix out_Y(0,0);
            // float out_sparse_sparsity;
            // float out_x_sparsity;
            // float out_y_sparsity;
            // SDDMM::Types::COO::hadamard_from_bin_file(
            //     target_folder + name,
            //     out_coo, out_csr, out_sparse_sparsity, 
            //     out_X, out_x_sparsity, 
            //     out_Y, out_y_sparsity);
        }


    };
}
