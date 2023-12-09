#include <iostream>
#include <vector>
#include <chrono>
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

        static std::string huge_generator(
            std::string target_folder,
            Types::vec_size_t K,
            Types::vec_size_t sizeof_X_in_byte,
            Types::vec_size_t sizeof_Y_in_byte,
            float S_sparsity
        ){
            if(target_folder.back() != Defines::path_separator){
                throw std::runtime_error(
                    std::string("target_folder path must end with a valid path-separator (") + 
                    Defines::path_separator + 
                    std::string(" on current OS)!")
                );
            }

            Types::vec_size_t N = sizeof_X_in_byte / sizeof(Types::expmt_t) / K;
            Types::vec_size_t M = sizeof_Y_in_byte / sizeof(Types::expmt_t) / K;
            double nnz = static_cast<double>(N)*static_cast<double>(M)*(1.0 - S_sparsity);
            double x_mb = static_cast<double>(N)*static_cast<double>(K)/1024.0/1024*sizeof(Types::expmt_t);
            double y_mb = static_cast<double>(M)*static_cast<double>(K)/1024.0/1024*sizeof(Types::expmt_t);
            double s_mb = nnz/1024.0/1024*(sizeof(Types::expmt_t)+2*sizeof(Types::vec_size_t));
            TEXT::Gadgets::print_colored_text_line(
                std::string("Generating\n") +
                std::string("...dense X:  [") + std::to_string(N) + std::string(" x ") + std::to_string(K) + std::string("], ") + std::to_string(x_mb) + std::string("MB\n") +
                std::string("...dense Y:  [") + std::to_string(K) + std::string(" x ") + std::to_string(M) + std::string("], ") + std::to_string(y_mb) + std::string("MB\n") +
                std::string("...sparse S: [") + std::to_string(N) + std::string(" x ") + std::to_string(M) + std::string("] with sparsity ") 
                        + std::to_string(S_sparsity) + std::string(", approx ") + std::to_string(nnz) + std::string(" nnz values, ") + std::to_string(s_mb) + std::string("MB"),
                TEXT::BRIGHT_MAGENTA
            );

            TEXT::Gadgets::print_colored_line(100, '=', TEXT::RED);
            TEXT::Gadgets::print_colored_text_line("Proceed? [y/n]", TEXT::RED);
            char ans = 'n';
            std::cin >> ans;
            if(!(ans == 'y')){
                return "";
            }
            
            return huge_generator(
                target_folder, N, M, K, S_sparsity, 0.0f, 0.0f  
            );
        }

        static std::string huge_generator(
            std::string target_folder, 
            Types::vec_size_t N, Types::vec_size_t M, Types::vec_size_t K,
            float S_sparsity, float X_sparsity, float Y_sparsity
        ){
            if(target_folder.back() != Defines::path_separator){
                throw std::runtime_error(
                    std::string("target_folder path must end with a valid path-separator (") + 
                    Defines::path_separator + 
                    std::string(" on current OS)!")
                );
            }

            std::vector<std::chrono::_V2::system_clock::time_point> ts;
            std::vector<std::string> ts_labels;

            ts.push_back(std::chrono::high_resolution_clock::now());
            ts_labels.push_back("start");
            auto X = SDDMM::Types::Matrix::generate_row_major(N, K, X_sparsity, -1.f, 1.f, true, 10000);
            ts.push_back(std::chrono::high_resolution_clock::now());
            ts_labels.push_back("generate X");

            auto Y = SDDMM::Types::Matrix::generate_col_major(K, M, Y_sparsity, -1.f, 1.f, true, 10000);
            ts.push_back(std::chrono::high_resolution_clock::now());
            ts_labels.push_back("generate Y");

            auto coo_mat = SDDMM::Types::COO::generate_row_major_unsorted(N, M, S_sparsity, -1.f, 1.f, true, 10000);
            ts.push_back(std::chrono::high_resolution_clock::now());
            ts_labels.push_back("generate COO S mat");

            std::cout << "1" << std::endl;
            auto csr_mat = coo_mat.to_csr();
            std::cout << "2" << std::endl;
            ts.push_back(std::chrono::high_resolution_clock::now());
            ts_labels.push_back("to csr_mat");

            std::cout << "3" << std::endl;
            std::string name = SDDMM::Types::COO::hadamard_to_bin_file(
                target_folder, coo_mat, S_sparsity, X, X_sparsity, Y, Y_sparsity);
            ts.push_back(std::chrono::high_resolution_clock::now());
            ts_labels.push_back("to file");
            std::cout << "4" << std::endl;

            TEXT::Gadgets::print_colored_text_line("Time results:", TEXT::BRIGHT_CYAN);
            std::cout << "5" << std::endl;
            for(int i=1; i<ts.size(); ++i){
                auto duration = std::chrono::duration_cast<SDDMM::Types::time_measure_unit>(ts[i]-ts[i-1]).count();
                TEXT::Gadgets::print_colored_text_line(ts_labels[i] + ":\t" + std::to_string(static_cast<double>(duration) / 1000.0) + std::string("ms"), TEXT::BRIGHT_BLUE);
            }
            std::cout << "6" << std::endl;

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
