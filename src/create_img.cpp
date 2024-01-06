#include <iostream>
#include <algorithm>
#include "defines.h"
#include "data_structures\coo\coo.h"
#include "data_structures\csr\csr.h"
#include "data_structures\matrix\matrix.h"
#include "mat_to_img\mat_to_img.h"

/**
 * This is the algo that will run
*/

using namespace SDDMM;

std::vector<std::string> split(const std::string& matrix, const char delimiter) {
    std::vector<std::string> res;
    std::string temp = "";
    for (const auto& s : matrix) {
        if (s == delimiter) {
            res.push_back(temp);
            temp = "";
            continue;
        }
        temp += s;
    }
    if (!temp.empty()) {
        res.push_back(temp);
    }

    return res;
}

int main(int argc, char** argv) {

    ////TEXT::Gadgets::print_colored_text_line(std::string("Running test ") + std::string(argv[1]) + std::string("..."), TEXT::BLUE);
    //int size_r = 1000;
    //int size_c = 1200;
    //float sparsity = 0.9f;
    //TEXT::Gadgets::print_colored_text_line("...Generate matrix...", TEXT::BRIGHT_GREEN);
    //auto mat2 = SDDMM::Types::COO::generate_row_major_curand(size_r, size_c, sparsity);
    //TEXT::Gadgets::print_colored_text_line("...Generate matrix: finished", TEXT::BRIGHT_GREEN);

    //int bins_r = static_cast<int>(std::roundf(size_r * (1 - sparsity) / 2.1));
    //int bins_c = static_cast<int>(std::roundf(size_c * (1 - sparsity) / 2.1));
    //TEXT::Gadgets::print_colored_text_line(std::string("...Use bin count [") + std::to_string(bins_r) + std::string(",") + std::to_string(bins_c) + std::string("]"), TEXT::BRIGHT_GREEN);
    ////std::string img_name = std::string("./__") + std::string(argv[1]) + std::string(".jpg");
    //std::string img_name = "__lol.jpg";
    //TEXT::Gadgets::print_colored_text_line(std::string("...Generate image ") + img_name + std::string("..."), TEXT::BRIGHT_GREEN);
    //bool success = Image().get_img(img_name, bins_r, bins_c, mat2);
    //TEXT::Gadgets::print_colored_text_line(std::string("...Generate image ") + img_name + std::string(": finished..."), TEXT::BRIGHT_GREEN);

    //TEXT::Gadgets::print_colored_text_line("...Finished", TEXT::GREEN);
    //return 0;

    if (argc != 5 && argc != 2) {
        TEXT::Gadgets::print_colored_line(100, '=', TEXT::GREEN);
        std::cout << std::endl;

        TEXT::Gadgets::print_colored_text_line("Usage:", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 1: path to storage or dense matrix to transform for example [1,2;3,4] (in quotes if it contains spaces)", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 2: width of img [px]", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 3: height of img [px]", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 4: name of img (in quotes if it contains spaces)", TEXT::BLUE);

        TEXT::Gadgets::print_colored_text_line("OR", TEXT::HIGHLIGHT_GREEN);

        TEXT::Gadgets::print_colored_text_line("Usage:", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 1: test1 or test2", TEXT::BLUE);

        std::cout << std::endl;
        TEXT::Gadgets::print_colored_line(100, '=', TEXT::GREEN);

        return 0;
    }

    if (argc == 2) {
        if (std::string(argv[1]).compare("test1") == 0) {
            TEXT::Gadgets::print_colored_text_line(std::string("Running test ") + std::string(argv[1]) + std::string("..."), TEXT::BLUE);
            int size_r = 1000;
            int size_c = 2000;
            std::vector<float> mat2_data(size_r * size_c, 0);
            int j = 1;
            for (int r = 0; r < size_r; ++r) {
                int i = 0;
                for (int c = 0; c < j; ++c) {
                    mat2_data[r * size_c + c] = j + i;
                    i++;
                }
                j++;
            }

            for (int r = 0; r < size_r; ++r) {
                int i = 0;
                for (int c = size_c - 100; c < size_c; ++c) {
                    mat2_data[r * size_c + c] = size_c;
                    i++;
                }
            }

            SDDMM::Types::Matrix mat2 = SDDMM::Types::Matrix::deterministic_gen_row_major(size_r, size_c, mat2_data);
            bool success = Image().get_img(std::string("./__") + std::string(argv[1]) + std::string(".jpg"), mat2);

            TEXT::Gadgets::print_colored_text_line("...Finished", TEXT::GREEN);
            return 0;
        }
        if (std::string(argv[1]).compare("test2") == 0) {
            TEXT::Gadgets::print_colored_text_line(std::string("Running test ") + std::string(argv[1]) + std::string("..."), TEXT::BLUE);
            int size_r = 1000;
            int size_c = 1200;
            float sparsity = 0.9f;
            TEXT::Gadgets::print_colored_text_line("...Generate matrix...", TEXT::BRIGHT_GREEN);
            auto mat2 = SDDMM::Types::COO::generate_row_major_curand(size_r, size_c, sparsity);
            TEXT::Gadgets::print_colored_text_line("...Generate matrix: finished", TEXT::BRIGHT_GREEN);

            int bins_r = static_cast<int>(std::roundf(size_r * (1 - sparsity) / 2.1));
            int bins_c = static_cast<int>(std::roundf(size_c * (1 - sparsity) / 2.1));
            TEXT::Gadgets::print_colored_text_line(std::string("...Use bin count [") + std::to_string(bins_r) + std::string(",") + std::to_string(bins_c) + std::string("]"), TEXT::BRIGHT_GREEN);
            std::string img_name = std::string("./__") + std::string(argv[1]) + std::string(".jpg");
            TEXT::Gadgets::print_colored_text_line(std::string("...Generate image ") + img_name + std::string("..."), TEXT::BRIGHT_GREEN);
            bool success = Image().get_img(img_name, bins_r, bins_c , mat2);
            TEXT::Gadgets::print_colored_text_line(std::string("...Generate image ") + img_name + std::string(": finished..."), TEXT::BRIGHT_GREEN);

            TEXT::Gadgets::print_colored_text_line("...Finished", TEXT::GREEN);
            return 0;
        }
    }

    std::string src = argv[1];
    int width = std::atoi(argv[2]);
    int height = std::atoi(argv[3]);
    std::string name = argv[4];

    if (src.find("[") != std::string::npos) {
        // we have a matrix => convert to COO and stuff into coo_mat

        src.erase(std::remove(src.begin(), src.end(), ' '), src.end());
        std::string matrix = src.substr(1, src.size() - 2);
        
        auto matrix_rows = split(matrix, ';');
        int n = matrix_rows.size();
        int m = 0;
        std::vector<float> values;
        for (const auto& row : matrix_rows) {
            std::cout << row << std::endl;
            auto res = split(row, ',');
            if (m == 0) m = res.size();
            else if (res.size() != m) {
                TEXT::Gadgets::print_colored_text_line("All rows must be the same length!", TEXT::HIGHLIGHT_RED);
                return 0;
            }
            for (const auto& s : res) {
                values.push_back(std::atof(s.c_str()));
            }
        }

        SDDMM::Types::Matrix mat = SDDMM::Types::Matrix::deterministic_gen_row_major(n, m, values);
        TEXT::Gadgets::print_colored_text_line("Supplied matrix:", TEXT::GREEN);
        std::cout << mat << std::endl;

        //TEXT::Gadgets::print_colored_text_line("Start conversion...", TEXT::BLUE);
        bool success = Image().get_img(name, mat);

        if (!success) {
            TEXT::Gadgets::print_colored_text_line("Error!", TEXT::HIGHLIGHT_RED);
        }
    }
    else {
        // we have a path
        TEXT::Gadgets::print_colored_line(100, '*', TEXT::HIGHLIGHT_YELLOW);
        std::cout << TEXT::Cast::Cyan("...loading data...") << std::endl;
        SDDMM::Types::COO coo_mat;
        SDDMM::Types::CSR csr_mat;
        SDDMM::Types::Matrix X(0, 0);
        SDDMM::Types::Matrix Y(0, 0);
        float sparse_sparsity;
        float X_sparsity;
        float Y_sparsity;
        uint64_t out_size_read;
        SDDMM::Types::COO::hadamard_from_bin_file(
            name,
            coo_mat, csr_mat, sparse_sparsity,
            X, X_sparsity,
            Y, Y_sparsity,
            out_size_read);

        Types::vec_size_t N = X.n;
        Types::vec_size_t M = Y.m;
        Types::vec_size_t K = X.m;

        std::cout << TEXT::Cast::Cyan(
            std::string("...stats:\n") +
            std::string("......N:        ") + std::to_string(N) + std::string("\n") +
            std::string("......M:        ") + std::to_string(M) + std::string("\n") +
            std::string("......K:        ") + std::to_string(K) + std::string("\n") +
            std::string("......sparsity: ") + std::to_string(sparse_sparsity)) << std::endl;
    }

    TEXT::Gadgets::print_colored_text_line(std::string("File [") + name + std::string("] saved!"), TEXT::BLUE);
    std::cout << std::endl;
    TEXT::Gadgets::print_colored_line(100, '=', TEXT::GREEN);

    return 0;
}
