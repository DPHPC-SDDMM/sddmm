#include <iostream>
#include "defines.h"
#include "sddmm_data_gen/huge_gen.cpp"

/**
 * This is the algo that will run
*/

using namespace SDDMM;

int main(int argc, char** argv) {

    if (argc != 7) {
        TEXT::Gadgets::print_colored_line(100, '=', TEXT::GREEN);
        std::cout << std::endl;

        TEXT::Gadgets::print_colored_text_line("Usage:", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 1: path to storage (in quotes if it contains spaces)", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 2: sizeof K (inner dimension, like 128)", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 3: sizeof N", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 4: sizeof M", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 5: sparsity of S (0.999 is 99.9% of entries are zero, 0.1 is 90% of entries are NOT zero)", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 6: skip=true or skip=false (skip asking the user if input is ok or not)", TEXT::BLUE);

        std::cout << std::endl;
        TEXT::Gadgets::print_colored_text_line("Check the indicated sizes before confirming...", TEXT::RED);
        TEXT::Gadgets::print_colored_text_line("Indicated file sizes are the best fit with given sizes and data types", TEXT::BLUE);

        std::cout << std::endl;
        TEXT::Gadgets::print_colored_line(100, '=', TEXT::GREEN);

        return 0;
    }

    std::string path = std::string(argv[1]);
    Types::vec_size_t K = std::atoi(argv[2]);
    Types::vec_size_t N = std::atoi(argv[3]);
    Types::vec_size_t M = std::atoi(argv[4]);
    float S_sparsity = std::atof(argv[5]);

    if (S_sparsity >= 1.0f) {
        std::cout << std::endl;
        TEXT::Gadgets::print_colored_line(100, '>', TEXT::HIGHLIGHT_YELLOW);
        TEXT::Gadgets::print_colored_text_line("ERROR: sparsity must be in [0, 1.0)!", TEXT::RED);
        TEXT::Gadgets::print_colored_line(100, '<', TEXT::HIGHLIGHT_YELLOW);
        std::cout << std::endl;
        return 0;
    }

    std::string skip_str = argv[6];
    bool skip = false;
    if (skip_str.compare("skip=true") == 0) {
        skip = true;
    }

    std::cout << skip << std::endl;

    if (K % 32 != 0) {
        std::cout << std::endl;
        TEXT::Gadgets::print_colored_line(100, '>', TEXT::HIGHLIGHT_YELLOW);
        TEXT::Gadgets::print_colored_text_line("WARNING: K should be a multiple of 32 but isn't!", TEXT::RED);
        TEXT::Gadgets::print_colored_line(100, '<', TEXT::HIGHLIGHT_YELLOW);
        std::cout << std::endl;
    }

    uint64_t out_size_written;
    std::string name = DataGenerator::huge_generator_companion(path, K, N, M, S_sparsity, skip, out_size_written);

    TEXT::Gadgets::print_colored_text_line(std::to_string(out_size_written) + std::string(" bytes written..."), TEXT::BLUE);
    TEXT::Gadgets::print_colored_text_line(std::string("File [") + name + std::string("] saved!"), TEXT::BLUE);
    std::cout << std::endl;
    TEXT::Gadgets::print_colored_line(100, '=', TEXT::GREEN);

    return 0;
}
