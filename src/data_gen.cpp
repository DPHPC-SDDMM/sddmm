#include <iostream>
#include "defines.h"
#include "sddmm_data_gen/huge_gen.cpp"

/**
 * This is the algo that will run
*/

using namespace SDDMM;

int main(int argc, char** argv) {

    if (argc < 6 || argc > 7) {
        TEXT::Gadgets::print_colored_line(100, '=', TEXT::GREEN);
        std::cout << std::endl;

        TEXT::Gadgets::print_colored_text_line("Usage:", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 1: path to storage (in quotes if it contains spaces)", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 2: sizeof K (inner dimension, like 128)", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 3: sizeof dense A in byte", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 4: sizeof dense B in byte", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 5: sparsity of S (0.999 is 99.9% of entries are zero, 0.1 is 90% of entries are NOT zero)", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 6: sizeof row-determining K (if not given, param 2 is used)", TEXT::BLUE);

        std::cout << std::endl;
        TEXT::Gadgets::print_colored_text_line("Param 6 can be set in order to produce matrices with the same N,M as matrices with larger K.", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("For example, generate [NxK1, K1xM, NxM] matrix tripel with K1>K2, then generate tripel", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("[NxK2, K2xM, NxM]. In this case, pass K1 as Param 6 and K2 as Param 2.", TEXT::BLUE);

        std::cout << std::endl;
        TEXT::Gadgets::print_colored_text_line("Check the indicated sizes before confirming...", TEXT::RED);
        TEXT::Gadgets::print_colored_text_line("Indicated file sizes are the best fit with given sizes and data types", TEXT::BLUE);

        std::cout << std::endl;
        TEXT::Gadgets::print_colored_line(100, '=', TEXT::GREEN);

        return 0;
    }
    std::cout << "lol" << std::endl;
    std::string path = std::string(argv[1]);
    Types::vec_size_t K = std::atoi(argv[2]);
    Types::vec_size_t sizeof_X_in_byte = std::atoi(argv[3]);
    Types::vec_size_t sizeof_Y_in_byte = std::atoi(argv[4]);
    float S_sparsity = std::atof(argv[5]);

    if (S_sparsity >= 1.0f) {
        std::cout << std::endl;
        TEXT::Gadgets::print_colored_line(100, '>', TEXT::HIGHLIGHT_YELLOW);
        TEXT::Gadgets::print_colored_text_line("ERROR: sparsity must be in [0, 1.0)!", TEXT::RED);
        TEXT::Gadgets::print_colored_line(100, '<', TEXT::HIGHLIGHT_YELLOW);
        std::cout << std::endl;
        return 0;
    }

    Types::vec_size_t K_row = K;
    bool eliminate_doubles = true;
    if (argc == 7) {
        K_row = std::atoi(argv[6]);
    }

    std::cout << path << " " << K << " " << sizeof_X_in_byte
        << " " << sizeof_Y_in_byte << " " << S_sparsity
        << " " << K_row
        << std::endl;

    //if (argc >= 8) {
    //    int b = std::atoi(argv[7]);
    //    if (b == 1) eliminate_doubles = false;
    //}

    if (K % 32 != 0) {
        std::cout << std::endl;
        TEXT::Gadgets::print_colored_line(100, '>', TEXT::HIGHLIGHT_YELLOW);
        TEXT::Gadgets::print_colored_text_line("WARNING: K should be a multiple of 32 but isn't!", TEXT::RED);
        TEXT::Gadgets::print_colored_line(100, '<', TEXT::HIGHLIGHT_YELLOW);
        std::cout << std::endl;
    }

    uint64_t out_size_written;
    std::string name = DataGenerator::huge_generator(path, K, sizeof_X_in_byte, sizeof_Y_in_byte, S_sparsity, K_row, out_size_written);

    TEXT::Gadgets::print_colored_text_line(std::to_string(out_size_written) + std::string(" bytes written..."), TEXT::BLUE);
    TEXT::Gadgets::print_colored_text_line(std::string("File [") + name + std::string("] saved!"), TEXT::BLUE);
    std::cout << std::endl;
    TEXT::Gadgets::print_colored_line(100, '=', TEXT::GREEN);

    return 0;
}
