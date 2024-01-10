#include <iostream>
#include "defines.h"
#include "sddmm_data_gen/huge_gen.cpp"

/**
 * This is the algo that will run
*/

using namespace SDDMM;

int main(int argc, char** argv) {

    if (argc != 4) {
        TEXT::Gadgets::print_colored_line(100, '=', TEXT::GREEN);
        std::cout << std::endl;

        TEXT::Gadgets::print_colored_text_line("Usage:", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 1: path to storage (in quotes if it contains spaces)", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 2: path to matrix market file", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 3: sizeof K (inner dimension, like 128)", TEXT::BLUE);

        std::cout << std::endl;
        TEXT::Gadgets::print_colored_text_line("Check the indicated sizes before confirming...", TEXT::RED);
        TEXT::Gadgets::print_colored_text_line("Indicated file sizes are the best fit with given sizes and data types", TEXT::BLUE);

        std::cout << std::endl;
        TEXT::Gadgets::print_colored_line(100, '=', TEXT::GREEN);

        return 0;
    }

    std::string path = std::string(argv[1]);
    std::string mm_path = std::string(argv[2]);
    Types::vec_size_t K = std::atoi(argv[3]);

    std::cout << path << " " << mm_path << " " << K << std::endl;

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
    std::string name = DataGenerator::huge_generator_matrix_market(path, mm_path, K, out_size_written);

    TEXT::Gadgets::print_colored_text_line(std::to_string(out_size_written) + std::string(" bytes written..."), TEXT::BLUE);
    TEXT::Gadgets::print_colored_text_line(std::string("File [") + name + std::string("] saved!"), TEXT::BLUE);
    std::cout << std::endl;
    TEXT::Gadgets::print_colored_line(100, '=', TEXT::GREEN);

    return 0;
}
