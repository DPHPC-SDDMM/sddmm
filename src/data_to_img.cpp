#include <iostream>
#include "defines.h"
#include "data_structures\coo\coo.h"
#include "data_structures\csr\csr.h"
#include "data_structures\matrix\matrix.h"
#include "toojpeg-master\toojpeg.h"

/**
 * This is the algo that will run
*/

using namespace SDDMM;

void init(unsigned char* img, int img_width, int img_height) {
    int64_t s = img_width * img_height * 3;
    for (int64_t ind = 0; ind < s; ++ind) {
        img[ind] = static_cast<char>(0);
    }
}

// source: https://bottosson.github.io/posts/oklab/
void linear_srgb_to_oklab(unsigned char rgb_r, unsigned char rgb_g, unsigned char rgb_b, float& oklab_L, float& oklab_a, float& oklab_b)
{
    float l = 0.4122214708f * static_cast<float>(rgb_r)/255.0f + 0.5363325363f * static_cast<float>(rgb_g)/255.0f + 0.0514459929f * static_cast<float>(rgb_b)/255.0f;
    float m = 0.2119034982f * static_cast<float>(rgb_r)/255.0f + 0.6806995451f * static_cast<float>(rgb_g)/255.0f + 0.1073969566f * static_cast<float>(rgb_b)/255.0f;
    float s = 0.0883024619f * static_cast<float>(rgb_r)/255.0f + 0.2817188376f * static_cast<float>(rgb_g)/255.0f + 0.6299787005f * static_cast<float>(rgb_b)/255.0f;

    float l_ = cbrtf(l);
    float m_ = cbrtf(m);
    float s_ = cbrtf(s);

    
   oklab_L = 0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_;
   oklab_a = 1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_;
   oklab_b = 0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_;
}

// source: https://bottosson.github.io/posts/oklab/
void oklab_to_linear_srgb(float oklab_L, float oklab_a, float oklab_b, unsigned char& rgb_r, unsigned char& rgb_g, unsigned char& rgb_b)
{
    float l_ = oklab_L + 0.3963377774f * oklab_a + 0.2158037573f * oklab_b;
    float m_ = oklab_L - 0.1055613458f * oklab_a - 0.0638541728f * oklab_b;
    float s_ = oklab_L - 0.0894841775f * oklab_a - 1.2914855480f * oklab_b;

    float l = l_ * l_ * l_;
    float m = m_ * m_ * m_;
    float s = s_ * s_ * s_;

    rgb_r = static_cast<unsigned char>((4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s)*255.0f);
    rgb_g = static_cast<unsigned char>((-1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s)*255.0f);
    rgb_b = static_cast<unsigned char>((-0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s)*255.0f);
}

void set(unsigned char* img, int img_width, int img_height, SDDMM::Types::COO& mat) {
    int bin_size_w = static_cast<int>(std::ceil(static_cast<double>(mat.m) / static_cast<double>(img_width)));
    int bin_size_h = static_cast<int>(std::ceil(static_cast<double>(mat.n) / static_cast<double>(img_height)));

    uint64_t S = mat.cols.size();
    uint64_t hs = img_width * img_height;
    int64_t* vals = new int64_t[hs];
    for (uint64_t s = 0; s < hs; ++s) {
        vals[s] = 0;
    }

    int64_t max = 0;
    for (uint64_t s = 0; s < S; ++s) {
        int bin_n = mat.cols[s] / bin_size_h;
        int bin_m = mat.rows[s] / bin_size_w;

        uint64_t ind = bin_n * bin_m;
        int64_t cur = vals[ind];
        cur++;
        if (cur > max) {
            max = cur;
        }
        vals[ind] = cur;
    }



    delete[] vals;
}

int main(int argc, char** argv) {

    if (argc != 5) {
        TEXT::Gadgets::print_colored_line(100, '=', TEXT::GREEN);
        std::cout << std::endl;

        TEXT::Gadgets::print_colored_text_line("Usage:", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 1: path to storage (in quotes if it contains spaces)", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 2: width of img [px]", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 3: height of img [px]", TEXT::BLUE);
        TEXT::Gadgets::print_colored_text_line("Param 4: name of img (in quotes if it contains spaces)", TEXT::BLUE);

        std::cout << std::endl;
        TEXT::Gadgets::print_colored_line(100, '=', TEXT::GREEN);

        return 0;
    }

    std::string src = argv[1];
    int width = std::atoi(argv[2]);
    int height = std::atoi(argv[3]);
    std::string name = argv[4];

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

    TEXT::Gadgets::print_colored_text_line("Start conversion...", TEXT::BLUE);
    auto pixels = new unsigned char[width * height * 3];
    init(pixels, width, height);

    TEXT::Gadgets::print_colored_text_line(std::string("File [") + name + std::string("] saved!"), TEXT::BLUE);
    std::cout << std::endl;
    TEXT::Gadgets::print_colored_line(100, '=', TEXT::GREEN);

    return 0;
}
