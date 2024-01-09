
#include <vector>
#include <fstream>

#include "../data_structures\coo\coo.h"
#include "../data_structures\csr\csr.h"
#include "../data_structures\matrix\matrix.h"
#include "../toojpeg-master\toojpeg.h"

namespace SDDMM {
    namespace HackyPrivate {
        std::ofstream _jpegFile;
    }
	class Image {;
        std::vector<unsigned char> _img_buffer;
        std::vector<float> _histogram_buffer;

        static void _output(unsigned char byte) {
            HackyPrivate::_jpegFile << byte;
        }

    public:
        Image() {}

        // source: https://bottosson.github.io/posts/oklab/
        static void linear_srgb_to_oklab(unsigned char rgb_r, unsigned char rgb_g, unsigned char rgb_b, float& oklab_L, float& oklab_a, float& oklab_b)
        {
            float l = 0.4122214708f * static_cast<float>(rgb_r) / 255.0f + 0.5363325363f * static_cast<float>(rgb_g) / 255.0f + 0.0514459929f * static_cast<float>(rgb_b) / 255.0f;
            float m = 0.2119034982f * static_cast<float>(rgb_r) / 255.0f + 0.6806995451f * static_cast<float>(rgb_g) / 255.0f + 0.1073969566f * static_cast<float>(rgb_b) / 255.0f;
            float s = 0.0883024619f * static_cast<float>(rgb_r) / 255.0f + 0.2817188376f * static_cast<float>(rgb_g) / 255.0f + 0.6299787005f * static_cast<float>(rgb_b) / 255.0f;

            float l_ = cbrtf(l);
            float m_ = cbrtf(m);
            float s_ = cbrtf(s);


            oklab_L = 0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_;
            oklab_a = 1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_;
            oklab_b = 0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_;
        }

        // source: https://bottosson.github.io/posts/oklab/
        static void oklab_to_linear_srgb(float oklab_L, float oklab_a, float oklab_b, unsigned char& rgb_r, unsigned char& rgb_g, unsigned char& rgb_b)
        {
            float l_ = oklab_L + 0.3963377774f * oklab_a + 0.2158037573f * oklab_b;
            float m_ = oklab_L - 0.1055613458f * oklab_a - 0.0638541728f * oklab_b;
            float s_ = oklab_L - 0.0894841775f * oklab_a - 1.2914855480f * oklab_b;

            float l = l_ * l_ * l_;
            float m = m_ * m_ * m_;
            float s = s_ * s_ * s_;

            rgb_r = static_cast<unsigned char>((4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s) * 255.0f);
            rgb_g = static_cast<unsigned char>((-1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s) * 255.0f);
            rgb_b = static_cast<unsigned char>((-0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s) * 255.0f);
        }

        static void oklab_lerp(float p, float min_oklab_L, float min_oklab_a, float min_oklab_b, float max_oklab_L, float max_oklab_a, float max_oklab_b, float& oklab_L, float& oklab_a, float& oklab_b) {
            oklab_L = (1.0f - p) * min_oklab_L + p * max_oklab_L;
            oklab_a = (1.0f - p) * min_oklab_a + p * max_oklab_a;
            oklab_b = (1.0f - p) * min_oklab_b + p * max_oklab_b;
        }

        bool get_img(std::string name, SDDMM::Types::Matrix& mat) {
            float sum = 0;
            float max = 0;
            for (uint64_t n = 0; n < mat.n; ++n) {
                for (uint64_t m = 0; m < mat.m; ++m) {
                    if (mat.at(n, m) > max) max = mat.at(n, m);
                    sum += mat.at(n, m);
                }
            }

            // turn value into percent of max
            std::vector<float> data = mat.data;
            for (Types::vec_size_t ind = 0; ind < mat.data.size(); ++ind) {
                data[ind] = data[ind] / max;
            }

            float midpoint_value = sum / static_cast<float>(mat.n) / static_cast<float>(mat.m) / max;
            std::cout << midpoint_value << std::endl;

            return get_img(name, mat.n, mat.m, data, midpoint_value);
        }

        bool get_img(std::string name, uint64_t bin_nr_N, uint64_t bin_nr_M, std::vector<float>& data, float midpoint_value){
            if (midpoint_value < 0 || midpoint_value > 1.0f) {
                TEXT::Gadgets::print_colored_text_line("Midpiont value must be between 0 and 1!", TEXT::HIGHLIGHT_RED);
                return false;
            }

            unsigned char max_r = 255;
            unsigned char max_g = 0;
            unsigned char max_b = 0;

            unsigned char min_r = 0;
            unsigned char min_g = 0;
            unsigned char min_b = 255;

            float min_oklab_L, min_oklab_a, min_oklab_b, max_oklab_L, max_oklab_a, max_oklab_b;
            linear_srgb_to_oklab(max_r, max_g, max_b, max_oklab_L, max_oklab_a, max_oklab_b);
            linear_srgb_to_oklab(min_r, min_g, min_b, min_oklab_L, min_oklab_a, min_oklab_b);

            // do some sort of gamma correction: ensure, midpoint percentage is half way between blue and red
            // => 1 - exp(-a*x) => a = -ln(1-x)/x
            // Note: this function is not completely straignt for x=0.5 but close enough
            float a = -std::logf(1 - 0.5f) / midpoint_value;

            _img_buffer.clear();
            _img_buffer.shrink_to_fit();
            _img_buffer.reserve(bin_nr_N * bin_nr_M * 3);
            for (const float& val : data) {
                if (val < 0 || val > 1.0f) {
                    TEXT::Gadgets::print_colored_text_line("Image must consist of values between 0 and 1!", TEXT::HIGHLIGHT_RED);
                    return false;
                }

                float p = 1.0f - std::expf(-a*val);
                //float p = val;
                if (p < 0.0f) p = 0.0f;
                if (p > 1.0f) p = 1.0f;

                float oklab_L, oklab_a, oklab_b;
                oklab_lerp(p, min_oklab_L, min_oklab_a, min_oklab_b, max_oklab_L, max_oklab_a, max_oklab_b, oklab_L, oklab_a, oklab_b);
                unsigned char r, g, b;
                oklab_to_linear_srgb(oklab_L, oklab_a, oklab_b, r, g, b);
                
                _img_buffer.push_back(r);
                _img_buffer.push_back(g);
                _img_buffer.push_back(b);
            }

            HackyPrivate::_jpegFile = std::ofstream(name, std::ios_base::out | std::ios_base::binary);
            if (!HackyPrivate::_jpegFile.is_open()) return false;

            const int bytesPerPixel = 3;
            const bool isRGB = true;
            const int quality = 100;
            const bool downsample = false;
            const char* comment = name.c_str();
            bool ok = TooJpeg::writeJpeg(_output, _img_buffer.data(), bin_nr_M, bin_nr_N, isRGB, quality, downsample, comment);

            HackyPrivate::_jpegFile.close();

            return true;
        }

        bool get_img(std::string name, int max_bin_nr_N, int max_bin_nr_M, SDDMM::Types::COO& mat, float nz_probability) {

            float remainder_n = mat.n % max_bin_nr_N;
            float remainder_m = mat.m % max_bin_nr_M;
            float bs_end_n = (mat.n - remainder_n) / max_bin_nr_N;
            float bs_begin_n = bs_end_n + 1;
            float bs_end_m = (mat.m - remainder_m) / max_bin_nr_M;
            float bs_begin_m = bs_end_m + 1;

            TEXT::Gadgets::print_colored_text_line(
                std::string("Remainder n: ") + std::to_string(remainder_n), TEXT::BLUE);
            TEXT::Gadgets::print_colored_text_line(
                std::string("Large bin n: ") + std::to_string(bs_begin_n), TEXT::BLUE);
            TEXT::Gadgets::print_colored_text_line(
                std::string("Small bin n: ") + std::to_string(bs_end_n), TEXT::BLUE);
            TEXT::Gadgets::print_colored_text_line(
                std::string("Remainder m: ") + std::to_string(remainder_m), TEXT::BLUE);
            TEXT::Gadgets::print_colored_text_line(
                std::string("Large bin m: ") + std::to_string(bs_begin_m), TEXT::BLUE);
            TEXT::Gadgets::print_colored_text_line(
                std::string("Small bin m: ") + std::to_string(bs_end_m), TEXT::BLUE);

            if (bs_begin_n * remainder_n + (max_bin_nr_N - remainder_n) * bs_end_n != mat.n) {
                TEXT::Gadgets::print_colored_text_line("Height out of bounds or too small!", TEXT::HIGHLIGHT_RED);
                return false;
            }
            if (bs_begin_m * remainder_m + (max_bin_nr_M - remainder_m) * bs_end_m != mat.m) {
                TEXT::Gadgets::print_colored_text_line("Width out of bounds or too small!", TEXT::HIGHLIGHT_RED);
                return false;
            }

            _histogram_buffer.clear();
            _histogram_buffer.shrink_to_fit();
            int img_size = max_bin_nr_N * max_bin_nr_M;
            _histogram_buffer.resize(img_size, 0);
            int64_t S = mat.cols.size();

            for (int64_t s = 0; s < S; ++s) {
                double col = static_cast<double>(mat.cols[s]);
                double row = static_cast<double>(mat.rows[s]);

                int bin_n = 0;
                if (row - remainder_n * bs_begin_n >= 0) {
                    bin_n = static_cast<int>(std::floor((row - remainder_n * bs_begin_n) / bs_end_n) + remainder_n);
                }
                else {
                    bin_n = static_cast<int>(std::floor(row / bs_begin_n));
                }

                int bin_m = 0;
                if (col - remainder_m * bs_begin_m >= 0) {
                    bin_m = static_cast<int>(std::floor((col - remainder_m * bs_begin_m) / bs_end_m) + remainder_m);
                }
                else {
                    bin_m = static_cast<int>(std::floor(col / bs_begin_m));
                }

                int64_t ind = bin_n * max_bin_nr_M + bin_m;
                if (ind >= _histogram_buffer.size()) {
                    TEXT::Gadgets::print_colored_text_line("Histogram index out of bounds!", TEXT::HIGHLIGHT_RED);
                    return false;
                }
                
                float cur = _histogram_buffer[ind];

                cur += 1;

                _histogram_buffer[ind] = cur;
            }

            float normalize_middle = bs_begin_n * bs_begin_m;
            float normalize_right = bs_begin_n * bs_end_m;
            float normalize_bottom = bs_end_n * bs_begin_m;
            float normalize_bottom_right = bs_end_n * bs_end_m;

            for (int64_t h = 0; h < max_bin_nr_N; ++h) {
                for (int64_t w = 0; w < max_bin_nr_M; ++w) {
                    int64_t ind = h * max_bin_nr_M + w;
                    if (ind > _histogram_buffer.size() - 1) {
                        std::cout << "lol" << std::endl;
                    }
                    float cur = _histogram_buffer[ind];

                    if (h >= remainder_n && w >= remainder_m) cur /= normalize_bottom_right;
                    else if (h >= remainder_n) cur /= normalize_bottom;
                    else if (w >= remainder_m) cur /= normalize_right;
                    else cur /= normalize_middle;

                    if (cur < 0) {
                        TEXT::Gadgets::print_colored_text_line("Warning: Image must consist of positive values!", TEXT::HIGHLIGHT_YELLOW);
                        cur = 0;
                    }
                    else if (cur > 1.0f) {
                        TEXT::Gadgets::print_colored_text_line("Warning: Image must consist of values smaller or equal to 1!", TEXT::HIGHLIGHT_YELLOW);
                        cur = 1.0f;
                    }

                    _histogram_buffer[ind] = cur;
                }

            }

            return get_img(name, max_bin_nr_N, max_bin_nr_M, _histogram_buffer, nz_probability);
        }
	};
}