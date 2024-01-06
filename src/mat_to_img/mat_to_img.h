
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

        //void init(unsigned char* img, int img_width, int img_height) {
        //    int64_t s = img_width * img_height * 3;
        //    for (int64_t ind = 0; ind < s; ++ind) {
        //        img[ind] = static_cast<char>(0);
        //    }
        //}

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
            return get_img(name, mat.n, mat.m, mat);
        }

        bool get_img(std::string name, uint64_t bin_nr_N, uint64_t bin_nr_M, SDDMM::Types::Matrix& mat) {
            float max = 0;
            for (uint64_t n = 0; n < bin_nr_N; ++n) {
                for (uint64_t m = 0; m < bin_nr_M; ++m) {
                    if (max < mat.at(n, m)) {
                        max = mat.at(n, m);
                    }
                }
            }

            return get_img(name, bin_nr_N, bin_nr_M, max, mat.data);
        }

        bool get_img(std::string name, uint64_t bin_nr_N, uint64_t bin_nr_M, float max, std::vector<float> data){
            unsigned char max_r = 255;
            unsigned char max_g = 0;
            unsigned char max_b = 0;

            unsigned char min_r = 0;
            unsigned char min_g = 0;
            unsigned char min_b = 255;

            float min_oklab_L, min_oklab_a, min_oklab_b, max_oklab_L, max_oklab_a, max_oklab_b;
            linear_srgb_to_oklab(max_r, max_g, max_b, max_oklab_L, max_oklab_a, max_oklab_b);
            linear_srgb_to_oklab(min_r, min_g, min_b, min_oklab_L, min_oklab_a, min_oklab_b);

            _img_buffer.clear();
            _img_buffer.shrink_to_fit();
            _img_buffer.reserve(bin_nr_N * bin_nr_M * 3);
            for (const float& val : data) {
                float p = val / max;
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
            const int quality = 90;
            const bool downsample = false;
            const char* comment = name.c_str();
            bool ok = TooJpeg::writeJpeg(_output, _img_buffer.data(), bin_nr_M, bin_nr_N, isRGB, quality, downsample, comment);

            HackyPrivate::_jpegFile.close();

            return true;
        }

        bool get_img(std::string name, int bin_nr_N, int bin_nr_M, SDDMM::Types::COO& mat) {
            double bin_size_w = std::ceil(static_cast<double>(mat.m) / static_cast<double>(bin_nr_M));
            double bin_size_h = std::ceil(static_cast<double>(mat.n) / static_cast<double>(bin_nr_N));

            _histogram_buffer.clear();
            _histogram_buffer.shrink_to_fit();
            _histogram_buffer.resize(bin_nr_N * bin_nr_M, 0);
            uint64_t S = mat.cols.size();
            float max = 0;
            for (uint64_t s = 0; s < S; ++s) {
                double col = static_cast<double>(mat.cols[s]);
                double row = static_cast<double>(mat.rows[s]);
                int bin_n = static_cast<int>(std::floor(row / bin_size_h));
                int bin_m = static_cast<int>(std::floor(col / bin_size_w));

                uint64_t ind = bin_n * bin_nr_M + bin_m;
                if (ind >= _histogram_buffer.size()) {
                    TEXT::Gadgets::print_colored_text_line("Histogram index out of bounds!", TEXT::HIGHLIGHT_RED);
                    return false;
                }
                float cur = _histogram_buffer[ind];
                cur++;
                if (cur > max) {
                    max = cur;
                }
                _histogram_buffer[ind] = cur;
            }

            return get_img(name, bin_nr_N, bin_nr_M, max, _histogram_buffer);
        }
	};
}