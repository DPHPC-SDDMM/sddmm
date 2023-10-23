#pragma once

/**
 * Add whatever generator classes here
*/

#include <vector>
#include <string>

namespace SDDMM {
    typedef std::vector<double>::size_type vec_size_t;
    
    class Defines {
    public:
        struct InitParams {
            int sampleParam;
        };

        struct ErrPlotData {
            double min;
            double max;
            std::vector<double> x;
            std::vector<std::vector<double>> runtimes;
        };

        struct CSR {
            std::vector<double> values;
            std::vector<int> col_idx;
            std::vector<int> row_ptr;
        };

        static void vector_fill(std::vector<double>& vector, double start, double step, double end){
            vector.clear();
            while(start < end){
                vector.push_back(start);
                start += step;
            }
        }

        static std::string get_title_str(std::string name){
            std::string lines = " =============================== ";
            return lines + name + lines;
        }
    };
}