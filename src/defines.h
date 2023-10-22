#pragma once

/**
 * Add whatever generator classes here
*/

#include <vector>
#include <string>

namespace SDDMM {
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