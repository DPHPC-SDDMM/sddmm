#pragma once
/**
 * This one has to be a *.hpp file to work with matplot
 * If I ever find out why, I will celebrate with 1000 alcoholic beverages
*/

#include "../defines.h"
#include <matplot/matplot.h>

namespace SDDMM {
    class Plots {
    public:

        static void test_plot(){
            std::vector<double> x = matplot::linspace(0, 2 * matplot::pi);
            std::vector<double> y = matplot::transform(x, [](auto x) { return sin(x); });

            matplot::plot(x, y, "-*");
            matplot::hold(matplot::on);
            matplot::plot(x, matplot::transform(y, [](auto y) { return -y; }), "--xr");
            matplot::plot(x, matplot::transform(x, [](auto x) { return x / matplot::pi - 1.; }), "-:gs");
            matplot::plot({1.0, 0.7, 0.4, 0.0, -0.4, -0.7, -1}, "k");

            matplot::show();
        }

        static void err_plot(const SDDMM::Defines::ErrPlotData& data){
            std::vector<double> err;
            std::vector<double> y;
            for(auto r : data.runtimes){
                double min_v = std::numeric_limits<double>::max();
                double max_v = std::numeric_limits<double>::min();
                double sum = 0;
                for(auto val : r){
                    sum += val;
                    if(min_v > val) min_v = val;
                    if(max_v < val) max_v = val;
                }
                err.push_back(max_v - min_v);
                y.push_back(sum/r.size());
            }

            matplot::errorbar(data.x, y, err);
            matplot::xrange({static_cast<double>(data.min), static_cast<double>(data.max)});

            // for(int rn=0; rn<data.runtimes.size(); ++rn){
            //     matplot::nexttile();
            //     matplot::hist(data.runtimes.at(rn), matplot::histogram::binning_algorithm::automatic);
            // }

            matplot::show(); 
        }
    };
}