#include <iostream>                                                  

/**
 * All algos as cpp files
*/

#include "defines.h"
#include "algos/sample_algo.cpp"
#include <matplot/matplot.h>
#include "libs/lib_plot.hpp"

/**
 * This is the algo that will run
*/

int main(int argc, char** argv){
#ifdef NONE
    std::cout << SDDMM::Defines::get_title_str("NONE") << std::endl;
    std::vector<double> x = matplot::linspace(0, 2 * matplot::pi);
    std::vector<double> y = matplot::transform(x, [](auto x) { return sin(x); });

    matplot::plot(x, y, "-*");
    matplot::hold(matplot::on);
    matplot::plot(x, matplot::transform(y, [](auto y) { return -y; }), "--xr");
    matplot::plot(x, matplot::transform(x, [](auto x) { return x / matplot::pi - 1.; }), "-:gs");
    matplot::plot({1.0, 0.7, 0.4, 0.0, -0.4, -0.7, -1}, "k");

    matplot::show();
#elif SAMPLE_ALGO
    std::cout << SDDMM::Defines::get_title_str("SAMPLE_ALGO") << std::endl;
    SDDMM::Defines::ErrPlotData data;
    SDDMM::SampleAlgo(data, 250, 100, 500, 100);
    SDDMM::Plots::err_plot(data);
#endif

    return 0;
}
