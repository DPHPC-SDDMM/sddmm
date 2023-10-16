#include <iostream>
#include "algos/algo.h"
#include "data_struct/COO/COO.h"
#include "data_struct/CSR/CSR.h"

/**
 * All algos as cpp files
*/
#include "algos/algo_sample.cpp"

/**
 * This is the algo that will run
*/
#define SAMPLE_ALGO

int main(int argc, char** argv){
    /**
     * Perform some global initialization code here
     * Add results into InitParams and pass to subsequent algos
    */
    const SDDMM::InitParams initParams {
        .sampleParam=5
    };

    /**
     * TODO: Probably we may want to add if/else if/else based on some string here
     *  => could be more convenient
     * Alternative would be to set "#define SAMPLE_ALGO" inside CMakeLists.txt
    */
#ifdef SAMPLE_ALGO
    SDDMM::AlgoSample().main(argc, argv, initParams);
#endif

    auto coo_matrix = SDDMM::COO::Generate(10, 10, 0.1);
    for (int i = 0 ; i < coo_matrix.size() ; ++i)
    {
        auto elmnt = coo_matrix[i];
        std::cout << std::get<0>(elmnt) << " " << std::get<1>(elmnt) << " " << std::get<2>(elmnt) << std::endl;
    }

    auto csr_matrix = SDDMM::CSR::Generate(10, 10, 0.1);
    auto row_pointers = std::get<0>(csr_matrix);
    auto clmn_indx = std::get<1>(csr_matrix);
    auto values = std::get<2>(csr_matrix);

    std::cout << "Printing 'row pointers': " << std::endl;
    for (int row = 0 ; row < row_pointers.size() ; ++row)
        std::cout << row_pointers[row] << " ";
    std::cout << std::endl;

    std::cout << "Printing 'column indices': " << std::endl;
    for (int elmnts = 0 ; elmnts < (int) 10*10*0.1 ; ++elmnts)
        std::cout << clmn_indx[elmnts] << " ";
    std::cout << std::endl;
    
    std::cout << "Printing 'values': " << std::endl;
    for (int elmnts = 0 ; elmnts < (int) 10*10*0.1 ; ++elmnts)
        std::cout << values[elmnts] << " ";
    std::cout << std::endl;

}
