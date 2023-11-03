#pragma once

/**
 * Add whatever generator classes here
*/

#include <vector>
#include <string>

namespace SDDMM {

    /**
     * All type declarations
    */
    namespace Types {
        /**
         * This is the data type used for all experiments!!
        */
        typedef float expmt_t;

        /**
         * These are all other data types that like to have aggregated names
        */
        typedef std::vector<expmt_t>::size_type vec_size_t;
        typedef std::vector<std::vector<expmt_t>> matrix_t;
    }
    
    /**
     * All defines like structs, constants etc.
    */
    class Defines {
    public:
        struct InitParams {
            int sampleParam;
        };

        struct ErrPlotData {
            Types::expmt_t min;
            Types::expmt_t max;
            std::vector<Types::expmt_t> x;
            std::vector<std::vector<Types::expmt_t>> runtimes;
        };

        struct RC {
            SDDMM::Types::vec_size_t row;
            SDDMM::Types::vec_size_t col;
            SDDMM::Types::vec_size_t val_offset;
        };

        static void vector_fill(std::vector<Types::expmt_t>& vector, Types::expmt_t start, Types::expmt_t step, Types::expmt_t end){
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