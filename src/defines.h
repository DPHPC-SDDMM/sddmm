#pragma once

/**
 * Add whatever generator classes here
*/

#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <cusparse.h>         // cusparseSpMM

namespace SDDMM {

    // "proper cuda error checking"
    // https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
    {
        if (code != cudaSuccess)
        {
            fprintf(stdout, "==================================================================\n");
            fprintf(stdout,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            fprintf(stdout, "==================================================================\n");
            if (abort) exit(code);
        }
    }

    #define sparse_gpuErrchk(func)                                                   \
    {                                                                              \
        cusparseStatus_t status = (func);                                          \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
                __LINE__, cusparseGetErrorString(status), status);              \
            exit(status);                                                          \
        }                                                                          \
    }

    /**
     * All type declarations
    */
    namespace Types {
        /**
         * This is the data type used for all experiments!
        */
        typedef float expmt_t;
        constexpr auto cuda_expmt_t = cudaDataType_t::CUDA_R_32F;

        /**
         * These are all other data types that like to have aggregated names
        */
        // cost for using 8 bytes vec_size_t: about +20%
        // typedef std::vector<expmt_t>::size_type vec_size_t;
        typedef uint32_t vec_size_t;
        constexpr auto cuda_vec_size_t = cusparseIndexType_t::CUSPARSE_INDEX_32I;

        typedef std::chrono::microseconds time_measure_unit;
        typedef int64_t time_duration_unit;
    }

    namespace Constants {
        constexpr int col_storage = 1;
        constexpr int row_storage = 2;
    }

    /**
     * All defines like structs, constants etc.
    */
    class Defines {
    public:
    #ifdef USE_LOW_PRECISION
        static constexpr SDDMM::Types::expmt_t epsilon = 1e-3;
    #elif USE_GPU_PRECISION
        static constexpr SDDMM::Types::expmt_t epsilon = 1e-5;
    #else
        static constexpr SDDMM::Types::expmt_t epsilon = 1e-6;
    #endif
        static constexpr int warp_size = 32;

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

        // some generic way to distinguish operating systems
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    static const char path_separator = '\\';
   //define something for Windows (32-bit and 64-bit, this part is common)
   #ifdef _WIN64
      //define something for Windows (64-bit only)
	  #define PLATFORM_WINDOWS_x64
   #else
      //define something for Windows (32-bit only)
	  #define PLATFORM_WINDOWS_x32
   #endif
#elif __APPLE__
    static const char path_separator = '/';
    #include <TargetConditionals.h>
    #if TARGET_IPHONE_SIMULATOR
         // iOS, tvOS, or watchOS Simulator
    #elif TARGET_OS_MACCATALYST
         // Mac's Catalyst (ports iOS API into Mac, like UIKit).
    #elif TARGET_OS_IPHONE
        // iOS, tvOS, or watchOS device
    #elif TARGET_OS_MAC
        // Other kinds of Apple platforms
		#define PLATFORM_MAC
    #else
    #   error "Unknown Apple platform"
    #endif
#elif __ANDROID__
    // Below __linux__ check should be enough to handle Android,
    // but something may be unique to Android.
#elif __linux__
    // linux
	#define PLATFORM_LINUX
    static const char path_separator = '/';
#elif __unix__ // all unices not caught above
    // Unix
#elif defined(_POSIX_VERSION)
    // POSIX
#else
#   error "Unknown compiler"
#endif
    };

    namespace TEXT {

        typedef std::string color_t;
        /**
        * Name            FG  BG
        * Black           30  40
        * Red             31  41
        * Green           32  42
        * Yellow          33  43
        * Blue            34  44
        * Magenta         35  45
        * Cyan            36  46
        * White           37  47
        * Bright Black    90  100
        * Bright Red      91  101
        * Bright Green    92  102
        * Bright Yellow   93  103
        * Bright Blue     94  104
        * Bright Magenta  95  105
        * Bright Cyan     96  106
        * Bright White    97  107
        */

        // basic text colors
        static const color_t BLACK =   "\x1B[30m";
        static const color_t RED =     "\x1B[31m";
        static const color_t GREEN =   "\x1B[32m";
        static const color_t YELLOW =  "\x1B[33m";
        static const color_t BLUE =    "\x1B[34m";
        static const color_t MAGENTA = "\x1B[35m";
        static const color_t CYAN =    "\x1B[36m";
        static const color_t WHITE =   "\x1B[37m";

        static const color_t BRIGHT_BLACK =   "\x1B[90m";
        static const color_t BRIGHT_RED =     "\x1B[91m";
        static const color_t BRIGHT_GREEN =   "\x1B[92m";
        static const color_t BRIGHT_YELLOW =  "\x1B[93m";
        static const color_t BRIGHT_BLUE =    "\x1B[94m";
        static const color_t BRIGHT_MAGENTA = "\x1B[95m";
        static const color_t BRIGHT_CYAN =    "\x1B[96m";
        static const color_t BRIGHT_WHITE =   "\x1B[97m";

        // format terminator
        static const color_t END = "\x1B[0m";

        // error headers
        static const color_t TRACE_TITLE =   "\x1B[3;44;97m";
        static const color_t LOG_TITLE =     "\x1B[3;42;97m";
        static const color_t MESSAGE_TITLE = "\x1B[3;107;30m";
        static const color_t WARN_TITLE =    "\x1B[3;43;97m";
        static const color_t ERROR_TITLE =   "\x1B[3;41;97m";
        static const color_t FATAL_TITLE =   "\x1B[3;45;97m";

        // text highlighters
        static const color_t HIGHLIGHT_YELLOW = "\x1B[3;43;30m";
        static const color_t HIGHLIGHT_GREEN = "\x1B[3;102;30m";
        static const color_t HIGHLIGHT_CYAN = "\x1B[3;46;30m";
        static const color_t HIGHLIGHT_RED = "\x1B[3;41;97m";

        class Cast {
            public:
            static std::string HighlightYellow(std::string message) { return HIGHLIGHT_YELLOW + message + END; }
            static std::string HighlightGreen(std::string message) { return HIGHLIGHT_GREEN + message + END; }
            static std::string HighlightCyan(std::string message) { return HIGHLIGHT_CYAN + message + END; }
            static std::string HighlightRed(std::string message) { return HIGHLIGHT_RED + message + END; }
            static std::string TraceTitle(std::string message) { return TRACE_TITLE + message + END; }
            static std::string LogTitle(std::string message) { return LOG_TITLE + message + END; }
            static std::string MessageTitle(std::string message) { return MESSAGE_TITLE + message + END; }
            static std::string WarnTitle(std::string message) { return WARN_TITLE + message + END; }
            static std::string ErrorTitle(std::string message) { return ERROR_TITLE + message + END; }
            static std::string FatalTitle(std::string message) { return FATAL_TITLE + message + END; }
            static std::string Black(std::string message) { return BLACK + message + END; }
            static std::string Red(std::string message) { return RED + message + END; }
            static std::string Green(std::string message) { return GREEN + message + END; }
            static std::string Yellow(std::string message) { return YELLOW + message + END; }
            static std::string Blue(std::string message) { return BLUE + message + END; }
            static std::string Magenta(std::string message) { return MAGENTA + message + END; }
            static std::string Cyan(std::string message) { return CYAN + message + END; }
            static std::string White(std::string message) { return WHITE + message + END; }
        };

        class Gadgets {
            public:
            /**
             * 1 <= current <= total
            */
            static void print_progress(int current, int total){
                std::cout << "\r" << GREEN << "[" << current << " / " << total << "]     " << END;
                if(current < total) std::cout << std::flush;
                else std::cout << std::endl;
            }

            static void print_colored_line(int length, char c, std::string color){
                std::cout << color;
                for(int i=0; i<length; ++i){
                    std::cout << c;
                }
                std::cout << END << std::endl;
            }

            static std::string get_cur(int cur_exp, int tot_exp){
                return std::string("..(") 
                     + std::to_string(cur_exp) 
                     + std::string("/") 
                     + std::to_string(tot_exp) 
                     + std::string(")..");
            }

            static std::string num_2_str(
                Types::vec_size_t r_num, 
                Types::vec_size_t k_num, 
                Types::vec_size_t c_num
            ){
                return "[" + std::to_string(r_num) + "," + std::to_string(k_num) + "," + std::to_string(c_num) + "]";
            }
        };
    }
}