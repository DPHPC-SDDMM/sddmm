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
         * This is the data type used for all experiments!
        */
        typedef float expmt_t;

        /**
         * These are all other data types that like to have aggregated names
        */
        typedef std::vector<expmt_t>::size_type vec_size_t;
    }
    
    /**
     * All defines like structs, constants etc.
    */
    class Defines {
    public:
        static constexpr SDDMM::Types::expmt_t epsilon = 1e-6;
        static constexpr int warp_size = 32;

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
}