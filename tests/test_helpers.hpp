#pragma once

#include <vector>
#include <functional>
#include "../src/defines.h"
#include "../utest/utest.h"

namespace TestHelpers {

    template<typename T>
    using FEquals = std::function<bool(T, T)>;

    template<typename T>
    bool compare_vectors(const std::vector<T>& v1, const std::vector<T>& v2, const FEquals<T>& equals = [](T x, T y) {
        return std::fabs(x - y) <= SDDMM::Defines::epsilon;
    }) {
        if(v1.size() != v2.size()){
            return false;
        }

        std::size_t s = v1.size();
        for(std::size_t i=0; i<s; ++i){
            if(!equals(v1.at(i), v2.at(i))){
                return false;
            }
        }
        return true;
    }
}