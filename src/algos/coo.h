#pragma once

#include <vector>
namespace SDDMM {

// COO represents a matrix in the COOrdinate format
class COO {
  public:
    std::vector<int> row_indexes;
    std::vector<int> column_indexes;
    std::vector<float> values;
};
} // namespace SDDMM