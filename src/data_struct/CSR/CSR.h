#include <random>
#include <stdexcept>
// Data structures
#include <tuple>
#include <vector>
#include <map>

#include <algorithm> // std::sort

namespace SDDMM{

    /* NOTE:
    The data primitive (arbitrarily) decided to store the matrix content
    are `double`.
    This was chosen, because they offer large accuracy,
    without encumbering the system with too much memory per element.
    Although I do not have citations to support this claim,
    I have heard that the use of double in numerical computations
    is also very common.
    */

    class CSR{

        public:

            /**
             * Generate a random matrix represented in the CSR format.
             * For details about the COO format, you may read
             * https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
             * 
             * @param n: Number of rows of the generated matrix.
             * @param m: Number of columns of the generated matrix.
             * @param sparsity: A percentage expressing the ratio of non-zero elements to all the elements (n * m).
             * @param distribution: Distribution according to which matrix elements are generated.
            */
            std::tuple<std::vector<int>, std::vector<int>, std::vector<double>> static Generate(
                int n, int m, float sparsity,
                bool sort=true, std::string distribution = "uniform"
                )
            {

                // TODO: Implement numerous random distribution schemes.
                // Currently, only the default (uniform) is implemented. 

                // Dynamic sanity check.
                if (sparsity < 0 | sparsity > 1.0)
                    throw std::out_of_range("Expected a sparsity value from zero (0) to one (1).");

                
                // Compute number of non-zero elements of the matrix.
                int nr_elements = (int) n * m * sparsity;

                // Define the three output 1D arrays.
                //? Does `reserve` and afterwards `push_back` create less overhead
                //? than `row_index(n+1)` and then `row_index[i] = row`? 
                std::vector<int> row_ptrs(n+1), column_index; std::vector<double> values;
                //row_ptrs.reserve(n+1);
                column_index.reserve(nr_elements); values.reserve(nr_elements);

                // Define random generator
                // and distribution functions for the random generation.
                // NOTE: Source for random number generator:
                // https://stackoverflow.com/questions/15461140/stddefault-random-engine-generate-values-between-0-0-and-1-0
                std::random_device rd;
                std::default_random_engine generator(rd());
                std::uniform_int_distribution<int> row_distribution(0, nr_elements);
                std::uniform_int_distribution<int> column_distribution(0, m-1);
                // [-1, 1] values were selected because neural networks often deal
                // with smalle values.
                std::uniform_real_distribution<double> value_distribution(-1.0, 1.0);

                // Construct the `row_ptrs` data structure.
                // row_ptrs.push_back(0); // The first element of the CSR row pointer array is 0.
                for (int i = 1 ; i <= n ; ++i)
                {
                    // Ensure that all rows have the same number elements.
                    // This is how uniformity is ensured globally.
                    row_ptrs[i] = row_ptrs[i-1] + nr_elements / n;
                    //! Possible bug, because of rounding during the definition of `nr_elements`?
                }

                // Define the data structure (hash map) which will ensure
                // that no (row, column) duplicates are inserted.
                // NOTE: In practice, that probability will be fairly low.
                std::map<int, int> row_to_column;

                // Construct the `column_index` and `values` data structures.
                for (int row = 1 ; row <= n ; ++row)
                {
                    size_t elements_remaining = row_ptrs[row] - row_ptrs[row-1];

                    while (elements_remaining)
                    {
                        int column = column_distribution(generator);
                        bool successful_insertion = std::get<1>(row_to_column.emplace(row, column));
// Genera
                        if (successful_insertion)
                        {
                            double value = value_distribution(generator); // Generate cell value.
                            
                            column_index.push_back(column);
                            values.push_back(value);
                            --elements_remaining;
                        }
                    }
                    

                }

                return std::tuple<std::vector<int>, std::vector<int>, std::vector<double>>(row_ptrs, column_index, values);
                
            }

    };
}