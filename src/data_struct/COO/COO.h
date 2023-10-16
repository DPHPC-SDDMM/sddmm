// Libraries for randomness
#include <random>
// Libraries for exception handling
#include <stdexcept>
// Data structures
#include <tuple>
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

    class COO{

        public:

            /**
             * The `COOelement` is a triplet of (int, int, double), whose elements represent:
             * 0: row index
             * 1: column index
             * 2: value of cell
             * respectively.
            */
            typedef std::tuple<int, int, double> COOelement;
            // NOTE: No need to define custom comparator function,
            // as the desired sorting method is the default behaviour of `tuple`.

            /**
             * Generate a random matrix represented in the COO format.
             * For details about the COO format, you may read
             * https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)
             * 
             * @param n: Number of rows of the generated matrix.
             * @param m: Number of columns of the generated matrix.
             * @param sparsity: A percentage expressing the ratio of non-zero elements to all the elements (n * m).
             * @param sort: Determines whether the output is returned in a sorted manner or not.
             *              The sorting takes place first by ascending row and then by ascending column.
             * @param distribution: Distribution according to which matrix elements are generated.
            */
            std::vector<COOelement> static Generate(
                int n, int m, float sparsity,
                bool sort=true, std::string distribution = "uniform"
                )
            {

                // TODO: Implement numerous random distribution schemes.
                // Currently, only the default (uniform) is implemented. 

                // Dynamic sanity check.
                if (sparsity < 0 | sparsity > 1.0)
                    throw std::out_of_range("Expected a sparsity value from zero (0) to one (1).");

                

                // Define the `output` data structure
                // and allocate sufficient memory for it in advance. 
                std::vector<COOelement> output;
                int nr_elements = (int) n * m * sparsity;
                output.reserve(nr_elements);

                // Define random generator
                // and distribution functions for the random generation. 
                // NOTE: Source for random number generator:
                // https://stackoverflow.com/questions/15461140/stddefault-random-engine-generate-values-between-0-0-and-1-0
                std::random_device rd;
                std::default_random_engine generator(rd());
                std::uniform_int_distribution<int> row_distribution(0, n-1);
                std::uniform_int_distribution<int> column_distribution(0, m-1);
                // [-1, 1] values were selected because neural networks often deal
                // with smalle values.
                std::uniform_real_distribution<double> value_distribution(-1.0, 1.0);
                

                // Define the data structure (hash map) which will ensure
                // that no (row, column) duplicates are inserted.
                // NOTE: In practice, that probability will be fairly low.
                std::map<int, int> row_to_column;
                size_t elements_remaining = nr_elements;

                while (elements_remaining) // 
                {
                    int row = row_distribution(generator);
                    int column = column_distribution(generator);
                    bool successful_insertion = std::get<1>(row_to_column.emplace(row, column));

                    if (successful_insertion)
                    {
                        double value = value_distribution(generator); // Generate cell value.

                        output.push_back({ row, column, value }); // Add element to output.
                        --elements_remaining; // Decrease counter of further elements required to add.
                    }
                }

                if (sort) { std::sort(output.begin(), output.end()); }

                return output;
                
            }

    };
}