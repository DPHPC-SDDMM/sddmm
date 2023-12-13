#include <filesystem>
#include <thread>
#include <chrono>

#include "../data_structures/coo/coo.h"
#include "../data_structures/csr/csr.h"
#include "../algos/cuda_sddmm/cuda_sddmm.cpp"
#include "../algos/sm-l2-sddmm/sm-l2-sddmm.cpp"
#include "../algos/cusparse_sddmm/cusparse_1.cpp"
#include "../results.h"

namespace SDDMM {
    namespace Experiments {
        class GPU_SDDMMBenchmarks {
            public:
            enum class TestSubject {
                Non_Tiled_Baseline,
                cuSPARSE,
                Sm_L2
            };

            static Results::ExperimentData run(
                TestSubject subject,
                int cur_exp, 
                int tot_exp,
                int n_experiment_iterations,
                Types::COO& coo_mat, 
                Types::CSR& csr_mat,
                float sparsity,
                Types::Matrix& X, 
                Types::Matrix& Y, 
                Types::expmt_t& total
            ){
                Results::ExperimentData data;
                switch(subject){
                    case TestSubject::Non_Tiled_Baseline:
                    data.label = "non_tiled Baseline";
                    break;
                    case TestSubject::cuSPARSE:
                    data.label = "cuSPARSE";
                    break;
                    case TestSubject::Sm_L2:
                    data.label = "sm_l2";
                    break;
                }
                
                TEXT::Gadgets::print_colored_text_line(std::string("Experiment ") + data.label, TEXT::BRIGHT_RED);
                std::cout << TEXT::Cast::Green("...preparations...") << std::endl;
                std::cout << TEXT::Cast::Cyan(TEXT::Gadgets::get_cur(cur_exp, tot_exp)) << data.label << std::endl;
                TEXT::Gadgets::print_progress(0, n_experiment_iterations);

                if(subject == TestSubject::Sm_L2){
                    Types::vec_size_t n_max = n_experiment_iterations+1;

                    auto params = SDDMM::Algo::SML2SDDMM::preparations(
                        coo_mat, sparsity,
                        // N, M, K
                        X.n, Y.m, Y.n,
                        X, Y);

                    for(Types::vec_size_t n=0; n<n_max; ++n){
                        TEXT::Gadgets::print_progress(n, n_experiment_iterations);
                        total += SDDMM::Algo::SML2SDDMM::run_sm_l2(
                            coo_mat, sparsity, 
                            X, Y,
                            // N, M, K
                            X.n, Y.m, Y.n, 
                            params, &data).values[0];
                    }
                }
                else{
                    Types::vec_size_t n_max = n_experiment_iterations+1;
                    for(Types::vec_size_t n=0; n<n_max; ++n){
                        TEXT::Gadgets::print_progress(n, n_experiment_iterations);
                        
                        switch(subject){
                            case TestSubject::Non_Tiled_Baseline:
                            total += SDDMM::Algo::cuda_sddmm(coo_mat, X, Y, &data).values[0];
                            break;
                            case TestSubject::cuSPARSE:
                            total += SDDMM::Algo::cuSPARSE_SDDMM(csr_mat, X, Y, &data).values[0];
                            break;
                        }
                    }
                }

                return data;
            }

            static bool compareNat(const std::string& a, const std::string& b)
            {
                // source:
                // https://copyprogramming.com/howto/how-to-implement-a-natural-sort-algorithm-in-c
                if (a.empty())
                    return true;
                if (b.empty())
                    return false;
                if (std::isdigit(a[0]) && !std::isdigit(b[0]))
                    return true;
                if (!std::isdigit(a[0]) && std::isdigit(b[0]))
                    return false;
                if (!std::isdigit(a[0]) && !std::isdigit(b[0]))
                {
                    if (std::toupper(a[0]) == std::toupper(b[0]))
                        return compareNat(a.substr(1), b.substr(1));
                    return (std::toupper(a[0]) < std::toupper(b[0]));
                }
                // Both strings begin with digit --> parse both numbers
                std::istringstream issa(a);
                std::istringstream issb(b);
                int ia, ib;
                issa >> ia;
                issb >> ib;
                if (ia != ib)
                    return ia < ib;
                // Numbers are the same --> remove numbers and recurse
                std::string anew, bnew;
                std::getline(issa, anew);
                std::getline(issb, bnew);
                return (compareNat(anew, bnew));
            }

            static void benchmark_static(std::string experiment_name, std::string experiment_variable, int n_experiment_iterations, std::string folder_location) {

                std::vector<std::string> mat_files;
                for (const auto& entry : std::filesystem::directory_iterator(folder_location)) {
                    std::string p = entry.path().string();
                    std::string end = p.substr(p.size() - 7, 7);
                    if (end.compare(".bindat") == 0) {
                        mat_files.push_back(entry.path().string());
                    }
                }

                std::sort(mat_files.begin(), mat_files.end(), compareNat);

                TEXT::Gadgets::print_colored_line(100, '=', TEXT::BRIGHT_RED);

                std::cout << TEXT::Cast::Cyan("Start measurements") << std::endl;
                // ===================================================================

                std::vector<GPU_SDDMMBenchmarks::TestSubject> subject = {
                    GPU_SDDMMBenchmarks::TestSubject::Non_Tiled_Baseline,
                    GPU_SDDMMBenchmarks::TestSubject::cuSPARSE,
                    GPU_SDDMMBenchmarks::TestSubject::Sm_L2
                };

                // run all tests
                std::vector<Types::expmt_t> total(subject.size(), 0.0);
                std::vector<Results::ExperimentData> results;

                for (auto& name : mat_files) {
                    std::vector<Types::expmt_t> total(subject.size(), 0.0);
                    std::vector<Results::ExperimentData> results;

                    TEXT::Gadgets::print_colored_line(100, '*', TEXT::HIGHLIGHT_YELLOW);
                    std::cout << TEXT::Cast::Cyan("...loading data...") << std::endl;
                    SDDMM::Types::COO coo_mat;
                    SDDMM::Types::CSR csr_mat;
                    SDDMM::Types::Matrix X(0, 0);
                    SDDMM::Types::Matrix Y(0, 0);
                    float sparse_sparsity;
                    float X_sparsity;
                    float Y_sparsity;
                    uint64_t out_size_read;
                    SDDMM::Types::COO::hadamard_from_bin_file(
                        name,
                        coo_mat, csr_mat, sparse_sparsity,
                        X, X_sparsity,
                        Y, Y_sparsity,
                        out_size_read);

                    Types::vec_size_t N = X.n;
                    Types::vec_size_t M = Y.m;
                    Types::vec_size_t K = X.m;
                    
                    std::cout << TEXT::Cast::Cyan(
                        std::string("...stats:\n ") +
                        std::string("......N:        ") + std::to_string(N) + std::string("\n") +
                        std::string("......M:        ") + std::to_string(M) + std::string("\n") +
                        std::string("......K:        ") + std::to_string(K) + std::string("\n") +
                        std::string("......sparsity: ") + std::to_string(sparse_sparsity)) << std::endl;

                    std::cout << TEXT::Cast::Cyan("...run experiment iterations...") << std::endl;
                    for (int i = 0; i < subject.size(); ++i) {
                        results.push_back(run(
                            subject[i], i + 1, subject.size(), 
                            n_experiment_iterations, 
                            coo_mat, csr_mat, sparse_sparsity, 
                            X, Y,
                            total[i])
                        );
                    }

                    std::stringstream info;
                    info << "[INFO]\n"
                        << "experiment_name " << experiment_name << "\n"
                        << "variable " << experiment_variable << "\n"
                        << "N " << N << "\n"
                        << "M " << M << "\n"
                        << "K " << K << "\n"
                        << "sparsity " << sparse_sparsity << "\n";
                    info << "[/INFO]";

                    std::stringstream str;
                    str << "iters-" << n_experiment_iterations << "_var-" << experiment_variable;

                    // ===================================================================
                    std::cout << TEXT::Cast::Cyan("Saving experiment data") << std::endl;
                    Results::to_file(experiment_name, str.str(), info.str(), results, folder_location);
                    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                }
            }
        };
    };
}