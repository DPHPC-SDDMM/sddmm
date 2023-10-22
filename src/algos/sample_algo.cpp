/**
 * Exercise 1 of hpc
 * 
 * A simple double loop with some performance measurement
*/

#include "../defines.h"
#include <iostream>
#include <time.h>
#include <chrono>
#include <vector>

namespace SDDMM {

    class SampleAlgo {
        void compute_steps(double* T1, double* T2, int N, int M){
            int n_end = N-1;
            int m_end = M-1;
            int n=0;
            int m=0;

            // middle
            for(n=1; n<n_end; n++){
                for(m=1; m<m_end; m++){
                    T2[n*N + m] = (
                        T1[(n-1)*N + m] + 
                        T1[(n-1)*N + m] + 
                        T1[n*N + m-1] + 
                        T1[n*N + m+1] +
                        T1[n*N + m]
                    ) / 5.0;
                }
            }

            // corners
            n=0; m=0;
            T2[n*N + m] = (T1[(n+1)*N + m] + T1[n*N + m+1] + T1[n*N + m]) / 3.0;
            
            n=0; m=m_end;
            T2[n*N + m] = (T1[(n+1)*N + m] + T1[n*N + m-1] + T1[n*N + m]) / 3.0;
            
            n=n_end; m=0;
            T2[n*N + m] = (T1[(n-1)*N + m] + T1[n*N + m+1] + T1[n*N + m]) / 3.0;
            
            n=n_end; m=m_end;
            T2[n*N + m] = (T1[(n-1)*N + m] + T1[n*N + m-11] + T1[n*N + m]) / 3.0;

            // edges
            n=0;
            for(m=1; m<m_end; m++){
                T2[n*N + m] = (T1[n*N + m-1] + T1[n*N + m+1] + T1[(n+1)*N + m] + T1[n*N + m]) / 4.0;
            }

            n=n_end;
            for(m=1; m<m_end; m++){
                T2[n*N + m] = (T1[n*N + m-1] + T1[n*N + m+1] + T1[(n-1)*N + m] + T1[n*N + m]) / 4.0;
            }

            m=0;
            for(n=1; n<n_end; n++){
                T2[n*N + m] = (T1[(n-1)*N + m] + T1[n*N + m] + T1[(n+1)*N + m] + T1[n*N + m+1]) / 4.0;
            }

            m=m_end;
            for(n=1; n<n_end; n++){
                T2[n*N + m] = (T1[(n-1)*N + m] + T1[n*N + m] + T1[(n+1)*N + m] + T1[n*N + m+1]) / 4.0;
            }
        }

        double total_energy(double* T, int N, int M){
            double res = 0;
            for(int n=0; n<N; ++n){
                for(int m=0; m<M; ++m){
                    res += T[n*N + m];
                }
            }
            return res;
        }

        void init(double* T, int N, int M){
            for(int n=0; n<N; ++n){
                for(int m=0; m<M; ++m){
                    T[n*N + m] = 0.0;
                }
            }
            T[(N/2)*N + (M/2)] = 1000;
        }

        void swap(double*& T1, double*& T2){
            double* tmp = T1;
            T1 = T2;
            T2 = tmp;
        }

    public:
        SampleAlgo(Defines::ErrPlotData& data, int n_experiments, int N_start, int N_stop, int N_delta){
            double n = N_start;
            Defines::vector_fill(data.x, N_start, N_delta, N_stop);
            data.min = N_start - N_delta;
            data.max = N_stop + N_delta;

            int num_steps = 100;
            int experiment = 0;
            for(int N=N_start; N<=N_stop; N+=N_delta){
                std::cout << "Start experiment with " << N << std::endl;
                data.runtimes.push_back(std::vector<double>());
                for(int n=0; n<n_experiments; ++n){
                    if(n%50==0) std::cout << " -- run experiment " << n << "/" << n_experiments << std::endl;
                    int M = N;
                    double* T1 = new double[N*M];
                    double* T2 = new double[N*M];

                    init(T1, N, M);
                    init(T2, N, M);

                    double total_start_t1 = total_energy(T1, N, M);
                    double total_start_t2 = total_energy(T2, N, M);

                    auto start = std::chrono::high_resolution_clock::now();
                    for(int t=0; t<num_steps; t++){
                        compute_steps(T1, T2, N, M);
                        swap(T1, T2);
                    }
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

                    double total_end_t1 = total_energy(T1, N, M);
                    double total_end_t2 = total_energy(T2, N, M);

                    data.runtimes.at(experiment).push_back(duration.count());

                    delete[] T1;
                    delete[] T2;
                }
                experiment++;
            }   
        }   
    };    
}
