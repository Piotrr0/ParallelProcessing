#include <omp.h>
#include <iostream>
#include <cmath>
#include <chrono>

#define THREADS_COUNT 8

enum Type {
    STATIC,
    DYNAMIC,
    GUIDED,
    AUTO
};

class Timer {
public:
    Timer() {
        std::cout << "Start Timer\n";
        startCPU = clock();
        start = std::chrono::high_resolution_clock::now();
        startOMP = omp_get_wtime();
    }
    
    ~Timer() {
        stopCPU = clock();
        stop = std::chrono::high_resolution_clock::now();
        stopOMP = omp_get_wtime();
        
        const double cpuTime = double(stopCPU - startCPU) / CLOCKS_PER_SEC;
        const double time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        const double ompTime = stopOMP - startOMP;
        
        std::cout << "CPU time: " << cpuTime << "\n";
        std::cout << "time: " << time << "\n";
        std::cout << "OMP time: " << ompTime << "\n";
    }
    
private:
    clock_t startCPU;
    clock_t stopCPU;
    double startOMP;
    double stopOMP;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop;
};

int factorial(int n) {
    int out = 1;
    #pragma omp parallel for reduction(*:out)
    for (int i = 2; i <= n; i++) {
        out *= i;
    }
    return out;
}

int newton(int n, int k) {
    int nFactorial, kFactorial, n_kFactorial;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            nFactorial = factorial(n);
        }

        #pragma omp section
        {
            kFactorial = factorial(k);
        }

        #pragma omp section
        {
            n_kFactorial = factorial(n - k);
        }
    }

    return nFactorial / (kFactorial * n_kFactorial);
}

bool is_prime(int n, Type type) {
    if (n <= 1) 
        return false;
    if (n == 2)
        return true;
    if (n % 2 == 0)
        return false;

    bool isPrime = true;
    int maxCheck = static_cast<int>(std::sqrt(n));

    switch (type) {
        case STATIC:
            #pragma omp parallel for schedule(static) shared(isPrime)
            for (int i = 3; i <= maxCheck; i += 2) {
                if (isPrime && n % i == 0) {
                    #pragma omp critical
                    {
                        isPrime = false;
                    }
                }
            }
            break;

        case DYNAMIC:
            #pragma omp parallel for schedule(dynamic) shared(isPrime)
            for (int i = 3; i <= maxCheck; i += 2) {
                if (isPrime && n % i == 0) {
                    #pragma omp critical
                    {
                        isPrime = false;
                    }
                }
            }
            break;

        case GUIDED:
            #pragma omp parallel for schedule(guided) shared(isPrime)
            for (int i = 3; i <= maxCheck; i += 2) {
                if (isPrime && n % i == 0) {
                    #pragma omp critical
                    {
                        isPrime = false;
                    }
                }
            }
            break;

            
        case AUTO:
            #pragma omp parallel for schedule(auto) shared(isPrime)
            for (int i = 3; i <= maxCheck; i += 2) {
                if (isPrime && n % i == 0) {
                    #pragma omp critical
                    {
                        isPrime = false;
                    }
                }
            }
            break;
    }

    return isPrime;
}

float example_func(float x) {
    return 4.f / (1.f + std::pow(x, 2));
}

double calculate_integral_omp(double start, double end, float(*func)(float), size_t steps = 10000000)
{
    auto area = [](float a, float b, float h) -> float {
        return ((a + b) * h) / 2.f;
    };
    
    double integral = 0;
    const float step = (end - start) / steps;
    
    #pragma omp parallel for reduction(+:integral)
    for (size_t i = 0; i < steps; i++) {
        float x0 = start + i * step;
        float x1 = start + (i + 1) * step;

        float y0 = func(x0);
        float y1 = func(x1);

        integral += area(y0, y1, step);
    }
    
    return integral;
}

int main() {
    omp_set_num_threads(THREADS_COUNT);

    #pragma omp parallel
    {
        #pragma omp master
        {
            std::cout << "Hello from master thread " << omp_get_thread_num() << std::endl;
        }

        #pragma omp single
        {
            std::cout << "Hello from single thread " << omp_get_thread_num() << std::endl;
        }

        std::cout << "Hello from thread " << omp_get_thread_num() << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Result: " << newton(7, 3) << std::endl << std::endl;
    

    {
        Timer timer;
        std::cout << "Result: " << calculate_integral_omp(0, 50000000, example_func) << "\n";
    }


    std::cout << std::endl;
    {
        std::cout << "STATIC\n";
        Timer timer;
        std::cout << (is_prime(13147, STATIC) ? "Prime" : "Not Prime") << std::endl;
    }
    std::cout << std::endl;
    {
        std::cout << "DYNAMIC\n";
        Timer timer;
        std::cout << (is_prime(13147, DYNAMIC) ? "Prime" : "Not Prime") << std::endl;
    }
    std::cout << std::endl;
    {
        std::cout << "GUIDED\n";
        Timer timer;
        std::cout << (is_prime(13147, GUIDED) ? "Prime" : "Not Prime") << std::endl;
    }
    std::cout << std::endl;
    {
        std::cout << "AUTO\n";
        Timer timer;
        std::cout << (is_prime(13147, AUTO) ? "Prime" : "Not Prime") << std::endl;
    }
    
    return 0;
}