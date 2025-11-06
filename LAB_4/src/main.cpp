#include <omp.h>
#include <iostream>
#include <cmath>

#define THREADS_COUNT 5

enum Type {
    STATIC,
    DYNAMIC,
    GUIDED,
    AUTO
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
                    isPrime = false;
                }
            }
            break;

        case DYNAMIC:
            #pragma omp parallel for schedule(dynamic) shared(isPrime)
            for (int i = 3; i <= maxCheck; i += 2) {
                if (isPrime && n % i == 0) {
                    isPrime = false;
                }
            }
            break;

        case GUIDED:
            #pragma omp parallel for schedule(guided) shared(isPrime)
            for (int i = 3; i <= maxCheck; i += 2) {
                if (isPrime && n % i == 0) {
                    isPrime = false;
                }
            }
            break;

        case AUTO:
            #pragma omp parallel for schedule(auto) shared(isPrime)
            for (int i = 3; i <= maxCheck; i += 2) {
                if (isPrime && n % i == 0) {
                    isPrime = false;
                }
            }
            break;
    }

    return isPrime;
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
    std::cout << "Result: " << newton(7, 3) << std::endl;
    std::cout << (is_prime(13147, STATIC) ? "Prime" : "Not Prime");
    return 0;
}