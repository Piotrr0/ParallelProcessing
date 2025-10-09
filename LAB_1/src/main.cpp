#include <cstddef>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include <vector>
#include <pthread.h>

class Timer {
    public:
        Timer() {
            std::cout << "Start Timer\n";
            startCPU = clock();
            start = std::chrono::high_resolution_clock::now();
        }

        ~Timer() {
            stopCPU = clock();
            stop = std::chrono::high_resolution_clock::now();

            const double cpuTime = double(stopCPU - startCPU) / CLOCKS_PER_SEC;
            const auto time = stop - start;

            std::cout << "CPU time: " << cpuTime << "\n";
            std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << "\n";
        }



    private:
        clock_t startCPU;
        clock_t stopCPU;
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        std::chrono::time_point<std::chrono::high_resolution_clock> stop;
};


int fib(int a) {
    if (a == 0)
        return 0;

    if (a == 1)
        return 1;

    return (fib(a-1) + fib(a-2));
}

void* thread_heavy_func(void * args) {
    sleep(1);
    return nullptr;
}

int main() {

    {
        Timer t;
        for(int i = 0; i<20; i++) {
            int fib_i = fib(i);
        }
    }
    

    {
        Timer t;

        int number;
        std::cout << "Enter number: " << std::endl;
        std::cin >> number;

        std::cout << "Your number: " << number << "\n";
    }


    {
        Timer t;
        sleep(3);
    }


    {
        Timer t;

        const size_t pids_count = 10;
        std::vector<pid_t> pids;
        pids.reserve(pids_count);
        for (int i = 0; i<pids_count; i++) {
            pid_t pid = fork();
            pids.emplace_back(pid);
        }

        for (const auto& pid : pids) {
            if(pid == 0)
                exit(0);
            else if (pid > 0)
                wait(0);
            else
                std::cout << "Error!\n" << std::endl;
        }
    }

    {
        Timer t;

        const size_t threads_count = 10;
        std::vector<pthread_t> threads;
        threads.reserve(threads_count);

        for (int i = 0; i<threads_count; i++)
        {
            pthread_t tid;
            pthread_create(&tid, nullptr, &thread_heavy_func, nullptr);

            threads.emplace_back(tid);
        }

        for (const auto& thread : threads) {
            pthread_join(thread, nullptr);
        }
    }

    return 0;
}