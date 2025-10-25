#include <print>
#include <chrono>

class Timer {
    public:
        Timer() {
            std::println("Start Timer");
            startCPU = clock();
            start = std::chrono::high_resolution_clock::now();
        }

        ~Timer() {
            stopCPU = clock();
            stop = std::chrono::high_resolution_clock::now();

            const double cpuTime = double(stopCPU - startCPU) / CLOCKS_PER_SEC;
            const auto time = stop - start;

            std::println("CPU time: {}", cpuTime);
            std::println("time: {}", std::chrono::duration_cast<std::chrono::milliseconds>(time).count());
        }
        
    private:
        clock_t startCPU;
        clock_t stopCPU;
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        std::chrono::time_point<std::chrono::high_resolution_clock> stop;
};