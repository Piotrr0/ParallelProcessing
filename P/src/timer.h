#include <print>
#include <chrono>

struct timerOutput_t {
    double cpuTime;
    double elapsedTime;
};

class Timer {
    public:
        Timer(timerOutput_t* output) : output(output) {
            std::println("Start Timer");
            startCPU = clock();
            start = std::chrono::high_resolution_clock::now();
        }

        ~Timer() {
            stopCPU = clock();
            stop = std::chrono::high_resolution_clock::now();

            const auto cpuTime = double(stopCPU - startCPU) / CLOCKS_PER_SEC;

            const auto time = stop - start;
            const double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(time).count(); 

            if (output != nullptr)
            {
                output->cpuTime = cpuTime;
                output->elapsedTime = elapsedTime;
            }


            std::println("CPU time: {}", cpuTime);
            std::println("time: {}", elapsedTime);
        }
        
    private:
        clock_t startCPU;
        clock_t stopCPU;
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        std::chrono::time_point<std::chrono::high_resolution_clock> stop;

        timerOutput_t* output;
};