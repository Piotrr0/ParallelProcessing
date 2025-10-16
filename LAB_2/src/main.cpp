#include <cmath>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include <pthread.h>
#include <vector>

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

typedef struct {
    float start;
    float end;
    double result;
    pthread_t tid;
} IntegralThreadInfo;

float example_func(float x) {
    return 4.f / (1.f + std::pow(x, 2));
}

double calculate_integral(float start, float end, float(*func)(float), size_t steps = 100000)
{
    auto area = [](float a, float b, float h) -> float {
        return ((a + b) * h) / 2.f;
    };

    double integral = 0;
    const float step = (end - start) / steps;

    for (int i = 0; i < steps; i++) {
        float x0 = start + i * step;
        float x1 = start + (i + 1) * step;

        float y0 = func(x0);
        float y1 = func(x1);

        integral += area(y0, y1, step);
    }

    return integral;
}

void* thread_integral(void* arg) {
    IntegralThreadInfo* info = static_cast<IntegralThreadInfo*>(arg);
    info->result = calculate_integral(info->start, info->end, example_func);
    return nullptr;
}

double calculate_integral_threaded(float start, float end, float(*func)(float), size_t steps = 10000000, size_t threads_count = 5)
{
    std::vector<IntegralThreadInfo> infos(threads_count);

    const float step = (end - start) / threads_count;

    for (int i = 0; i < threads_count; i++) {
        infos[i].start = start + i * step;
        infos[i].end = start + (i + 1) * step;

        pthread_create(&infos[i].tid, nullptr, thread_integral, &infos[i]);
    }

    double total = 0;
    for (int i = 0; i < threads_count; i++) {
        pthread_join(infos[i].tid, nullptr);
        total += infos[i].result;
    }
    return total;
}


int main() {
    
    {
        Timer timer;
        std::cout << "Result: " << calculate_integral_threaded(0, 1000000, example_func) << "\n";
    }

    std:: cout << "\n";

    {
        Timer timer;
        std::cout << "Result: " << calculate_integral(0, 1000000, example_func) << "\n";
    }

    return 0;
}
