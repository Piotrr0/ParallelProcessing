#include <cmath>
#include <cstddef>
#include <ostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include <pthread.h>
#include <vector>
#include <atomic>

struct TimerOutput {
    double time;
    double CPUtime;
};

class Timer {
    public:
        Timer(TimerOutput& output) : output(output) {
            std::cout << "Start Timer\n";
            startCPU = clock();
            start = std::chrono::high_resolution_clock::now();
        }

        ~Timer() {
            stopCPU = clock();
            stop = std::chrono::high_resolution_clock::now();

            const double cpuTime = double(stopCPU - startCPU) / CLOCKS_PER_SEC;
            const double time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

            output.CPUtime = cpuTime;
            output.time = time;

            std::cout << "CPU time: " << cpuTime << "\n";
            std::cout << "time: " << time << "\n";
        }

    private:
        clock_t startCPU;
        clock_t stopCPU;
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        std::chrono::time_point<std::chrono::high_resolution_clock> stop;
        TimerOutput& output;
};

typedef struct {
    float start;
    float end;
    size_t steps;
    double result;
    pthread_t tid;
} IntegralThreadInfo;

float example_func(float x) {
    return 4.f / (1.f + std::pow(x, 2));
}

double calculate_integral(double start, double end, float(*func)(float), size_t steps = 10000000)
{
    auto area = [](float a, float b, float h) -> float {
        return ((a + b) * h) / 2.f;
    };

    double integral = 0;
    const float step = (end - start) / steps;

    for (size_t i = 0; i < steps; i++) {
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
    info->result = calculate_integral(info->start, info->end, example_func, info->steps);
    return nullptr;
}

double calculate_integral_threaded(double start, float end, float(*func)(float), size_t steps = 10000000, size_t threads_count = 5)
{
    std::vector<IntegralThreadInfo> infos(threads_count);

    const float range_step = (end - start) / threads_count;
    const size_t steps_per_thread = steps / threads_count;

    for (size_t i = 0; i < threads_count; i++) {
        infos[i].start = start + i * range_step;
        infos[i].end = start + (i + 1) * range_step;
        infos[i].steps = steps_per_thread;

        pthread_create(&infos[i].tid, nullptr, thread_integral, &infos[i]);
    }

    double total = 0;
    for (size_t i = 0; i < threads_count; i++) {
        pthread_join(infos[i].tid, nullptr);
        total += infos[i].result;
    }
    return total;
}

constexpr int iterations = 10000;

int counter = 0;
void* threaded_function(void* args) {
    for (int i = 0; i<iterations; i++) {
        counter++;
    }

    return nullptr;
}

std::atomic_int atomicCounter = 0;
void* threaded_function_atomic(void* args) {
    for (int i = 0; i<iterations; i++) {
        atomicCounter.fetch_add(1, std::memory_order_relaxed);
    }

    return nullptr;
}

pthread_mutex_t mutex;
int result = pthread_mutex_init(&mutex, nullptr);

int mutexCounter = 0;
void* threaded_function_mutex(void* args) {
    for (int i = 0; i<iterations; i++) {
        pthread_mutex_lock(&mutex);
        mutexCounter++;
        pthread_mutex_unlock(&mutex);
    }

    return nullptr;
}

int main() {

    TimerOutput parallel;
    TimerOutput sequential;
    
    const size_t total_steps = 10000000;
    const size_t thread_count = 5;
    
    {
        Timer timer(parallel);
        std::cout << "Result: " << calculate_integral_threaded(0, 50000000, example_func, total_steps, thread_count) << "\n";
    }

    std::cout << "\n";

    {
        Timer timer(sequential);
        std::cout << "Result: " << calculate_integral(0, 50000000, example_func, total_steps) << "\n";
    }

    const double speedUp = sequential.time / parallel.time;
    std::cout << "SpeedUp: " << speedUp << "x" << std::endl;
    std::cout << std::endl;

    const int threadCount = 100; 
    std::vector<pthread_t> threads;
    threads.reserve(threadCount);
    for (int i = 0 ; i<threadCount; i++) {
        pthread_t tid;
        pthread_create(&tid, nullptr, threaded_function, nullptr);
        threads.emplace_back(tid);
    }

    for (auto& thread : threads) {
        pthread_join(thread, nullptr);
    }
    std::cout << "Counter: " << counter << std::endl;

    threads.clear();
    for (int i = 0 ; i<threadCount; i++) {
        pthread_t tid;
        pthread_create(&tid, nullptr, threaded_function_atomic, nullptr);
        threads.emplace_back(tid);
    }

    for (auto& thread : threads) {
        pthread_join(thread, nullptr);
    }
    std::cout << "Atomic Counter: " << atomicCounter << std::endl;

    threads.clear();
    for (int i = 0 ; i<threadCount; i++) {
        pthread_t tid;
        pthread_create(&tid, nullptr, threaded_function_mutex, nullptr);
        threads.emplace_back(tid);
    }

    for (auto& thread : threads) {
        pthread_join(thread, nullptr);
    }
    std::cout << "Mutex Counter: " << mutexCounter << std::endl;

    return 0;
}