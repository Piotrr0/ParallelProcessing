#include <cstdlib>
#include <print>
#include <format>
#include <cstring>
#include <random>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <unistd.h>
#include <grpcpp/grpcpp.h>
#include "server.h"
#include "timer.h"

using grpc::ServerBuilder;

Server::Server(int port) : port(port) {}

void Server::listenForConnection() {
    std::string server_address = std::format("0.0.0.0:{}", port);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(this);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    server->Wait();
}

Status Server::MultiplyMatrix(ServerContext* context, const MatrixRequest* request, MatrixReply* reply) {
    int rows_a = request->rows_a();
    int cols_a = request->cols_a();
    int rows_b = request->rows_b();
    int cols_b = request->cols_b();

    if (cols_a != rows_b) {
        std::string error = std::format("Dimension mismatch: Matrix A columns ({}) != Matrix B rows ({})", cols_a, rows_b);
        return Status(grpc::StatusCode::INVALID_ARGUMENT, error);
    }

    int processes = request->forks();

    matrix_t a;
    matrix_t b;

    if (request->use_custom())
    {
        a.resize(rows_a, cols_a);
        b.resize(rows_b, cols_b);
        
        for (int i = 0; i < request->matrix_a_size(); ++i) {
            a.data[i] = request->matrix_a(i);
        }
        for (int i = 0; i < request->matrix_b_size(); ++i) {
            b.data[i] = request->matrix_b(i);
        }
    }
    else
    {
        a = genMatrix(rows_a, cols_a);
        b = genMatrix(rows_b, cols_b);
    }

    timerOutput_t parallelTiming;
    timerOutput_t serialTiming;
    matrix_t resultParallel;
    matrix_t resultSerial;
    {
        Timer t(&parallelTiming);
        resultParallel = multiplyMatrixParallel(a, b, processes);
    }

    {
        Timer t(&serialTiming);
        resultSerial = multiplyMatrix(a, b);
    }

    double speedup = serialTiming.elapsedTime > 0 ? serialTiming.elapsedTime / parallelTiming.elapsedTime : 0.0;

    std::string timingMsg = std::format(
        "Matrix {}x{} multiplication completed:\n"
        "Parallel ({} processes):\n"
        "  CPU time: {:.6f}s\n"
        "  time: {}ms\n"
        "Serial:\n"
        "  CPU time: {:.6f}s\n"
        "  time: {}ms\n"
        "Speedup: {:.2f}x\n",
        rows_a, cols_b,
        processes,
        parallelTiming.cpuTime, parallelTiming.elapsedTime,
        serialTiming.cpuTime, serialTiming.elapsedTime,
        speedup
    );

    reply->set_message(timingMsg);
    reply->set_result_rows(resultParallel.rows);
    reply->set_result_cols(resultParallel.cols);
    
    for (int val : resultParallel.data) {
        reply->add_result_matrix(val);
    }
    return Status::OK;
}

matrix_t Server::multiplyMatrixParallel(const matrix_t& a, const matrix_t& b, int numProcesses) const {
    matrix_t result(a.rows, b.cols);

    size_t sharedSize = result.rows * result.cols * sizeof(int);
    int* sharedResult = (int*)mmap(NULL, sharedSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (sharedResult == MAP_FAILED) {
        return result;
    }

    memset(sharedResult, 0, sharedSize);
    std::vector<pid_t> children = forkProcesses(a, b, result, sharedResult, numProcesses);

    waitForChildren(children);
    copySharedToResult(result, sharedResult);

    munmap(sharedResult, sharedSize);
    return result;
}

std::vector<pid_t> Server::forkProcesses(const matrix_t& a, const matrix_t& b, matrix_t& result, int* sharedResult, int numProcesses) const {
    std::vector<pid_t> children;

    for (int p = 0; p < numProcesses; p++)
    {
        int startRow = p * result.rows / numProcesses;
        int endRow = (p + 1) * result.rows / numProcesses;

        pid_t pid = fork();
        if (pid < 0) {
            continue;
        } 
        else if (pid == 0) {
            computeRows(a, b, result, sharedResult, startRow, endRow);
            exit(0);
        }
        else {
            children.push_back(pid);
        }
    }

    return children;
}

void Server::computeRows(const matrix_t& a, const matrix_t& b, matrix_t& result, int* sharedResult, int startRow, int endRow) const {
    for (int i = startRow; i < endRow; i++)
    {
        for (int j = 0; j < b.cols; j++) {
            int sum = 0;
            for (int k = 0; k < a.cols; k++) {
                sum += a.at(i, k) * b.at(k, j);
            }
            sharedResult[i * result.cols + j] = sum;
        }
    }
}

void Server::waitForChildren(const std::vector<pid_t>& children) const {
    for (pid_t pid : children) {
        int status;
        waitpid(pid, &status, 0);
    }
}

void Server::copySharedToResult(matrix_t& result, int* sharedResult) const {
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            result.at(i, j) = sharedResult[i * result.cols + j];
        }
    }
}

matrix_t Server::multiplyMatrix(const matrix_t& a, const matrix_t& b) const {
    matrix_t result(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            result.at(i,j) = 0;
            for (int k = 0; k < a.cols; k++) {
                result.at(i, j) += a.at(i, k) * b.at(k, j);
            }
        }
    }
    return result;
}

matrix_t Server::genMatrix(int row, int col) const {
    matrix_t randomMatrix(row, col);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);

    for (int i = 0; i<row; i++)
    {
        for(int j = 0; j<col; j++)
        {
            randomMatrix.at(i, j) = dis(gen);
        }
    }

    return randomMatrix;
}

int main() {
    Server server(8080);
    server.listenForConnection();
    return 0;
}