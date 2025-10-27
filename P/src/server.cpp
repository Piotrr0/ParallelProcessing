#include <cstdlib>
#include <print>
#include <format>
#include <cstring>
#include <random>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "server.h"
#include "timer.h"

Server::Server(int port) {
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket < 0)
        return;
        
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(port);
    serverAddress.sin_addr.s_addr = INADDR_ANY;

    int opt = 1;
    setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    if (bind(serverSocket, (struct sockaddr*)&serverAddress, sizeof(serverAddress)) < 0)
        return;
    }

Server::~Server() {
    if (connectedClientSocket >= 0)
        close(connectedClientSocket);

    if (serverSocket >= 0)
        close(serverSocket);
}

bool Server::isClientConnected() const {
    if (connectedClientSocket < 0)
        return false;
    
    int error = 0;
    socklen_t len = sizeof(error);
    int retval = getsockopt(connectedClientSocket, SOL_SOCKET, SO_ERROR, &error, &len);
    
    return (retval == 0 && error == 0);
}

void Server::listenForConnection() {
    if (serverSocket < 0 || listen(serverSocket, maxPendingConnections) < 0)
        return;

    while(true)
    {
        sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(struct sockaddr_in);

        connectedClientSocket = accept(serverSocket, nullptr, nullptr);
        if (connectedClientSocket < 0)
            return;

        handleClient();

        close(connectedClientSocket);
        connectedClientSocket = -1;
    }
}

void Server::handleClient() {
    while (isClientConnected()) {
        if (!receiveMessage()) {
            break;
        }
    }
}

bool Server::receiveMessage() {
    char messageBuffer[MAX_MESSAGE_LENGTH];
    ssize_t bytesReceived = recv(connectedClientSocket, messageBuffer, sizeof(messageBuffer) - 1, 0);
    if (bytesReceived == -1)
        return false;
    
    messageBuffer[bytesReceived] = '\0';
    parseReceiveMessge(messageBuffer);
    return true;
}

bool Server::sendMessage(const char* msg) {
    if (!isClientConnected())
        return false;

    ssize_t bytesSent = send(connectedClientSocket, msg, strlen(msg), 0);
    if (bytesSent == -1)
        return false;

    return true;
}

void Server::parseReceiveMessge(char msg[MAX_MESSAGE_LENGTH]) {
    int rows, cols, processes;
    if (sscanf(msg, "%d, %d, %d", &rows, &cols, &processes) != 3)
        return;

    matrix_t a = genMatrix(rows, cols);
    matrix_t b = genMatrix(rows, cols);

    timerOutput_t parallelTiming;
    timerOutput_t serialTiming;
    {
        Timer t(&parallelTiming);
        multiplyMatrixParallel(a, b, processes);
    }

    {
        Timer t(&serialTiming);
        multiplyMatrix(a, b);
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
        "Speedup: {:.2f}x\n\n",
        rows, cols,
        processes,
        parallelTiming.cpuTime, parallelTiming.elapsedTime,
        serialTiming.cpuTime, serialTiming.elapsedTime,
        speedup
    );

    sendMessage(timingMsg.c_str());
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

    for (int p = 0; p < numProcesses; p++) {
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
    for (int i = startRow; i < endRow; i++) {
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
