#include <cstdlib>
#include <print>
#include <cstring>
#include <random>
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

    if (bind(serverSocket, (struct sockaddr*)&serverAddress, sizeof(serverAddress)) < 0)
        return;
    }

Server::~Server() {
    if (connectedClientSocket >= 0)
        close(connectedClientSocket);

    if (serverSocket >= 0)
        close(serverSocket);
}

void Server::listenForConnection() {
    if (listen(serverSocket, maxPendingConnections) < 0)
        return;

    connectedClientSocket = accept(serverSocket, nullptr, nullptr);
    if (connectedClientSocket < 0)
        return;

    receiveMessage();
}

void Server::receiveMessage() {
    char messageBuffer[MAX_MESSAGE_LENGTH];
    int bytesReceived = recv(connectedClientSocket, messageBuffer, sizeof(messageBuffer) - 1, 0);
    if (bytesReceived > 0) {
        messageBuffer[bytesReceived] = '\0';
        parseReceiveMessge(messageBuffer);
    }
}

void Server::sendMessage(const char* msg) {
    int bytesSent = send(connectedClientSocket, msg, strlen(msg), 0);
    if (bytesSent == -1)
        return;
}

void Server::parseReceiveMessge(char msg[MAX_MESSAGE_LENGTH]) {
    int rows, cols;
    if (sscanf(msg, "%d, %d", &rows, &cols) != 2)
        return;

    matrix_t a = genMatrix(rows, cols);
    matrix_t b = genMatrix(rows, cols);
    {
        Timer t;
        multiplyMatrixParallel(a, b, 5);
    }

    {
        Timer t;
        multiplyMatrix(a, b);
    }

    std::string response = "Done\n";
    sendMessage(response.c_str());
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
