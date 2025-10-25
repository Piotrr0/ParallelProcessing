#ifndef SERVER_H
#define SERVER_H

#include <netinet/in.h>
#include "matrix.h"

#define MAX_MESSAGE_LENGTH 1024

class Server {
public:
    const int maxPendingConnections = 5;

    int serverSocket = -1;
    int connectedClientSocket = -1;

    sockaddr_in serverAddress{};
    char messageBuffer[MAX_MESSAGE_LENGTH];

    Server(int port);
    ~Server();

    void listenForConnection();
    void receiveMessage();

    matrix_t multiplyMatrixParallel(const matrix_t& a, const matrix_t& b, int numProcesses) const;
    matrix_t multiplyMatrix(const matrix_t& a, const matrix_t& b) const;
private:
    std::vector<pid_t> forkProcesses(const matrix_t& a, const matrix_t& b, matrix_t& result, int* sharedResult, int numProcesses) const;
    void computeRows(const matrix_t& a, const matrix_t& b, matrix_t& result, int* sharedResult, int startRow, int endRow) const;
    void waitForChildren(const std::vector<pid_t>& children) const;
    void copySharedToResult(matrix_t& result, int* sharedResult) const;

};

#endif