#ifndef SERVER_H
#define SERVER_H

#include <string>
#include <vector>
#include <grpcpp/grpcpp.h>
#include "matrix.grpc.pb.h"
#include "matrix.h"

using grpc::ServerContext;
using grpc::Status;
using matrix::MatrixRequest;
using matrix::MatrixReply;
using matrix::MatrixService;

class Server final : public MatrixService::Service {
public:
    Server(int port);

    void listenForConnection();
    Status MultiplyMatrix(ServerContext* context, const MatrixRequest* request, MatrixReply* reply) override;

private:
    int port;

    matrix_t multiplyMatrixParallel(const matrix_t& a, const matrix_t& b, int numProcesses) const;
    matrix_t multiplyMatrix(const matrix_t& a, const matrix_t& b) const;
    matrix_t genMatrix(int row, int col) const;

    std::vector<pid_t> forkProcesses(const matrix_t& a, const matrix_t& b, matrix_t& result, int* sharedResult, int numProcesses) const;
    void computeRows(const matrix_t& a, const matrix_t& b, matrix_t& result, int* sharedResult, int startRow, int endRow) const;
    void waitForChildren(const std::vector<pid_t>& children) const;
    void copySharedToResult(matrix_t& result, int* sharedResult) const;
};

#endif