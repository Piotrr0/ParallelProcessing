#ifndef CLIENT_H
#define CLIENT_H

#include <string>
#include <memory>
#include <grpcpp/grpcpp.h>
#include "matrix.grpc.pb.h"
#include "matrix.pb.h"

using matrix::MatrixReply;

class Client {
public:
    Client(int serverPort, const char* serverIP);

    bool sendMessenge(int rows_a, int cols_a, int rows_b, int cols_b, int forks, const std::vector<int>& matA, const std::vector<int>& matB, bool useCustom);
    bool receiveMessage();

private:
    std::unique_ptr<matrix::MatrixService::Stub> stub;
    std::string lastResponse;
    MatrixReply reply;
};

#endif