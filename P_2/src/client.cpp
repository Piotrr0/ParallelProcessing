#include <string>
#include <print>
#include <memory>
#include <grpcpp/grpcpp.h>
#include "client.h"
#include "client_cli.h"
#include "matrix.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using matrix::MatrixRequest;
using matrix::MatrixReply;
using matrix::MatrixService;

Client::Client(int serverPort, const char* serverIP) {
    std::string target_str = std::string(serverIP) + ":" + std::to_string(serverPort);
    stub = MatrixService::NewStub(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
}

bool Client::sendMessenge(int rows_a, int cols_a, int rows_b, int cols_b, int forks, const std::vector<int>& matA, const std::vector<int>& matB, bool useCustom) {
    MatrixRequest request;
    request.set_rows_a(rows_a);
    request.set_cols_a(cols_a);

    request.set_rows_b(rows_b);
    request.set_cols_b(cols_b);

    request.set_forks(forks);
    request.set_use_custom(useCustom);

    if (useCustom) {
        for (int val : matA) {
            request.add_matrix_a(val);
        }
        for (int val : matB) {
            request.add_matrix_b(val);
        }
    }

    ClientContext context;
    Status status = stub->MultiplyMatrix(&context, request, &reply);

    if (status.ok()) {
        lastResponse = reply.message();
        return true;
    } 
    else {
        std::println("RPC failed");
        return false;
    }
}

bool Client::receiveMessage() {
    if (lastResponse.empty())
        return false;

    std::println("{}", lastResponse);

    if (reply.result_matrix_size() <= 0)
        return false;

    std::println("Result Matrix ({}x{}):", reply.result_rows(), reply.result_cols());

    for (int i = 0; i < reply.result_rows(); i++)
    {
        for (int j = 0; j < reply.result_cols(); j++)
        {
            int index = i * reply.result_cols() + j;
            std::print("{} ", reply.result_matrix(index));
        }
        std::println("");
    }

    return true;
}

int main(int argc, char* argv[])
{
    using namespace Client_CLI;

    if (argc == 1) {
        printUsage(argv[0]);
        return 0;
    }

    Config config;
    if (!parseArguments(argc, argv, config)) {
        return 1;
    }

    Client client(config.serverPort, config.serverIP);
    client.sendMessenge(config.rows_a, config.cols_a, config.rows_b, config.cols_b, config.forks, config.matrixA, config.matrixB, config.useCustom);
    client.receiveMessage();
    return 0;
}