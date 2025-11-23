#ifndef CLIENT_CLI_H
#define CLIENT_CLI_H

#include <string>
#include <vector>

namespace Client_CLI {

    struct Config {
        const char* serverIP = "127.0.0.1";
        int serverPort = 8080;
        int rows_a = 0;
        int cols_a = 0;

        int rows_b = 0;
        int cols_b = 0;

        int forks = 4;

        bool useCustom = false;
        std::vector<int> customValues;

        std::vector<int> matrixA;
        std::vector<int> matrixB;
    };

    bool parsePositiveInt(const char* str, int& value, const char* fieldName);
    bool parsePort(const char* portStr, int& port);
    bool parseArguments(int argc, char* argv[], Config& config);
    void printUsage(const char* programName);

}

#endif