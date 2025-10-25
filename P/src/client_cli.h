#ifndef CLIENT_CLI_H
#define CLIENT_CLI_H

#include <string>

namespace Client_CLI {

    struct Config {
        const char* serverIP = "127.0.0.1";
        int serverPort = 8080;
        int rows = 0;
        int cols = 0;
    };

    bool parsePositiveInt(const char* str, int& value, const char* fieldName);
    bool parsePort(const char* portStr, int& port);
    bool parseArguments(int argc, char* argv[], Config& config);
    void printUsage(const char* programName);
    std::string formatMessage(int rows, int cols);
}

#endif