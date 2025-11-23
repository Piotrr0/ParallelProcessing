#include <print>
#include <string>
#include "client_cli.h"

bool Client_CLI::parsePositiveInt(const char* str, int& value, const char* fieldName) {
    value = std::stoi(str);
    if (value <= 0) {
        std::println("Error: {} must be a positive integer", fieldName);
        return false;
    }
    return true;
}

bool Client_CLI::parsePort(const char* portStr, int& port) {
    port = std::stoi(portStr);
    if (port < 1 || port > 65535) {
        std::println("Error: Port must be between 1 and 65535");
        return false;
    }
    return true;
}

bool Client_CLI::parseArguments(int argc, char* argv[], Config& config) {
    if (argc == 1) {
        return true;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-s" || arg == "--server") {
            if (i + 1 < argc) {
                config.serverIP = argv[++i];
            } 
            else {
                std::println("Error: {} requires an argument", arg);
                return false;
            }
        }

        else if (arg == "-p" || arg == "--port") {
            if (i + 1 < argc) {
                if (!parsePort(argv[++i], config.serverPort)) {
                    return false;
                }
            }
            else {
                std::println("Error: {} requires an argument", arg);
                return false;
            }
        }

        else if (arg == "-r" || arg == "--rows") {
            if (i + 1 < argc) {
                if (!parsePositiveInt(argv[++i], config.rows, "rows")) {
                    return false;
                }
            } else {
                std::println("Error: {} requires an argument", arg);
                return false;
            }
        }
        
        else if (arg == "-c" || arg == "--cols") {
            if (i + 1 < argc) {
                if (!parsePositiveInt(argv[++i], config.cols, "cols")) {
                    return false;
                }
            } else {
                std::println("Error: {} requires an argument", arg);
                return false;
            }
        }

        else if (arg == "-f" || arg == "-forks") {
            if (i + 1 < argc) {
                if (!parsePositiveInt(argv[++i], config.forks, "processes")) {
                    return false;
                }
            } else {
                std::println("Error: {} requires an argument", arg);
                return false;
            }
        }
    }
    
    return true;
}

void Client_CLI::printUsage(const char* programName)
{
    std::println("Client");
    std::println("Usage: {} [OPTIONS]", programName);
    std::println("\nOptions:");
    std::println("  -s, --server IP      Server IP address (default: 127.0.0.1)");
    std::println("  -p, --port PORT      Server port (default: 8080)");
    std::println("  -r, --rows ROWS      Number of matrix rows (required)");
    std::println("  -c, --cols COLS      Number of matrix columns (required)");
    std::println("  -f, --forks FORKS    Number of processes (default: 4)");
}

std::string Client_CLI::formatMessage(int rows, int cols, int forks) {
    return std::to_string(rows) + ", " + std::to_string(cols) + ", " + std::to_string(forks);
}