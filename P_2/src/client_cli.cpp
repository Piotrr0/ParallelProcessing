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

        else if (arg == "-r_a" || arg == "--rows_a") {
            if (i + 1 < argc) { 
                if (!parsePositiveInt(argv[++i], config.rows_a, "rows_a")) 
                    return false;
            }
            else {
                 std::println("Error: {} requires an argument", arg);
                 return false;
            }
        }
        else if (arg == "-c_a" || arg == "--cols_a") {
            if (i + 1 < argc) {
                if (!parsePositiveInt(argv[++i], config.cols_a, "cols_a"))
                    return false;
                }
            else { 
                std::println("Error: {} requires an argument", arg);
                return false;
            }
        }
        else if (arg == "-r_b" || arg == "--rows_b") {
            if (i + 1 < argc) { 
                if (!parsePositiveInt(argv[++i], config.rows_b, "rows_b")) 
                    return false;
            }
            else { 
                std::println("Error: {} requires an argument", arg);
                return false; 
            }
        }
        else if (arg == "-c_b" || arg == "--cols_b") {
            if (i + 1 < argc) {
                if (!parsePositiveInt(argv[++i], config.cols_b, "cols_b")) 
                    return false;
                }
            else { 
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
        else if (arg == "--custom") {
            config.useCustom = true;
        }
        else {
            int val = std::stoi(arg);
            config.customValues.push_back(val);
        }
    }

    if (config.cols_a != config.rows_b) {
        std::println("Error: Matrix Multiplication Rule Violated.\n");
        std::println("Columns of A ({}) must equal Rows of B ({})", config.cols_a, config.rows_b);
        return false;
    }

    if (config.useCustom) {
        size_t sizeA = config.rows_a * config.cols_a;
        size_t sizeB = config.rows_b * config.cols_b;
        size_t expected = sizeA + sizeB;

        if (config.customValues.size() != expected) {
            std::println("Error: Expected {} custom values (A: {} + B: {}), but got {}", expected, sizeA, sizeB, config.customValues.size());
            return false;
        }

        config.matrixA.assign(config.customValues.begin(), config.customValues.begin() + sizeA);
        config.matrixB.assign(config.customValues.begin() + sizeA, config.customValues.end());
    }

    return true;
}

void Client_CLI::printUsage(const char* programName)
{
    std::println("Client");
    std::println("Usage: {} [OPTIONS] [MATRIX_DATA...]", programName);
    std::println("\nRequired:");
    std::println("  -r_a, --rows_a     Rows for Matrix A");
    std::println("  -c_a, --cols_a     Cols for Matrix A (Must match rows_b)");
    std::println("  -r_b, --rows_b     Rows for Matrix B");
    std::println("  -c_b, --cols_b     Cols for Matrix B");
    std::println("\nOptions:");
    std::println("  -s, --server       Server IP (default: 127.0.0.1)");
    std::println("  -p, --port         Server Port (default: 8080)");
    std::println("  -f, --forks        Number of processes (default: 4)");
    std::println("  --custom           Follow with integer values for A then B");
}