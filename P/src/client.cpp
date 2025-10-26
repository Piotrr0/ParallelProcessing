#include <string>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <print>
#include "client.h"
#include "client_cli.h"

Client::Client(int serverPort, const char* serverIP) {
    clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket < 0)
        return;

    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(serverPort);
    inet_pton(AF_INET, serverIP, &serverAddress.sin_addr);
    
    if (connect(clientSocket, (struct sockaddr*)&serverAddress, sizeof(serverAddress)) < 0)
        return;
}

Client::~Client() {
    if (clientSocket >= 0)
        close(clientSocket);
}

bool Client::sendMessenge(const char* message) {
    if (clientSocket < 0) 
        return false;

    send(clientSocket, message, strlen(message), 0);
    return true;
}

bool Client::receiveMessage() {
    char messageBuffer[MAX_MESSAGE_LENGTH];
    
    int bytesReceived = recv(clientSocket, messageBuffer, sizeof(messageBuffer) - 1, 0);
    if (bytesReceived <= 0) {
        return false;
    }

    std::println("{}", messageBuffer);
    return true;
}

int main(int argc, char* argv[]) {
    
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
    std::string message = formatMessage(config.rows, config.cols, config.forks);
    client.sendMessenge(message.c_str());
    client.receiveMessage();
    return 0;
}
