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

void Client::sendMessenge(const char* message) {
    if (clientSocket < 0) 
        return;

    send(clientSocket, message, strlen(message), 0);
    receiveMessage();
}

void Client::receiveMessage() {
    char messageBuffer[MAX_MESSAGE_LENGTH];
    
    int bytesReceived = recv(clientSocket, messageBuffer, sizeof(messageBuffer) - 1, 0);
    if (bytesReceived > 0) {
        messageBuffer[bytesReceived] = '\0';
        std::println("{}", messageBuffer);
    }
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
    std::string message = formatMessage(config.rows, config.cols);
    client.sendMessenge(message.c_str());
    return 0;
}
