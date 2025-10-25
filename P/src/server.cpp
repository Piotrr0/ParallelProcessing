#include <print>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "server.h"

Server::Server(int port) {
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket < 0)
        return;
        
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(port);
    serverAddress.sin_addr.s_addr = INADDR_ANY;
    if (bind(serverSocket, (struct sockaddr*)&serverAddress, sizeof(serverAddress)) < 0)
        return;
    }

Server::~Server() {
    if (connectedClientSocket >= 0)
        close(connectedClientSocket);
    if (serverSocket >= 0)
        close(serverSocket);
}

void Server::listenForConnection() {
    if (listen(serverSocket, maxPendingConnections) < 0)
        return;
    connectedClientSocket = accept(serverSocket, nullptr, nullptr);
    if (connectedClientSocket < 0)
        return;
    receiveMessage();
}

void Server::receiveMessage() {
    int bytesReceived = recv(connectedClientSocket, messageBuffer, sizeof(messageBuffer) - 1, 0);
    if (bytesReceived > 0) {
        messageBuffer[bytesReceived] = '\0';
        std::println("Message from client: {}", messageBuffer);
    }
}

int main() {
    Server server(8080);
    server.listenForConnection();
    return 0;
}
