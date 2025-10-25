#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include "client.h"

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
}

int main() {
    Client client(8080, "127.0.0.1");
    client.sendMessenge("Hello!");

    return 0;
}
