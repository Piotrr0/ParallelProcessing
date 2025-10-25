#ifndef SERVER_H
#define SERVER_H

#include <netinet/in.h>

#define MAX_MESSAGE_LENGTH 1024

class Server {
public:
    const int maxPendingConnections = 5;

    int serverSocket = -1;
    int connectedClientSocket = -1;

    sockaddr_in serverAddress{};
    char messageBuffer[MAX_MESSAGE_LENGTH];

    Server(int port);
    ~Server();

    void listenForConnection();
    void receiveMessage();
};

#endif