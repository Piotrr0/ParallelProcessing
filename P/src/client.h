#ifndef CLIENT_H
#define CLIENT_H

#include <netinet/in.h>

class Client {
public:
    int clientSocket = -1;
    sockaddr_in serverAddress{};

    Client(int serverPort, const char* serverIP);
    ~Client();

    void sendMessenge(const char* message);
};

#endif