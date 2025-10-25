#ifndef CLIENT_H
#define CLIENT_H

#include <netinet/in.h>

#define MAX_MESSAGE_LENGTH 1024

class Client {
public:
    int clientSocket = -1;
    sockaddr_in serverAddress{};

    Client(int serverPort, const char* serverIP);
    ~Client();

    void sendMessenge(const char* message);
    void receiveMessage();
};

#endif