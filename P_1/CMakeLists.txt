cmake_minimum_required(VERSION 3.10)

project(P VERSION 1.0)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)


add_executable(server src/server.cpp)
add_executable(client src/client.cpp src/client_cli.cpp)