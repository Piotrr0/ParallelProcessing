cmake_minimum_required(VERSION 3.10)
project(LAB_9 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(LAB_9 src/main.cpp)
target_link_libraries(LAB_9 PRIVATE MPI::MPI_CXX)