cmake_minimum_required(VERSION 3.16)
project(LAB_5 LANGUAGES CXX)

find_package(OpenMP REQUIRED)
add_executable(LAB_5 src/main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(LAB_5 PUBLIC OpenMP::OpenMP_CXX)
endif()

set_property(TARGET LAB_5 PROPERTY CXX_STANDARD 17)