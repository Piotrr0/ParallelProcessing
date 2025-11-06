cmake_minimum_required(VERSION 3.16)
project(LAB_4 LANGUAGES CXX)

find_package(OpenMP REQUIRED)
add_executable(LAB_4 src/main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(LAB_4 PUBLIC OpenMP::OpenMP_CXX)
endif()

set_property(TARGET LAB_4 PROPERTY CXX_STANDARD 17)
