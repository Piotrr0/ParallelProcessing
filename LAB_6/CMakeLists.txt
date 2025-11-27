cmake_minimum_required(VERSION 3.18)
project(LAB_6 LANGUAGES CXX CUDA)

add_executable(LAB_6 src/main.cu)
set_target_properties(LAB_6 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)