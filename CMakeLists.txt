cmake_minimum_required(VERSION 2.8.12)
project(smeint)

set (CMAKE_CXX_STANDARD 14)

include_directories(include)

add_subdirectory(pybind11)
pybind11_add_module(smeint smeint.cpp)
