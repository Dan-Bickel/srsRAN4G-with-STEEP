cmake_minimum_required(VERSION 3.10)
project(steep_eve CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(steep_eve steep_eve.cpp)
target_compile_options(steep_eve PRIVATE -Wall -Wextra -O2)
target_link_libraries(steep_eve m)

message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "Binary     : steep_eve")