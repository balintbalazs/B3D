# Minimum CMake version requirement
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
cmake_policy(VERSION 3.9)

# # For MSVC and FindThreads, respectively.
# enable_language(CXX C)

project(B3D
        VERSION 1.0.0
        LANGUAGES CXX CUDA)

message(STATUS "Configuring B3D compression library, version ${B3D_VERSION}")

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}")

option(BUILD_SHARED_LIBS "Build shared libararies" ON)

# We want c++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Pass options to NVCC
set(CMAKE_CUDA_FLAGS "-gencode arch=compute_30,code=sm_30 \
                      -gencode arch=compute_50,code=sm_50 \
                      -gencode arch=compute_52,code=sm_52 \
                      -gencode arch=compute_60,code=sm_60")

# Specify include directories
include_directories(
  include
  src
  ${CUDA_INCLUDE_DIRS}sr
)

add_subdirectory( src )
