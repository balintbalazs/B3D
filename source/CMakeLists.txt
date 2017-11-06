# Minimum CMake version requirement
cmake_minimum_required(VERSION 3.2.0)
cmake_policy(VERSION 3.2.0)

# # For MSVC and FindThreads, respectively.
# enable_language(CXX C)

project(B3D)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}")

option(BUILD_SHARED_LIBS "Build shared libararies" ON)

# We want c++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Enable C99 support universally
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED 99)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3
	-gencode arch=compute_30,code=sm_30
	-gencode arch=compute_50,code=sm_50
	-gencode arch=compute_52,code=sm_52
	-gencode arch=compute_60,code=sm_60)

# Specify include directories
include_directories(
	include
	src
	)

add_subdirectory( src )
