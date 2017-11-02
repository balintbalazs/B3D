cmake_minimum_required(VERSION 2.8)
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
# CMake < 3.1 does not support above variables
if (CMAKE_VERSION VERSION_LESS "3.1")
	if ((CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
		set(CMAKE_CXX_FLAGS "-std=c++11")
	else()
		message(FATAL_ERROR "Cannot enable c++11, don't know how for this compilers use CMake > 3.1 or update CMakeLists.txt for your compiler.")
	endif()

endif()

# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -std=c++11
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
