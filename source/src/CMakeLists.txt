add_library ( cudaCompress STATIC
	cudaCompress/CPU/YCoCgCPU.cpp
	cudaCompress/CPU/HuffmanCPU.cpp
	cudaCompress/CPU/EncodeCPU.cpp
	cudaCompress/CPU/ArithmeticCPU.cpp
	cudaCompress/CPU/HuffmanTableCPU.cpp
	cudaCompress/CPU/RBUCCPU.cpp
	cudaCompress/CPU/GolombRiceCPU.cpp
	cudaCompress/CPU/RunLengthCPU.cpp
	cudaCompress/CPU/HistogramCPU.cpp
	cudaCompress/CPU/PredictorsCPU.cpp
	cudaCompress/CPU/QuantizeCPU.cpp
	cudaCompress/RunLength.cu
	cudaCompress/Huffman.cu
	cudaCompress/Timing.cpp
	cudaCompress/reduce/reduce_plan.cpp
	cudaCompress/Histogram.cu
	cudaCompress/HuffmanTable.cu
	cudaCompress/profiler/profilerstreamwriter.cpp
	cudaCompress/profiler/profilerlogwriter.cpp
	cudaCompress/profiler/profiler.cpp
	cudaCompress/InstanceImpl.cpp
	cudaCompress/PackInc.cu
	cudaCompress/Instance.cpp
	cudaCompress/scan/scan_plan.cpp
	cudaCompress/Encode.cpp
	cudaCompress/util/Bits.cu
	cudaCompress/util/DWTInt.cu
	cudaCompress/util/YCoCg.cu
	cudaCompress/util/Quantize.cu
	cudaCompress/util/CudaTimer.cpp
  cudaCompress/util/DWTFloat.cu
  cudaCompress/util/Predictors.cu
  cudaCompress/B3D/B3DcompressFunctions.cpp
)

target_compile_features(cudaCompress PUBLIC cxx_std_11)

set_target_properties(cudaCompress
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable ( B3D_examples
  examples/direct_example+benchmark.cu
  examples/tools/rawfile.cpp
  examples/tthread/tinythread.cpp
	)

set_target_properties(B3D_examples PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries (B3D_examples PRIVATE cudaCompress)

if(APPLE)
	# We need to add the default path to the driver (libcuda.dylib) as an rpath,
	# so that the static cuda runtime can find it at runtime.
	set_property(TARGET particle_test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()


if (BUILD_SHARED_LIBS)
	target_compile_definitions(cudaCompress PUBLIC CUCOMP_BUILD_DLL)
else()
	target_compile_definitions(cudaCompress PUBLIC CUCOMP_BUILD_STANDALONE)
	target_compile_definitions(cudaCompressTest PUBLIC CUCOMP_BUILD_STANDALONE)
	target_compile_definitions(B3D_examples PUBLIC CUCOMP_BUILD_STANDALONE)
endif()
