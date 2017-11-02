cuda_add_library ( cudaCompress
	cudaCompress/CPU/YCoCgCPU.cpp
	cudaCompress/CPU/HuffmanCPU.cpp
	cudaCompress/CPU/EncodeCPU.cpp
	cudaCompress/CPU/ArithmeticCPU.cpp
	cudaCompress/CPU/HuffmanTableCPU.cpp
	cudaCompress/CPU/EncoderTestSuite.cpp
	cudaCompress/CPU/tools/stb_image.c
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


cuda_add_executable ( B3D_examples
  examples/direct_example+benchmark.cpp
  examples/tools/rawfile.cpp
  examples/tthread/tinythread.cpp
	)

target_link_libraries ( B3D_examples cudaCompress)


if (BUILD_SHARED_LIBS)
	target_compile_definitions(cudaCompress PUBLIC CUCOMP_BUILD_DLL)
else()
	target_compile_definitions(cudaCompress PUBLIC CUCOMP_BUILD_STANDALONE)
	target_compile_definitions(cudaCompressTest PUBLIC CUCOMP_BUILD_STANDALONE)
	target_compile_definitions(B3D_examples PUBLIC CUCOMP_BUILD_STANDALONE)
endif()
