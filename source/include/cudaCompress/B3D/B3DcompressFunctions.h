#ifndef __B3D__COMPR_FUNC_H__
#define __B3D__COMPR_FUNC_H__


#include <cuda_runtime.h>

#include <cudaCompress/Encode.h>
#include <cudaCompress/util/DWT.h>
#include <cudaCompress/util/Quantize.h>
#include <cudaCompress/util/Predictors.h>
#include <cudaCompress/cudaUtil.h>

#include <cudaCompress/CPU/EncodeCPU.h>
#include <cudaCompress/CPU/QuantizeCPU.h>
#include <cudaCompress/CPU/PredictorsCPU.h>

namespace cudaCompress {

	//namespace util {
		CUCOMP_DLL void compressImageLL(
			Instance* pInstance,
			std::vector<uint>& i_bitStream, // bitStream for compressed data
			int16_t* dpImage,  // input image in GPU memory
			int16_t* dpBuffer,
			int16_t* dpScratch,
			uint16_t* dpSymbols,
			int sizeX, int sizeY, int sizeZ,         // image size
			size_t dwtLevel, int tileSize);               // quantization step
		
		CUCOMP_DLL void decompressImageLL(
			Instance* pInstance,
			std::vector<uint>& i_bitStream, // bitStream for compressed data
			int16_t* dpImage,  // input image in GPU memory
			int16_t* dpBuffer,
			int16_t* dpScratch,
			uint16_t* dpSymbols,
			int sizeX, int sizeY, int sizeZ,         // image size
			size_t dwtLevel, int tileSize);

		CUCOMP_DLL void compressImage(
			Instance* pInstance,
			std::vector<uint>& i_bitStream, // bitStream for compressed data
			int16_t* dpImage,  // input image in GPU memory
			float* dpBuffer,
			float* dpScratch,
			uint16_t* dpSymbols,
			int sizeX, int sizeY, int sizeZ,         // image size
			size_t dwtLevel, float quantStep, float bgLevel, int tileSize, float conversion = 1., float readNoise = 0.);               // quantization step
		
		CUCOMP_DLL void decompressImage(
			Instance* pInstance,
			std::vector<uint>& i_bitStream, // bitStream for compressed data
			int16_t* dpImage,  // input image in GPU memory
			float* dpBuffer,
			float* dpScratch,
			uint16_t* dpSymbols,
			int sizeX, int sizeY, int sizeZ,         // image size
			size_t dwtLevel, float quantStep, float bgLevel, int tileSize, float conversion = 1., float readNoise = 0.);


		CUCOMP_DLL void compressImageLLCPU(
			//Instance* pInstance,
			std::vector<uint>& i_bitStream, // bitStream for compressed data
			int16_t* pImage,  // input image in GPU memory
			int16_t* pBuffer,
			int16_t* pScratch,
			uint16_t* pSymbols,
			int sizeX, int sizeY, int sizeZ,         // image size
			size_t dwtLevel, int tileSize);               // quantization step

		CUCOMP_DLL void decompressImageLLCPU(
			//Instance* pInstance,
			std::vector<uint>& i_bitStream, // bitStream for compressed data
			int16_t* pImage,  // input image in GPU memory
			int16_t* pBuffer,
			int16_t* pScratch,
			uint16_t* pSymbols,
			int sizeX, int sizeY, int sizeZ,         // image size
			size_t dwtLevel, int tileSize);

		CUCOMP_DLL void compressImageCPU(
			//Instance* pInstance,
			std::vector<uint>& i_bitStream, // bitStream for compressed data
			int16_t* pImage,  // input image in GPU memory
			float* pBuffer,
			float* pScratch,
			uint16_t* pSymbols,
			int sizeX, int sizeY, int sizeZ,         // image size
			size_t dwtLevel, float quantStep, float bgLevel, int tileSize, float conversion = 1., float readNoise = 0.);               // quantization step

		CUCOMP_DLL void decompressImageCPU(
			//Instance* pInstance,
			std::vector<uint>& i_bitStream, // bitStream for compressed data
			int16_t* pImage,  // input image in GPU memory
			float* pBuffer,
			float* pScratch,
			uint16_t* pSymbols,
			int sizeX, int sizeY, int sizeZ,         // image size
			size_t dwtLevel, float quantStep, float bgLevel, int tileSize, float conversion = 1., float readNoise = 0.);
	//}
}

#endif
