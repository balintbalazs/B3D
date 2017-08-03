#ifndef __B3D__PREDCPU_H__
#define __B3D__PREDCPU_H__


#include <cudaCompress/global.h>

#include <cstdint>


namespace cudaCompress {

	namespace util {
		CUCOMP_DLL void predictor7_tilesCPU(const int16_t* in, int16_t* out, int pitch, int width, int height, int tileSize);
		CUCOMP_DLL void unPredictor7_tilesCPU(const int16_t* in, int16_t* out, int pitch, int width, int height, int tileSize);

		CUCOMP_DLL void predictor7_tiles_wnllCPU(const float* in, float* buffer, int16_t* out, int pitch, int width, int height, int tileSize);
		CUCOMP_DLL void unPredictor7_tiles_wnllCPU(const int16_t* in, float* out, int pitch, int width, int height, int tileSize);

		CUCOMP_DLL void vstCPU(float* in, float* out, int num, float offset = 0.0, float conversion = 1.0, float sigma = 0.0);
		CUCOMP_DLL void invVstCPU(float* in, float* out, int num, float offset = 0.0, float conversion = 1.0, float sigma = 0.0);


		CUCOMP_DLL void s2fCPU(int16_t* in, float* out, int num);
		CUCOMP_DLL void f2sCPU(float* in, int16_t* out, int num);

		CUCOMP_DLL void u2fCPU(uint16_t* in, float* out, int num);
		CUCOMP_DLL void f2uCPU(float* in, uint16_t* out, int num);

		CUCOMP_DLL void multiplyCPU(float* in, float* out, float factor, int num);
		CUCOMP_DLL void multiplyCPU(int16_t* in, float* out, float factor, int num);

	}
}

#endif