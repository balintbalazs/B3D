#ifndef __B3D__PRED_H__
#define __B3D__PRED_H__


#include <cudaCompress/global.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdint>


namespace cudaCompress {

	namespace util {

		CUCOMP_DLL void predictor1(const int16_t* in, int16_t* out, int pitch, int width, int height);
		CUCOMP_DLL void unPredictor1(const int16_t* in, int16_t* out, int pitch, int width, int height);

		CUCOMP_DLL void predictor1_tiles(const int16_t* in, int16_t* out, int pitch, int width, int height, int tileSize);
		CUCOMP_DLL void unPredictor1_tiles(const int16_t* in, int16_t* out, int pitch, int width, int height, int tileSize);

		CUCOMP_DLL void predictor2(const int16_t* in, int16_t* out, int pitch, int width, int height);
		CUCOMP_DLL void unPredictor2(const int16_t* in, int16_t* out, int pitch, int width, int height);

		CUCOMP_DLL void predictor2_tiles(const int16_t* in, int16_t* out, int pitch, int width, int height, int tileSize);
		CUCOMP_DLL void unPredictor2_tiles(const int16_t* in, int16_t* out, int pitch, int width, int height, int tileSize);

		CUCOMP_DLL void predictor4(const int16_t* in, int16_t* out, int pitch, int width, int height);
		CUCOMP_DLL void unPredictor4(const int16_t* in, int16_t* out, int pitch, int width, int height);

		CUCOMP_DLL void predictor7(const int16_t* in, int16_t* out, int pitch, int width, int height);
		CUCOMP_DLL void unPredictor7(const int16_t* in, int16_t* out, int pitch, int width, int height);

		CUCOMP_DLL void predictor8(const int16_t* in, int16_t* out, int pitch, int width, int height);
		CUCOMP_DLL void unPredictor8(const int16_t* in, int16_t* out, int pitch, int width, int height);

		CUCOMP_DLL void predictor7_tiles(const int16_t* in, int16_t* out, int pitch, int width, int height, int tileSize);
		CUCOMP_DLL void unPredictor7_tiles(const int16_t* in, int16_t* out, int pitch, int width, int height, int tileSize);
		
		CUCOMP_DLL void predictor7_tiles_nll(const int16_t* in, int16_t* buffer, int16_t* out, int pitch, int width, int height, int num_tiles, int16_t delta);
		CUCOMP_DLL void unPredictor7_tiles_nll(const int16_t* in, int16_t* out, int pitch, int width, int height, int num_tiles, int16_t delta);
		
		CUCOMP_DLL void predictor7_tiles_wnll(const float* in, float* buffer, int16_t* out, int pitch, int width, int height, int tileSize);
		CUCOMP_DLL void predictor7_tiles_wnll2(const float* in, float* buffer, int16_t* out, int pitch, int width, int height, int tileSize);
		CUCOMP_DLL void unPredictor7_tiles_wnll(const int16_t* in, float* out, int pitch, int width, int height, int tileSize);

		CUCOMP_DLL void predictor1_tiles_wnll(const float* in, float* buffer, int16_t* out, int pitch, int width, int height, int num_tiles);
		CUCOMP_DLL void unPredictor1_tiles_wnll(const int16_t* in, float* out, int pitch, int width, int height, int num_tiles);

		CUCOMP_DLL void predictorMed_tiles(const int16_t* in, int16_t* out, int pitch, int width, int height);
		CUCOMP_DLL void unPredictorMed_tiles(const int16_t* in, int16_t* out, int pitch, int width, int height);

		CUCOMP_DLL void predictor3D7(const int16_t* in, int16_t* out, int pitch, int slicePitch, int width, int height, int depth);

		CUCOMP_DLL void quantize(const float* in, int16_t* out, int pitch, int width, int height);
		CUCOMP_DLL void unQuantize(const int16_t* in, float* out, int pitch, int width, int height);

		//CUCOMP_DLL void runQuantize(const float* in, float* out, int pitch, int width, int height, float delta);
		//CUCOMP_DLL void unQuantize(const float* in, float* out, int pitch, int width, int height, float delta);

		CUCOMP_DLL void Anscombe(float* in, float* out, int num, float sigma = 0.0);
		CUCOMP_DLL void invAnscombe(float* in, float* out, int num, float sigma = 0.0);

		CUCOMP_DLL void vst(float* in, float* out, int num, float offset=0.0, float conversion=1.0, float sigma=0.0);
		CUCOMP_DLL void invVst(float* in, float* out, int num, float offset=0.0, float conversion=1.0, float sigma=0.0);

		CUCOMP_DLL void sqrtArray(float* in, float* out, int num);
		CUCOMP_DLL void sqrArray(float* in, float* out, int num);

		CUCOMP_DLL void sqrtArray(int16_t* in, float* out, int num);
		CUCOMP_DLL void sqrArray(float* in, int16_t* out, int num);

		CUCOMP_DLL void s2f(int16_t* in, float* out, int num);
		CUCOMP_DLL void f2s(float* in, int16_t* out, int num);

		CUCOMP_DLL void u2f(uint16_t* in, float* out, int num);
		CUCOMP_DLL void f2u(float* in, uint16_t* out, int num);

		CUCOMP_DLL void u8tou16(uint8_t* in, uint16_t* out, int num);
		CUCOMP_DLL void u16tou8(uint16_t* in, uint8_t* out, int num);

		CUCOMP_DLL void multiply(float* in, float* out, float factor, int num);
		CUCOMP_DLL void multiply(int16_t* in, float* out, float factor, int num);

		CUCOMP_DLL void offset(float* in, float* out, float offset, int num);
		CUCOMP_DLL void offsetAbs(float* in, float* out, uint16_t* signs, float level, int num);
		CUCOMP_DLL void applySignOffset(float* in, float* out, uint16_t* signs, float level, int num);
		CUCOMP_DLL void offsetSeparate(float* in, float* out, float* negatives, float level, int num);
		CUCOMP_DLL void mergeOffset(float* in, float* out, float* negatives, float level, int num);

	}

}


#endif