#ifndef __B3D__QUANTIZECPU_H__
#define __B3D__QUANTIZECPU_H__


#include <cudaCompress/global.h>

#include <cuda_runtime.h>


namespace cudaCompress {

	class Instance;

	namespace util {

		/*enum EQuantizeType
		{
			QUANTIZE_DEADZONE = 0, // midtread quantizer with twice larger zero bin
			QUANTIZE_UNIFORM,      // standard uniform midtread quantizer
			QUANTIZE_COUNT
		};*/
		
		// convert signed shorts to symbols (>= 0 -> even, < 0 -> odd)
		CUCOMP_DLL void symbolizeCPU(ushort* pSymbols, const short* pData, uint sizeX, uint sizeY, uint sizeZ, uint rowPitchSrc = 0, uint slicePitchSrc = 0);
		CUCOMP_DLL void unsymbolizeCPU(short* pData, const ushort* pSymbols, uint sizeX, uint sizeY, uint sizeZ, uint rowPitchDst = 0, uint slicePitchDst = 0);

		// convert unsigned shorts to symbols (reorder 1,N,2,N-1,3,N-2...)
		CUCOMP_DLL void symbolizeCPU(ushort* pSymbols, const ushort* pData, uint sizeX, uint sizeY, uint sizeZ, uint rowPitchSrc = 0, uint slicePitchSrc = 0);
		CUCOMP_DLL void unsymbolizeCPU(ushort* pData, const ushort* pSymbols, uint sizeX, uint sizeY, uint sizeZ, uint rowPitchDst = 0, uint slicePitchDst = 0);

	}

}


#endif
