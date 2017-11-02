#include <cudaCompress/CPU/QuantizeCPU.h>

#include <cudaCompress/tools/Operator.h>
#include <cudaCompress/tools/Functor.h>

#include <cudaCompress/cudaUtil.h>
#include <cudaCompress/InstanceImpl.h>


inline int getNegativeSign(short val)
{
	return (val >> 15);
}
inline int getNegativeSign(int val)
{
	return (val >> 31);
}

namespace cudaCompress {

	namespace util {

		// convert signed shorts to symbols (>= 0 -> even, < 0 -> odd)
		void symbolizeCPU(ushort* pSymbols, const short* pData, uint sizeX, uint sizeY, uint sizeZ, uint rowPitchSrc, uint slicePitchSrc)
		{
			for (size_t i = 0; i < sizeX*sizeY*sizeZ; i++)
			{
				pSymbols[i] = 2 * abs(pData[i]) + getNegativeSign(pData[i]);
			}
			return;
		}

		void unsymbolizeCPU(short* pData, const ushort* pSymbols, uint sizeX, uint sizeY, uint sizeZ, uint rowPitchDst, uint slicePitchDst)
		{
			int negative = 0;
			for (size_t i = 0; i < sizeX*sizeY*sizeZ; i++)
			{
				negative = pSymbols[i] % 2;
				pData[i] = (1 - 2 * negative) * ((pSymbols[i] + negative) / 2);
			}
			return;
		}


		// convert unsigned shorts to symbols (reorder 1,N,2,N-1,3,N-2...)
		void symbolizeCPU(ushort* pSymbols, const ushort* pData, uint sizeX, uint sizeY, uint sizeZ, uint rowPitchSrc, uint slicePitchSrc)
		{
			for (size_t i = 0; i < sizeX*sizeY*sizeZ; i++)
			{
				pSymbols[i] = 2 * abs(pData[i]) + getNegativeSign(pData[i]);
			}
			return;
		}

		void unsymbolizeCPU(ushort* pData, const ushort* pSymbols, uint sizeX, uint sizeY, uint sizeZ, uint rowPitchDst, uint slicePitchDst)
		{
			int negative = 0;
			for (size_t i = 0; i < sizeX*sizeY*sizeZ; i++)
			{
				negative = pSymbols[i] % 2;
				pData[i] = (1 - 2 * negative) * ((pSymbols[i] + negative) / 2);
			}
			return;
		}


	}
}