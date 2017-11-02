#include <cudaCompress/B3D/B3DcompressFunctions.h>

namespace cudaCompress {

	//namespace util {

		void compressImageLL(
			Instance* pInstance,
			std::vector<uint>& i_bitStream, // bitStream for compressed data
			int16_t* dpImage,  // input image in GPU memory
			int16_t* dpBuffer,
			int16_t* dpScratch,
			uint16_t* dpSymbols,
			int sizeX, int sizeY, int sizeZ,         // image size
			size_t dwtLevel, int tileSize)               // quantization step
		{
			sizeY = sizeY * sizeZ;
			// Do multi-level DWT in the same buffers. Need to specify pitch now!
			if (dwtLevel > 0 && dwtLevel < 100) {
				cudaMemcpy(dpBuffer, dpImage, sizeX*sizeY * sizeof(int16_t), cudaMemcpyDeviceToDevice);
				switch (dwtLevel) {
					/*case 1:
					cudaCompress::util::predictor1(dpImage, dpBuffer, sizeX * sizeof(int16_t), sizeX, sizeY);
					break;
					case 2:
					cudaCompress::util::predictor2(dpImage, dpBuffer, sizeX * sizeof(int16_t), sizeX, sizeY);
					break;
					case 4:
					cudaCompress::util::predictor4(dpImage, dpBuffer, sizeX * sizeof(int16_t), sizeX, sizeY);
					break;*/
				case 1:
				case 2:
				case 7:
					cudaCompress::util::predictor7_tiles(dpImage, dpBuffer, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);
					break;
				case 9:
					cudaCompress::util::predictorMed_tiles(dpImage, dpBuffer, sizeX * sizeof(int16_t), sizeX, sizeY);
					break;
				default:
					break;
				}
				cudaCheckMsg("predictor failed");
			}
			else {
				dwtLevel = dwtLevel - 100;
				cudaCompress::util::dwtIntForward(
					dpBuffer, dpScratch, dpImage, sizeX, sizeY, 1, sizeX, 0, sizeX, 0);
				for (int i = 1; i < dwtLevel; i++)
				{
					cudaCompress::util::dwtIntForward(
						dpBuffer, dpScratch, dpBuffer, sizeX / pow(2.0, i), sizeY / pow(2.0, i), 1, sizeX, 0, sizeX, 0);
				}				
			}

			// dpBuffer now contains the multi-level DWT decomposition.

			// Quantize the coefficients and convert them to unsigned values (symbols).
			// For better compression, quantStep should be adapted to the transform level!
			cudaCompress::util::symbolize(dpSymbols, dpBuffer, sizeX, sizeY, 1);
			//cudaMemcpy(dpSymbols, dpBuffer, sizeX * sizeY * sizeof(int16_t), cudaMemcpyDeviceToDevice);

			// Run-length + Huffman encode the quantized coefficients.
			cudaCompress::BitStream bitStream(&i_bitStream);
			cudaCompress::encodeRLHuff(pInstance, bitStream, &dpSymbols, 1, sizeX * sizeY);
			//return bitStream;
		}

		// dpSymbols has to be initialized to 0
		void decompressImageLL(
			Instance* pInstance,
			std::vector<uint>& i_bitStream, // bitStream for compressed data
			int16_t* dpImage,  // input image in GPU memory
			int16_t* dpBuffer,
			int16_t* dpScratch,
			uint16_t* dpSymbols,
			int sizeX, int sizeY, int sizeZ,         // image size
			size_t dwtLevel, int tileSize)
		{
			sizeY = sizeY * sizeZ;
			BitStreamReadOnly bitStream(i_bitStream.data(), uint(i_bitStream.size() * sizeof(uint) * 8));
			cudaCompress::decodeRLHuff(pInstance, bitStream, &dpSymbols, 1, sizeX * sizeY);

			cudaCompress::util::unsymbolize(dpBuffer, dpSymbols, sizeX, sizeY, 1);
			//cudaMemcpy(dpBuffer, dpSymbols, sizeX * sizeY * sizeof(int16_t), cudaMemcpyDeviceToDevice);

			if (dwtLevel > 0 && dwtLevel < 100) {
				cudaMemcpy(dpImage, dpBuffer, sizeX*sizeY * sizeof(int16_t), cudaMemcpyDeviceToDevice);
				switch (dwtLevel) {
					/*case 1:
					cudaCompress::util::unPredictor1(dpBuffer, dpImage, sizeX * sizeof(int16_t), sizeX, sizeY);
					break;
					case 2:
					cudaCompress::util::unPredictor2(dpBuffer, dpImage, sizeX * sizeof(int16_t), sizeX, sizeY);
					break;
					case 4:
					cudaCompress::util::unPredictor4(dpBuffer, dpImage, sizeX * sizeof(int16_t), sizeX, sizeY);
					break;*/
				case 1:
				case 2:
				case 7:
					cudaCompress::util::unPredictor7_tiles(dpBuffer, dpImage, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);
					break;
				case 9:
					cudaCompress::util::unPredictorMed_tiles(dpBuffer, dpImage, sizeX * sizeof(int16_t), sizeX, sizeY);
					break;
				default:
					break;
				}
				cudaCheckMsg("unpredictor failed");
			}
			else {
				dwtLevel = dwtLevel - 100;
				for (int i = dwtLevel - 1; i > 0; i--)
				{
					cudaCompress::util::dwtIntInverse(
						dpBuffer, dpScratch, dpBuffer, sizeX / pow(2.0, i), sizeY / pow(2.0, i), 1, sizeX, 0, sizeX, 0);
				}
				cudaCompress::util::dwtIntInverse(
					dpImage, dpScratch, dpBuffer, sizeX, sizeY, 1, sizeX, 0, sizeX, 0); 
				
			}
		}

		void compressImage(
			Instance* pInstance,
			std::vector<uint>& i_bitStream, // bitStream for compressed data
			int16_t* dpImage,  // input image in GPU memory
			float* dpBuffer,
			float* dpScratch,
			uint16_t* dpSymbols,
			int sizeX, int sizeY, int sizeZ,         // image size
			size_t dwtLevel, float quantStep, float bgLevel, int tileSize, float conversion, float readNoise)               // quantization step
		{
			sizeY = sizeY * sizeZ;
			sizeZ = 1;
			if (dwtLevel > 0 && dwtLevel < 100) {
				uint16_t* pdpSymbols[1] = { dpSymbols };
				switch (dwtLevel) {
					/*case 7:
					cudaMemcpy(dpBuffer, dpImage, sizeX*sizeY*sizeZ*sizeof(int16_t), cudaMemcpyDeviceToDevice);
					cudaMemcpy(dpScratch, dpImage, sizeX*sizeY*sizeZ*sizeof(int16_t), cudaMemcpyDeviceToDevice);
					cudaCompress::util::predictor7_tiles_nll(dpImage, (int16_t*)dpScratch, (int16_t*)dpBuffer, sizeX * sizeof(int16_t), sizeX, sizeY, num_tiles, quantStep);
					cudaCompress::util::symbolize(dpSymbols, (int16_t*)dpBuffer, sizeX, sizeY, sizeZ);
					break;*/
				case 1: // first version, square root /w readnoise + prediction7 + quantization within noise level
				case 3: // different offset in decompression to test bias
				case 11: // cpu decompression
					cudaCompress::util::u2f((uint16_t*)dpImage, dpBuffer, sizeX * sizeY);
					// variance stabilization
					cudaCompress::util::vst(dpBuffer, dpBuffer, sizeX * sizeY, bgLevel, conversion, readNoise);
					// scale with quantization step
					cudaCompress::util::multiply(dpBuffer, dpBuffer, 1 / quantStep, sizeX * sizeY);
					// run prediction + quantization
					cudaCompress::util::predictor7_tiles_wnll(dpBuffer, dpScratch, dpImage, sizeX, sizeX, sizeY, tileSize);
					cudaCompress::util::symbolize(dpSymbols, dpImage, sizeX, sizeY, sizeZ);
					break;
				case 2: // swapped: square root /w readnoise + quantization + prediction7
					cudaCompress::util::u2f((uint16_t*)dpImage, dpBuffer, sizeX * sizeY);
					// variance stabilization
					cudaCompress::util::vst(dpBuffer, dpBuffer, sizeX * sizeY, bgLevel, conversion, readNoise);
					// scale with quantization step
					cudaCompress::util::multiply(dpBuffer, dpBuffer, 1 / quantStep, sizeX * sizeY);
					// run  quantization first then prediction
					cudaCompress::util::f2u(dpBuffer, (uint16_t*)dpScratch, sizeX * sizeY);
					//cudaMemcpy(dpImage, dpScratch, sizeX*sizeY * sizeof(int16_t), cudaMemcpyDeviceToDevice);
					cudaCompress::util::predictor7_tiles((int16_t*)dpScratch, dpImage, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);
					cudaCompress::util::symbolize(dpSymbols, dpImage, sizeX, sizeY, sizeZ);
					break;
				case 7: // first version, square root + prediction7 + quantization within noise level
					cudaCompress::util::u2f((uint16_t*)dpImage, dpBuffer, sizeX * sizeY);
					cudaCompress::util::offset(dpBuffer, dpBuffer, -bgLevel, sizeX * sizeY);
					cudaCompress::util::multiply(dpBuffer, dpBuffer, 1 / conversion, sizeX * sizeY);
					cudaCompress::util::sqrtArray(dpBuffer, dpBuffer, sizeX * sizeY);
					//upscale for more precision
					cudaCompress::util::multiply(dpBuffer, dpBuffer, 1 / quantStep, sizeX * sizeY);
					// run prediction + quantization
					cudaCompress::util::predictor7_tiles_wnll(dpBuffer, dpScratch, dpImage, sizeX, sizeX, sizeY, tileSize);
					cudaCompress::util::symbolize(dpSymbols, dpImage, sizeX, sizeY, sizeZ);
					break;
				case 17: // swapped: square root + quantization + prediction7
					cudaCompress::util::u2f((uint16_t*)dpImage, dpBuffer, sizeX * sizeY);
					cudaCompress::util::offset(dpBuffer, dpBuffer, -bgLevel, sizeX * sizeY);
					cudaCompress::util::sqrtArray(dpBuffer, dpBuffer, sizeX * sizeY);
					//upscale for more precision
					cudaCompress::util::multiply(dpBuffer, dpBuffer, 1 / quantStep, sizeX * sizeY);
					// run  quantization first then prediction
					cudaCompress::util::f2u(dpBuffer, (uint16_t*)dpScratch, sizeX * sizeY);
					//cudaMemcpy(dpImage, dpScratch, sizeX*sizeY * sizeof(int16_t), cudaMemcpyDeviceToDevice);
					cudaCompress::util::predictor7_tiles((int16_t*)dpScratch, dpImage, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);
					cudaCompress::util::symbolize(dpSymbols, dpImage, sizeX, sizeY, sizeZ);
					break;
				case 27: // prediction7 + quantization + dithering
					cudaCompress::util::u2f((uint16_t*)dpImage, dpBuffer, sizeX * sizeY);
					cudaCompress::util::offset(dpBuffer, dpBuffer, -bgLevel, sizeX * sizeY);
					cudaCompress::util::sqrtArray(dpBuffer, dpBuffer, sizeX * sizeY);
					//upscale for more precision
					cudaCompress::util::multiply(dpBuffer, dpBuffer, 1 / quantStep, sizeX * sizeY);
					// run prediction + quantization
					cudaCompress::util::predictor7_tiles_wnll2(dpBuffer, dpScratch, dpImage, sizeX, sizeX, sizeY, tileSize);
					cudaCompress::util::symbolize(dpSymbols, dpImage, sizeX, sizeY, sizeZ);
					break;
				case 37: // proper Anscombe + prediction + quantization
					cudaCompress::util::u2f((uint16_t*)dpImage, dpBuffer, sizeX * sizeY);
					// scale back to photons
					cudaCompress::util::offset(dpBuffer, dpBuffer, -bgLevel, sizeX * sizeY);
					cudaCompress::util::multiply(dpBuffer, dpBuffer, 1 / conversion, sizeX * sizeY);
					// Anscombe transform
					cudaCompress::util::Anscombe(dpBuffer, dpBuffer, sizeX * sizeY);
					//scale by quant step
					cudaCompress::util::multiply(dpBuffer, dpBuffer, 1 / (2 * quantStep), sizeX * sizeY);
					// run prediction + quantization
					cudaCompress::util::predictor7_tiles_wnll(dpBuffer, dpScratch, dpImage, sizeX, sizeX, sizeY, tileSize);
					cudaCompress::util::symbolize(dpSymbols, dpImage, sizeX, sizeY, sizeZ);
					break;
				default:
					break;
				}
				cudaCheckMsg("predictor failed");
				cudaCompress::BitStream bitStream(&i_bitStream);
				cudaCompress::encodeRLHuff(pInstance, bitStream, pdpSymbols, 1, sizeX * sizeY);
			}
			else {
				dwtLevel = dwtLevel - 100;
				cudaCompress::util::dwtFloat2DForwardFromUshort(
					dpBuffer, dpScratch, (uint16_t*)dpImage, sizeX, sizeY, 1, sizeX, sizeX, 0);
				for (int i = 1; i < dwtLevel; i++)
				{
					cudaCompress::util::dwtFloat2DForward(
						dpBuffer, dpScratch, dpBuffer, sizeX / pow(2.0, i), sizeY / pow(2.0, i), 1, sizeX, sizeX, 0);
				}
				cudaCompress::util::quantizeToSymbols2D(dpSymbols, dpBuffer, sizeX, sizeY, quantStep);
				cudaCompress::BitStream bitStream(&i_bitStream);
				cudaCompress::encodeRLHuff(pInstance, bitStream, &dpSymbols, 1, sizeX * sizeY);
				//uint16_t* dpSymbolsN = dpSymbols + sizeX * sizeY;				
			}
		}

		void decompressImage(
			Instance* pInstance,
			std::vector<uint>& i_bitStream, // bitStream for compressed data
			int16_t* dpImage,  // input image in GPU memory
			float* dpBuffer,
			float* dpScratch,
			uint16_t* dpSymbols,
			int sizeX, int sizeY, int sizeZ,         // image size
			size_t dwtLevel, float quantStep, float bgLevel, int tileSize, float conversion, float readNoise)
		{
			sizeY = sizeY * sizeZ;
			sizeZ = 1;
			if (dwtLevel > 0 && dwtLevel < 100) {
				BitStreamReadOnly bitStream(i_bitStream.data(), uint(i_bitStream.size() * sizeof(uint) * 8));
				//uint16_t* dpSymbolsN = dpSymbols + sizeX * sizeY;
				uint16_t* pdpSymbols[1] = { dpSymbols };
				switch (dwtLevel) {
					/*case 7:
					cudaCompress::util::unsymbolize((int16_t*)dpBuffer, dpSymbols, sizeX, sizeY, sizeZ);
					cudaMemcpy(dpImage, dpBuffer, sizeX*sizeY*sizeZ*sizeof(int16_t), cudaMemcpyDeviceToDevice);
					cudaCompress::util::unPredictor7_tiles_nll((int16_t*)dpBuffer, dpImage, sizeX * sizeof(int16_t), sizeX, sizeY, num_tiles, quantStep);
					break;*/
				case 1:
				case 11:
					cudaCompress::decodeRLHuff(pInstance, bitStream, pdpSymbols, 1, sizeX * sizeY);
					cudaCompress::util::unsymbolize(dpImage, dpSymbols, sizeX, sizeY, sizeZ);
					//cudaCompress::util::unQuantize(dpImage, dpBuffer, sizeX, sizeX, sizeY);
					cudaCompress::util::unPredictor7_tiles_wnll(dpImage, dpBuffer, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);
					// undo scaling from quantization
					cudaCompress::util::multiply(dpBuffer, dpBuffer, quantStep, sizeX * sizeY);
					// inverse variance stabilization
					cudaCompress::util::invVst(dpBuffer, dpBuffer, sizeX * sizeY, bgLevel, conversion, readNoise);
					//back to int16_t from float
					cudaCompress::util::f2u(dpBuffer, (uint16_t*)dpImage, sizeX*sizeY);
					break;
					/*case 11:
					cudaCompress::decodeRLHuffCPU(pInstance, bitStream, pdpSymbols, 1, sizeX * sizeY);
					cudaCompress::util::unsymbolizeCPU(dpImage, dpSymbols, sizeX, sizeY, sizeZ);
					//cudaCompress::util::unQuantize(dpImage, dpBuffer, sizeX, sizeX, sizeY);
					cudaCompress::util::unPredictor7_tiles_wnll(dpImage, dpBuffer, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);
					// undo scaling from quantization
					cudaCompress::util::multiply(dpBuffer, dpBuffer, quantStep, sizeX * sizeY);
					// inverse variance stabilization
					cudaCompress::util::invVst(dpBuffer, dpBuffer, sizeX * sizeY, bgLevel, conversion, readNoise);
					//back to int16_t from float
					cudaCompress::util::f2u(dpBuffer, (uint16_t*)dpImage, sizeX*sizeY);
					break;*/
				case 2:
					cudaCompress::decodeRLHuff(pInstance, bitStream, pdpSymbols, 1, sizeX * sizeY);
					cudaCompress::util::unsymbolize(dpImage, dpSymbols, sizeX, sizeY, sizeZ);

					cudaCompress::util::unPredictor7_tiles(dpImage, (int16_t*)dpScratch, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);
					cudaCompress::util::u2f((uint16_t *)dpScratch, dpBuffer, sizeX*sizeY);

					cudaCompress::util::multiply(dpBuffer, dpBuffer, quantStep, sizeX * sizeY);
					// inverse variance stabilization
					cudaCompress::util::invVst(dpBuffer, dpBuffer, sizeX * sizeY, bgLevel, conversion, readNoise);
					//back to int16_t from float
					cudaCompress::util::f2u(dpBuffer, (uint16_t*)dpImage, sizeX*sizeY);
					break;
				case 3:
					cudaCompress::decodeRLHuff(pInstance, bitStream, pdpSymbols, 1, sizeX * sizeY);
					cudaCompress::util::unsymbolize(dpImage, dpSymbols, sizeX, sizeY, sizeZ);
					//cudaCompress::util::unQuantize(dpImage, dpBuffer, sizeX, sizeX, sizeY);
					cudaCompress::util::unPredictor7_tiles_wnll(dpImage, dpBuffer, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);
					// undo scaling from quantization
					cudaCompress::util::multiply(dpBuffer, dpBuffer, quantStep, sizeX * sizeY);
					// inverse variance stabilization
					cudaCompress::util::invVst(dpBuffer, dpBuffer, sizeX * sizeY, bgLevel, conversion, readNoise);
					cudaCompress::util::offset(dpBuffer, dpBuffer, -quantStep*quantStep / (12.0*conversion), sizeX * sizeY);
					//back to int16_t from float
					cudaCompress::util::f2u(dpBuffer, (uint16_t*)dpImage, sizeX*sizeY);
					break;
				case 27:
				case 7:
					cudaCompress::decodeRLHuff(pInstance, bitStream, pdpSymbols, 1, sizeX * sizeY);

					cudaCompress::util::unsymbolize(dpImage, dpSymbols, sizeX, sizeY, sizeZ);
					//cudaCompress::util::unQuantize(dpImage, dpBuffer, sizeX, sizeX, sizeY);
					cudaCompress::util::unPredictor7_tiles_wnll(dpImage, dpBuffer, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);

					cudaCompress::util::multiply(dpBuffer, dpBuffer, quantStep, sizeX * sizeY);
					cudaCompress::util::sqrArray(dpBuffer, dpBuffer, sizeX * sizeY);

					cudaCompress::util::multiply(dpBuffer, dpBuffer, conversion, sizeX * sizeY);
					cudaCompress::util::offset(dpBuffer, dpBuffer, bgLevel, sizeX * sizeY);
					//back to int16_t from float
					cudaCompress::util::f2u(dpBuffer, (uint16_t*)dpImage, sizeX*sizeY);
					break;
				case 17:
					cudaCompress::decodeRLHuff(pInstance, bitStream, pdpSymbols, 1, sizeX * sizeY);
					cudaCompress::util::unsymbolize(dpImage, dpSymbols, sizeX, sizeY, sizeZ);

					cudaCompress::util::unPredictor7_tiles(dpImage, (int16_t*)dpScratch, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);
					cudaCompress::util::u2f((uint16_t *)dpScratch, dpBuffer, sizeX*sizeY);

					cudaCompress::util::multiply(dpBuffer, dpBuffer, quantStep, sizeX * sizeY);
					cudaCompress::util::sqrArray(dpBuffer, dpBuffer, sizeX * sizeY);

					cudaCompress::util::offset(dpBuffer, dpBuffer, bgLevel, sizeX * sizeY);
					//back to int16_t from float
					cudaCompress::util::f2u(dpBuffer, (uint16_t*)dpImage, sizeX*sizeY);
					break;
				case 37:
					cudaCompress::decodeRLHuff(pInstance, bitStream, pdpSymbols, 1, sizeX * sizeY);

					cudaCompress::util::unsymbolize(dpImage, dpSymbols, sizeX, sizeY, sizeZ);
					//cudaCompress::util::unQuantize(dpImage, dpBuffer, sizeX, sizeX, sizeY);
					cudaCompress::util::unPredictor7_tiles_wnll(dpImage, dpBuffer, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);
					// scale back by quant step
					cudaCompress::util::multiply(dpBuffer, dpBuffer, 2 * quantStep, sizeX * sizeY);
					// invese Anscombe
					cudaCompress::util::invAnscombe(dpBuffer, dpBuffer, sizeX * sizeY);
					// convert back to numbers from photons
					cudaCompress::util::multiply(dpBuffer, dpBuffer, conversion, sizeX * sizeY);
					cudaCompress::util::offset(dpBuffer, dpBuffer, bgLevel, sizeX * sizeY);
					//back to int16_t from float
					cudaCompress::util::f2u(dpBuffer, (uint16_t*)dpImage, sizeX*sizeY);
					break;
				default:
					break;
				}
				cudaCheckMsg("unpredictor failed");
			}
			else {
				dwtLevel = dwtLevel - 100;
				BitStreamReadOnly bitStream(i_bitStream.data(), uint(i_bitStream.size() * sizeof(uint) * 8));
				cudaCompress::decodeRLHuff(pInstance, bitStream, &dpSymbols, 1, sizeX * sizeY);
				cudaCompress::util::unquantizeFromSymbols2D(dpBuffer, dpSymbols, sizeX, sizeY, quantStep);
				for (int i = dwtLevel - 1; i > 0; i--)
				{
					cudaCompress::util::dwtFloat2DInverse(
						dpBuffer, dpScratch, dpBuffer, sizeX / pow(2.0, i), sizeY / pow(2.0, i), 1, sizeX, sizeX, 0);
				}
				cudaCompress::util::dwtFloat2DInverseToUshort(
					(uint16_t*)dpImage, dpScratch, dpBuffer, sizeX, sizeY, 1, sizeX, sizeX, 0);
				
				
				
			}
		}





		// TODO: swap cases in CPU


		void compressImageLLCPU(
			//Instance* pInstance,
			std::vector<uint>& i_bitStream, // bitStream for compressed data
			int16_t* pImage,  // input image in GPU memory
			int16_t* pBuffer,
			int16_t* pScratch,
			uint16_t* pSymbols,
			int sizeX, int sizeY, int sizeZ,         // image size
			size_t dwtLevel, int tileSize)               // quantization step
		{
			sizeY = sizeY * sizeZ;
			// Do multi-level DWT in the same buffers. Need to specify pitch now!
			
			memcpy(pBuffer, pImage, sizeX*sizeY * sizeof(int16_t));
			switch (dwtLevel) {
			case 1:
			case 2:
			case 7:
				cudaCompress::util::predictor7_tilesCPU(pImage, pBuffer, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);
				break;
			default:
				break;
			}
			cudaCheckMsg("predictor failed");
			

			cudaCompress::util::symbolizeCPU(pSymbols, pBuffer, sizeX, sizeY, 1);

			// Run-length + Huffman encode the quantized coefficients.
			cudaCompress::BitStream bitStream(&i_bitStream);
			cudaCompress::BitStream* pbitStream = &bitStream;
			
			std::vector<Symbol16> symbolsVec(pSymbols, pSymbols + sizeX*sizeY);
			std::vector<Symbol16>* pSymbolsVec = &symbolsVec;
			//cudaCompress::encodeRLHuff(pInstance, bitStream, &pSymbols, 1, sizeX * sizeY);
			cudaCompress::encodeRLHuffCPU(&pbitStream, &pSymbolsVec, 1, 128);
			//return bitStream;
		}

		// pSymbols has to be initialized to 0
		void decompressImageLLCPU(
			//Instance* pInstance,
			std::vector<uint>& i_bitStream, // bitStream for compressed data
			int16_t* pImage,  // input image in GPU memory
			int16_t* pBuffer,
			int16_t* pScratch,
			uint16_t* pSymbols,
			int sizeX, int sizeY, int sizeZ,         // image size
			size_t dwtLevel, int tileSize)
		{
			sizeY = sizeY * sizeZ;
			BitStreamReadOnly bitStream(i_bitStream.data(), uint(i_bitStream.size() * sizeof(uint) * 8));
			//cudaCompress::decodeRLHuff(pInstance, bitStream, &pSymbols, 1, sizeX * sizeY);
			BitStreamReadOnly* pBitStream = &bitStream;
			std::vector<Symbol16> symbolsReconst;
			std::vector<Symbol16>* psymbolsReconst = &symbolsReconst;
			cudaCompress::decodeRLHuffCPU(&pBitStream, &psymbolsReconst, sizeX * sizeY, 1, 128, true);
			assert(symbolsReconst.size() == sizeX * sizeY);
			memcpy(pSymbols, symbolsReconst.data(), sizeX * sizeY * sizeof(Symbol16));

			cudaCompress::util::unsymbolizeCPU(pBuffer, pSymbols, sizeX, sizeY, 1);
			//cudaMemcpy(dpBuffer, dpSymbols, sizeX * sizeY * sizeof(int16_t), cudaMemcpyDeviceToDevice);

			
			memcpy(pImage, pBuffer, sizeX*sizeY * sizeof(int16_t));
			switch (dwtLevel) {					
			case 1:
			case 2:
			case 7:
				cudaCompress::util::unPredictor7_tilesCPU(pBuffer, pImage, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);
				break;
			default:
				break;
			}
			cudaCheckMsg("unpredictor failed");
			
		}

		void compressImageCPU(
			//Instance* pInstance,
			std::vector<uint>& i_bitStream, // bitStream for compressed data
			int16_t* pImage,  // input image in GPU memory
			float* pBuffer,
			float* pScratch,
			uint16_t* pSymbols,
			int sizeX, int sizeY, int sizeZ,         // image size
			size_t dwtLevel, float quantStep, float bgLevel, int tileSize, float conversion, float readNoise)               // quantization step
		{
			sizeY = sizeY * sizeZ;
			sizeZ = 1;
			
			//uint16_t* dpSymbolsN = dpSymbols + sizeX * sizeY;
			uint16_t* ppSymbols[1] = { pSymbols };

			cudaCompress::util::u2fCPU((uint16_t*)pImage, pBuffer, sizeX * sizeY);
			memset(pScratch, 0, sizeX *  sizeY * sizeof(int16_t));

			switch (dwtLevel) {					
			case 1: // first version, square root /w readnoise + prediction7 + quantization within noise level
			case 3: // different offset in decompression to test bias
			case 11: // cpu decompression
				// variance stabilization
				cudaCompress::util::vstCPU(pBuffer,pBuffer, sizeX * sizeY, bgLevel, conversion, readNoise);
				// scale with quantization step
				cudaCompress::util::multiplyCPU(pBuffer, pBuffer, 1 / quantStep, sizeX * sizeY);
				// run prediction + quantization
				cudaCompress::util::predictor7_tiles_wnllCPU(pBuffer, pScratch, pImage, sizeX, sizeX, sizeY, tileSize);
				break;
			case 2: // swapped: square root /w readnoise + quantization + prediction7
				// variance stabilization
				cudaCompress::util::vstCPU(pBuffer, pBuffer, sizeX * sizeY, bgLevel, conversion, readNoise);
				// scale with quantization step
				cudaCompress::util::multiplyCPU(pBuffer, pBuffer, 1 / quantStep, sizeX * sizeY);
				// run  quantization first then prediction
				cudaCompress::util::f2uCPU(pBuffer, (uint16_t*)pScratch, sizeX * sizeY);
				//cudaMemcpy(dpImage, dpScratch, sizeX*sizeY * sizeof(int16_t), cudaMemcpyDeviceToDevice);
				cudaCompress::util::predictor7_tilesCPU((int16_t*)pScratch, pImage, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);
				break;
			default:
				break;
			}
			cudaCheckMsg("predictor failed");

			cudaCompress::util::symbolizeCPU(pSymbols, pImage, sizeX, sizeY, sizeZ);				
			// Run-length + Huffman encode the quantized coefficients.
			cudaCompress::BitStream bitStream(&i_bitStream);
			cudaCompress::BitStream* pbitStream = &bitStream;

			std::vector<Symbol16> symbolsVec(pSymbols, pSymbols + sizeX*sizeY);
			std::vector<Symbol16>* pSymbolsVec = &symbolsVec;
			//cudaCompress::encodeRLHuff(pInstance, bitStream, &pSymbols, 1, sizeX * sizeY);
			cudaCompress::encodeRLHuffCPU(&pbitStream, &pSymbolsVec, 1, 128);
			
		}

		void decompressImageCPU(
			//Instance* pInstance,
			std::vector<uint>& i_bitStream, // bitStream for compressed data
			int16_t* pImage,  // input image in GPU memory
			float* pBuffer,
			float* pScratch,
			uint16_t* pSymbols,
			int sizeX, int sizeY, int sizeZ,         // image size
			size_t dwtLevel, float quantStep, float bgLevel, int tileSize, float conversion, float readNoise)
		{
			sizeY = sizeY * sizeZ;
			sizeZ = 1;
			
			dwtLevel = dwtLevel - 100;
			BitStreamReadOnly bitStream(i_bitStream.data(), uint(i_bitStream.size() * sizeof(uint) * 8));
			//cudaCompress::decodeRLHuff(pInstance, bitStream, &pSymbols, 1, sizeX * sizeY);
			BitStreamReadOnly* pBitStream = &bitStream;
			std::vector<Symbol16> symbolsReconst;
			std::vector<Symbol16>* psymbolsReconst = &symbolsReconst;
			cudaCompress::decodeRLHuffCPU(&pBitStream, &psymbolsReconst, sizeX * sizeY, 1, 128, true);
			assert(symbolsReconst.size() == sizeX * sizeY);
			memcpy(pSymbols, symbolsReconst.data(), sizeX * sizeY * sizeof(Symbol16));

			cudaCompress::util::unsymbolizeCPU(pImage, pSymbols, sizeX, sizeY, sizeZ);

			switch (dwtLevel) {
			case 1:
			case 11:
				//cudaCompress::util::unQuantize(dpImage, dpBuffer, sizeX, sizeX, sizeY);
				cudaCompress::util::unPredictor7_tiles_wnllCPU(pImage, pBuffer, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);
				// undo scaling from quantization
				cudaCompress::util::multiplyCPU(pBuffer, pBuffer, quantStep, sizeX * sizeY);
				// inverse variance stabilization
				cudaCompress::util::invVstCPU(pBuffer, pBuffer, sizeX * sizeY, bgLevel, conversion, readNoise);
				//back to int16_t from float
				cudaCompress::util::f2uCPU(pBuffer, (uint16_t*)pImage, sizeX*sizeY);
				break;
			case 2:
				cudaCompress::util::unPredictor7_tilesCPU(pImage, (int16_t*)pScratch, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);
				cudaCompress::util::u2fCPU((uint16_t *)pScratch, pBuffer, sizeX*sizeY);

				cudaCompress::util::multiplyCPU(pBuffer, pBuffer, quantStep, sizeX * sizeY);
				// inverse variance stabilization
				cudaCompress::util::invVstCPU(pBuffer, pBuffer, sizeX * sizeY, bgLevel, conversion, readNoise);
				//back to int16_t from float
				cudaCompress::util::f2uCPU(pBuffer, (uint16_t*)pImage, sizeX*sizeY);
				break;
			default:
				break;
			}
			cudaCheckMsg("unpredictor failed");
		}


	//}

}