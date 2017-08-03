#include "cudaCompressFunctions.h"

using namespace cudaCompress;

void compressImageLL3D(
	Instance* pInstance,
	std::vector<uint>& i_bitStream, // bitStream for compressed data
	cudaPitchedPtr dppImage,  // input image in GPU memory
	cudaPitchedPtr dppBuffer,
	cudaPitchedPtr dppScratch,
	ushort* dpSymbols,
	cudaExtent extent,
	cudaPos offset,
	size_t dwtLevel)               // DWT leveles
{

	// Do multi-level DWT in the same buffers. Need to specify pitch now!
	if (dwtLevel < 1) {
		cudaMemcpy3DParms myParams = { 0 };
		myParams.dstPtr = dppBuffer;
		myParams.srcPtr = dppImage;
		myParams.extent = extent;
		myParams.kind = cudaMemcpyDeviceToDevice;
		myParams.srcPos = offset;
		cudaMemcpy3D(&myParams);
	}
	else {
		cudaCompress::util::dwtIntForward(
			(short*)dppBuffer.ptr, (short*)dppScratch.ptr, (short*)((char*)dppImage.ptr + offset.y*dppImage.pitch + offset.z*dppImage.pitch*dppImage.ysize + offset.x),
			extent.width / sizeof(short), extent.height, extent.depth,
			dppBuffer.pitch / sizeof(short), dppBuffer.pitch / sizeof(short)* dppBuffer.ysize,
			dppImage.pitch / sizeof(short), dppImage.pitch / sizeof(short)* dppImage.ysize);
		for (size_t i = 1; i < dwtLevel; i++)
		{
			cudaCompress::util::dwtIntForward(
				(short*)dppBuffer.ptr, (short*)dppScratch.ptr, (short*)dppBuffer.ptr,
				extent.width / sizeof(short) / pow(2, i), extent.height / pow(2, i), std::max(extent.depth / pow(2, i), 1.0),
				dppBuffer.pitch / sizeof(short), dppBuffer.pitch / sizeof(short)* dppBuffer.ysize,
				dppBuffer.pitch / sizeof(short), dppBuffer.pitch / sizeof(short)* dppBuffer.ysize);
		}
	}

	cudaCompress::util::symbolize(dpSymbols, (short*)dppBuffer.ptr, extent.width / sizeof(short), extent.height, extent.depth, dppBuffer.pitch / sizeof(short), dppBuffer.pitch / sizeof(short)* dppBuffer.ysize);

	cudaCompress::BitStream bitStream(&i_bitStream);
	cudaCompress::encodeRLHuff(pInstance, bitStream, &dpSymbols, 1, extent.width / sizeof(short)* extent.height * extent.depth);
}

void decompressImageLL3D(
	Instance* pInstance,
	const std::vector<uint>& i_bitStream, // compressed image data
	cudaPitchedPtr dppImage,  // input image in GPU memory
	cudaPitchedPtr dppBuffer,
	cudaPitchedPtr dppScratch,
	ushort* dpSymbols,
	cudaExtent extent,
	cudaPos offset,
	int dwtLevel)
{
	BitStreamReadOnly bitStream(i_bitStream.data(), uint(i_bitStream.size() * sizeof(uint)* 8));
	cudaCompress::decodeRLHuff(pInstance, bitStream, &dpSymbols, 1, extent.width / sizeof(short)* extent.height * extent.depth);

	cudaCompress::util::unsymbolize((short*)dppBuffer.ptr, dpSymbols,
		extent.width / sizeof(short), extent.height, extent.depth,
		dppBuffer.pitch / sizeof(short), dppBuffer.pitch / sizeof(short)* dppBuffer.ysize);

	if (dwtLevel < 1) {
		cudaMemcpy3DParms myParams = { 0 };
		myParams.srcPtr = dppBuffer;
		myParams.dstPtr = dppImage;
		myParams.extent = extent;
		myParams.dstPos = offset;
		myParams.kind = cudaMemcpyDeviceToDevice;
		cudaMemcpy3D(&myParams);
	}
	else {
		for (size_t i = dwtLevel - 1; i > 0; i--)
		{
			cudaCompress::util::dwtIntInverse(
				(short*)dppBuffer.ptr, (short*)dppScratch.ptr, (short*)dppBuffer.ptr,
				extent.width / sizeof(short) / pow(2, i), extent.height / pow(2, i), std::max(extent.depth / pow(2, i), 1.0),
				dppBuffer.pitch / sizeof(short), dppBuffer.pitch / sizeof(short)* dppBuffer.ysize,
				dppBuffer.pitch / sizeof(short), dppBuffer.pitch / sizeof(short)* dppBuffer.ysize);
		}
		cudaCompress::util::dwtIntInverse(
			(short*)((char*)dppImage.ptr + offset.y*dppImage.pitch + offset.z*dppImage.pitch*dppImage.ysize + offset.x), (short*)dppScratch.ptr, (short*)dppBuffer.ptr,
			extent.width / sizeof(short), extent.height, extent.depth,
			dppImage.pitch / sizeof(short), dppImage.pitch / sizeof(short)* dppImage.ysize,
			dppBuffer.pitch / sizeof(short), dppBuffer.pitch / sizeof(short)* dppBuffer.ysize);
	}
}

void compressImage3D(
	Instance* pInstance,
	std::vector<uint>& i_bitStream, // bitStream for compressed data
	cudaPitchedPtr dppImage,  // input image in GPU memory
	cudaPitchedPtr dppBuffer,
	cudaPitchedPtr dppScratch1,
	cudaPitchedPtr dppScratch2,
	ushort* dpSymbols,
	cudaExtent extent,
	cudaPos offset,
	size_t dwtLevel,
	float quantStep)
{

	if (dwtLevel < 1) {

		cudaMemcpy3DParms myParams = { 0 };
		myParams.dstPtr = dppBuffer;
		myParams.srcPtr = dppImage;
		myParams.extent = make_cudaExtent(extent.width / sizeof(float)* sizeof(short), extent.height, extent.depth);
		myParams.kind = cudaMemcpyDeviceToDevice;
		myParams.srcPos = offset;
		cudaMemcpy3D(&myParams);

		cudaCompress::util::symbolize(dpSymbols, (short*)dppBuffer.ptr,
			extent.width / sizeof(float), extent.height, extent.depth,
			dppBuffer.pitch / sizeof(short), dppBuffer.pitch / sizeof(short)* dppBuffer.ysize);
	}
	else {
		cudaCompress::util::dwtFloat3DForwardFromUshort(
			(float*)dppBuffer.ptr, (float*)dppScratch1.ptr, (float*)dppScratch2.ptr, (ushort*)((char*)dppImage.ptr + offset.y*dppImage.pitch + offset.z*dppImage.pitch*dppImage.ysize + offset.x),
			extent.width / sizeof(float), extent.height, extent.depth, 1,
			dppBuffer.pitch / sizeof(float), dppBuffer.pitch / sizeof(float)* dppBuffer.ysize,
			dppImage.pitch / sizeof(ushort), dppImage.pitch / sizeof(ushort)* dppImage.ysize);
		for (size_t i = 1; i < dwtLevel; i++)
		{
			cudaCompress::util::dwtFloat3DForward(
				(float*)dppBuffer.ptr, (float*)dppScratch1.ptr, (float*)dppScratch2.ptr, (float*)dppBuffer.ptr,
				extent.width / sizeof(float) / pow(2, i), extent.height / pow(2, i), std::max(extent.depth / pow(2, i), 1.0), 1,
				dppBuffer.pitch / sizeof(float), dppBuffer.pitch / sizeof(float)* dppBuffer.ysize,
				dppBuffer.pitch / sizeof(float), dppBuffer.pitch / sizeof(float)* dppBuffer.ysize);
		}
		cudaCompress::util::quantizeToSymbols(dpSymbols, (float*)dppBuffer.ptr,
			extent.width / sizeof(float), extent.height, extent.depth, quantStep,
			dppBuffer.pitch / sizeof(float), dppBuffer.pitch / sizeof(float)* dppBuffer.ysize);
	}



	cudaCompress::BitStream bitStream(&i_bitStream);
	cudaCompress::encodeRLHuff(pInstance, bitStream, &dpSymbols, 1, extent.width / sizeof(float)* extent.height * extent.depth);
}

void decompressImage3D(
	Instance* pInstance,
	const std::vector<uint>& i_bitStream, // compressed image data
	cudaPitchedPtr dppImage,  // input image in GPU memory
	cudaPitchedPtr dppBuffer,
	cudaPitchedPtr dppScratch1,
	cudaPitchedPtr dppScratch2,
	ushort* dpSymbols,
	cudaExtent extent,
	cudaPos offset,
	int dwtLevel,
	float quantStep)
{
	BitStreamReadOnly bitStream(i_bitStream.data(), uint(i_bitStream.size() * sizeof(uint)* 8));
	cudaCompress::decodeRLHuff(pInstance, bitStream, &dpSymbols, 1, extent.width / sizeof(float)* extent.height * extent.depth);



	if (dwtLevel < 1) {
		cudaCompress::util::unsymbolize((short*)dppBuffer.ptr, dpSymbols,
			extent.width / sizeof(float), extent.height, extent.depth,
			dppBuffer.pitch / sizeof(short), dppBuffer.pitch / sizeof(short)* dppBuffer.ysize);

		cudaMemcpy3DParms myParams = { 0 };
		myParams.srcPtr = dppBuffer;
		myParams.dstPtr = dppImage;
		myParams.extent = make_cudaExtent(extent.width / sizeof(float)* sizeof(short), extent.height, extent.depth);
		myParams.dstPos = offset;
		myParams.kind = cudaMemcpyDeviceToDevice;
		cudaMemcpy3D(&myParams);

	}
	else {
		cudaCompress::util::unquantizeFromSymbols((float*)dppBuffer.ptr, dpSymbols,
			extent.width / sizeof(float), extent.height, extent.depth, quantStep,
			dppBuffer.pitch / sizeof(float), dppBuffer.pitch / sizeof(float)* dppBuffer.ysize);
		for (size_t i = dwtLevel - 1; i > 0; i--)
		{
			cudaCompress::util::dwtFloat3DInverse(
				(float*)dppBuffer.ptr, (float*)dppScratch1.ptr, (float*)dppScratch2.ptr, (float*)dppBuffer.ptr,
				extent.width / sizeof(float) / pow(2, i), extent.height / pow(2, i), std::max(extent.depth / pow(2, i), 1.0), 1,
				dppBuffer.pitch / sizeof(float), dppBuffer.pitch / sizeof(float)* dppBuffer.ysize,
				dppBuffer.pitch / sizeof(float), dppBuffer.pitch / sizeof(float)* dppBuffer.ysize);
		}
		cudaCompress::util::dwtFloat3DInverseToUshort(
			(ushort*)((char*)dppImage.ptr + offset.y*dppImage.pitch + offset.z*dppImage.pitch*dppImage.ysize + offset.x), (float*)dppScratch1.ptr, (float*)dppScratch2.ptr, (float*)dppBuffer.ptr,
			extent.width / sizeof(float), extent.height, extent.depth, 1,
			dppImage.pitch / sizeof(short), dppImage.pitch / sizeof(short)* dppImage.ysize,
			dppBuffer.pitch / sizeof(float), dppBuffer.pitch / sizeof(float)* dppBuffer.ysize);
	}
}