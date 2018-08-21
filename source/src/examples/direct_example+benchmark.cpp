#include <fstream>
#include <string>

#include "global.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cudaCompress/B3D/B3DcompressFunctions.h>
#include <cudaCompress/Instance.h>
#include <cudaCompress/cudaUtil.h>

#include "tools/imgtools.h"


using namespace cudaCompress;

int cutilDeviceInitS(int dev)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		fprintf(stderr, "CUTIL CUDA error: no devices supporting CUDA.\n");
		exit(-1);
	}
	
	if (dev < 0)
		dev = 0;
	if (dev > deviceCount - 1) {
		fprintf(stderr, "\n");
		fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
		fprintf(stderr, ">> cutilDeviceInit (-device=%d) is not a valid GPU device. <<\n", dev);
		fprintf(stderr, "\n");
		return -dev;
	}
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	if (deviceProp.major < 1) {
		fprintf(stderr, "cutil error: GPU device does not support CUDA.\n");
		exit(-1);
	}
	if (deviceProp.major < 3) {
		fprintf(stderr, "cutil error: GPU device must support at least compute 3.0 (Kepler).\n");
		exit(-1);
	}
	printf("> Using CUDA device [%d]: %s\n", dev, deviceProp.name);
	cudaError_t err = cudaSetDevice(dev);
	if (err) fprintf(stderr, "Runtime API error : %s.\n", cudaGetErrorString(err));
	return dev;
}





#define MAX_SIZE 301*2048*2048
#define NUM_FILES 1
#define NUM_FILTERS 3
#define FILTER_NOTHING 0
#define FILTER_GZIP 200

#define TITAN_X 1
#define GTX_1080 0

#define ORCA_FLASH4_REAL 1.57284 // 65535/30000*0.72
#define ORCA_FLASH4 1.5625
#define ORCA_ER 0.15925 // 4095/18000*0.7

int main()
{
	int device = 0;
	cutilDeviceInitS(device);

	cudaEvent_t g_start, g_end;
	cudaEventCreate(&g_start);
	cudaEventCreate(&g_end);
	cudaEventRecord(g_start);


	cudaCompress::Instance* pInstance = nullptr; // the cudaCompress Instance.
	cudaCompress::Symbol16* dpSymbols = nullptr; // input/output for entropy coder.
	//cudaCompress::Symbol16* dpSigns = nullptr; // input/output to store sign bits.

	int repetitions = 1;
	int return_code = 1;
#ifdef _WIN32
	std::string base_folder("R:\\");
#else
	std::string base_folder("/dev/shm/");
#endif
	std::string files[NUM_FILES] = { 
		"drosophila_D2re_BG-_masked24_768x1600x8"
	};
	size_t shapes[NUM_FILES][3] = { { 8, 1600, 768 } };
	size_t chunkshapes[NUM_FILES][3] = { { 1, 1600, 768 } };

	float conversion = ORCA_FLASH4;

	int numFiles = NUM_FILES;
	
	unsigned int filters[NUM_FILTERS] = { 1,1,1 };
	unsigned int quant_steps[NUM_FILTERS] = { 0, 1000, 3000 };
	unsigned int bgLevels[NUM_FILTERS] = { 0, 0, 0 };
	unsigned int tiles[NUM_FILTERS] = { 48,48,48 };
	std::string filter_names[NUM_FILTERS] = { "B3D_LL", "B3D_1.0",  "B3D_3.0" };
	// open benhcmark file
	std::ofstream benchmark("benchmark.csv");
	// write header
	benchmark << "\t";
	for (size_t i = 0; i < NUM_FILTERS; i++)
	{
		benchmark << filter_names[i] << "\t\t\t";
	}
	benchmark << "\n";
	benchmark << "filename";
	for (size_t i = 0; i < NUM_FILTERS; i++)
	{
		benchmark << "\tratio\twrite (MB/s)\tread (MB/s)\tpsnr";
	}
	benchmark << "\n";
	
	float time = 0.0f;
	float read_speed, write_speed, compression_ratio;

	

	for (size_t f = 0; f < numFiles; f++)
	{
		fprintf(stdout, "Testing file %s\n", files[f].c_str());
		benchmark << files[f];

		//int sizeX = 2048, sizeY = 2048;
		int sizeX = shapes[f][2], sizeY = shapes[f][1], sizeZ = shapes[f][0];
		//int sizeX = 100, sizeY = 50, sizeZ = 10;
		int chunkSizeX = chunkshapes[f][2], chunkSizeY = chunkshapes[f][1], chunkSizeZ = chunkshapes[f][0];
		bool invDWT = true;

		//size_t dwtLevels = 3;

		int iX = (sizeX + chunkSizeX - 1) / chunkSizeX;
		int iY = (sizeY + chunkSizeY - 1) / chunkSizeY;
		int iZ = (sizeZ + chunkSizeZ - 1) / chunkSizeZ;

		int alignedSizeX = iX * chunkSizeX;
		int alignedSizeY = iY * chunkSizeY;
		int alignedSizeZ = iZ * chunkSizeZ;

		//int16_t* dpImage = nullptr;

		cudaError status;

		// Read image data from file.
		std::vector<int16_t> image(sizeX * sizeY * sizeZ);
		std::ifstream file(base_folder + files[f] + ".raw", std::ifstream::binary);
		//std::ifstream file("F:/data/gradient_100x50x10.raw", std::ifstream::binary);
		if (!file.good()) return 2;
		file.read((char*)image.data(), sizeX * sizeY * sizeZ * sizeof(int16_t));
		file.close();

		//for (size_t i = 0; i < image.size(); i++) {
		//image[i] = _byteswap_ushort(image[i]);
		//image[i] += 1000;
		//}
		
		for (size_t iFilter = 0; iFilter < NUM_FILTERS; iFilter++)
		{
			fprintf(stdout, "  Testing filter %s\n", filter_names[iFilter].c_str());

			float quantStep = (float)quant_steps[iFilter] / 1000.0 * sqrt(conversion) * 0.8;
			// Initialize cudaCompress, allocate GPU resources and upload data.
			//cutilDeviceInitS(device);
			pInstance = cudaCompress::createInstance(device, 2, chunkSizeX * chunkSizeY * chunkSizeZ * sizeof(float), 128, 16);

			
			int16_t* dpImage = nullptr;
			int16_t* dpBuffer = nullptr;
			int16_t* dpScratch = nullptr;
			cudaMalloc(&dpImage, sizeX * sizeY * sizeZ * sizeof(int16_t));
			cudaMalloc(&dpScratch, chunkSizeX * chunkSizeY * chunkSizeZ * sizeof(float));
			cudaMalloc(&dpBuffer, chunkSizeX * chunkSizeY * chunkSizeZ * sizeof(float));
			cudaMalloc(&dpSymbols, chunkSizeX * chunkSizeY * chunkSizeZ * sizeof(cudaCompress::Symbol16));
			


			float file_size = sizeX * sizeY * sizeZ * sizeof(int16_t);


			//cudaMalloc(&dpScratch, sizeX * sizeY * sizeof(float));
			//cudaMalloc(&dpBuffer, sizeX * sizeY * sizeof(float));

			// create event objects
			cudaEvent_t start, end;
			cudaEventCreate(&start);
			cudaEventCreate(&end);

			uint iterations = iX * iY * iZ;

			// Compress the image.

			std::vector<std::vector<uint>> bitStream(iterations);
			//std::vector<uint> bitStream;

			cudaPos offset = { 0 };

			//std::vector<int16_t> signs(sizeX * sizeY * sizeZ);

			//start 3d chunked compressing
			cudaEventRecord(start);
			//cutilDeviceInitS(device);
			for (size_t rep = 0; rep < repetitions; rep++)
			{
				//cutilDeviceInitS(device);
				cudaMemcpy(dpImage, image.data(), sizeX * sizeY * sizeZ * sizeof(int16_t), cudaMemcpyHostToDevice);
				
				for (size_t i = 0; i < iterations; i++)
				{
					//cutilDeviceInitS(device);
					cudaMemset(dpSymbols, 0, chunkSizeX*chunkSizeY*chunkSizeZ * sizeof(cudaCompress::Symbol16));
					if (quantStep == 0)
						compressImageLL(pInstance, bitStream[i], dpImage + chunkSizeX*chunkSizeY*chunkSizeZ*i, dpBuffer, dpScratch, dpSymbols, chunkSizeX, chunkSizeY, chunkSizeZ, filters[iFilter], tiles[iFilter]);
					else
						compressImage(pInstance, bitStream[i], dpImage + chunkSizeX*chunkSizeY*chunkSizeZ*i, (float*)dpBuffer, (float*)dpScratch, dpSymbols, chunkSizeX, chunkSizeY, chunkSizeZ, filters[iFilter], quantStep, bgLevels[iFilter], tiles[iFilter]);
					//cudaMemcpy(signs.data() + sizeX*sizeY*i, dpSigns, sizeX * sizeY * sizeof(int16_t), cudaMemcpyDeviceToHost);
				}

			}
			cudaEventRecord(end);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&time, start, end);

			

			// Write compression rate to stdout.
			unsigned int compressedSize = 0;
			for (size_t i = 0; i < iterations; i++)
			{
				compressedSize += bitStream[i].size() * sizeof(uint);
			}

			float ratio = float(compressedSize) / float(sizeX * sizeY * sizeZ * sizeof(int16_t));
			compression_ratio = 1 / ratio;
			printf("Compressed size: %.2f kB  (%.2f%%, %.4f)\n", compressedSize / 1024.0f, ratio * 100, 1 / ratio);
			printf("Compression time: %.2f ms\n", time / repetitions);
			write_speed = file_size / 1024.0 / 1024.0 / time * repetitions * 1000;
			printf("Compression speed: %.2f MB/s\n", write_speed);
			
			std::ofstream out;
			
			//out.open(base_folder + "output\\" + files[f] + "_" + filter_names[iFilter] + "_" + "signs.raw", std::ofstream::binary);
			//out.write((char*)signs.data(), sizeX * sizeY * sizeZ * sizeof(int16_t));
			//out.close();
			//cutilDeviceInitS(device);
			cudaMemset(dpImage, 0, sizeX * sizeY * sizeof(int16_t));
			//status = cudaMemset(dppImage.ptr, 0, dppImage.pitch * alignedSizeY * alignedSizeZ);

			std::vector<int16_t> imageReconst(sizeX * sizeY * sizeZ);
			//std::vector<int16_t> signsReconst(sizeX * sizeY * sizeZ);
						
			
			cudaEventRecord(start);
			//cutilDeviceInitS(device);
			for (size_t rep = 0; rep < repetitions; rep++)
			{
				for (size_t i = 0; i < iterations; i++)
				{
					//cutilDeviceInitS(device);
					cudaMemset(dpSymbols, 0, chunkSizeX*chunkSizeY*chunkSizeZ* sizeof(cudaCompress::Symbol16));
					if (quantStep == 0)
						decompressImageLL(pInstance, bitStream[i], dpImage + chunkSizeX*chunkSizeY*chunkSizeZ*i, dpBuffer, dpScratch, dpSymbols, chunkSizeX, chunkSizeY, chunkSizeZ, filters[iFilter], tiles[iFilter]);
					else
						decompressImage(pInstance, bitStream[i], dpImage + chunkSizeX*chunkSizeY*chunkSizeZ*i, (float*)dpBuffer, (float*)dpScratch, dpSymbols, chunkSizeX, chunkSizeY, chunkSizeZ, filters[iFilter], quantStep, bgLevels[iFilter], tiles[iFilter]);

					//cudaMemcpy(signsReconst.data() + sizeX*sizeY*i, dpSigns, sizeX * sizeY * sizeof(int16_t), cudaMemcpyDeviceToHost);
				}

				
				cudaMemcpy(imageReconst.data(), dpImage, sizeX * sizeY * sizeZ * sizeof(int16_t), cudaMemcpyDeviceToHost);
			}
			cudaEventRecord(end);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&time, start, end);

			printf("Decompression time: %.2f ms\n", time / repetitions);
			read_speed = file_size / 1024.0 / 1024.0 / time * repetitions * 1000;
			printf("Decompression speed: %.2f MB/s\n", read_speed);
			// Download reconstructed image and write to file.
			double psnr = computePSNR(image.data(), imageReconst.data(), sizeX * sizeY * sizeZ);
			printf("PSNR: %.2f\n", psnr);

			//out.open(base_folder + "output\\" + files[f] + "_" + filter_names[iFilter] + "_" + "signsReconst.raw", std::ofstream::binary);
			//out.write((char*)signsReconst.data(), sizeX * sizeY * sizeZ * sizeof(int16_t));
			//out.close();

			//out.open(base_folder + "output\\" + files[f] + "_" + filter_names[iFilter] + ".raw", std::ofstream::binary);
			out.open(base_folder +  files[f] + "_" + filter_names[iFilter] + ".raw", std::ofstream::binary);
			out.write((char*)imageReconst.data(), sizeX * sizeY * sizeZ * sizeof(int16_t));
			out.close();

			// Cleanup.
			//cudaFree(dpSigns);
			cudaFree(dpSymbols);
			cudaFree(dpImage);
			cudaFree(dpBuffer);
			cudaFree(dpScratch);

			/*
			cudaFree(dppBuffer.ptr);
			cudaFree(dppScratch.ptr);
			cudaFree(dppImage.ptr);
			*/

			cudaCompress::destroyInstance(pInstance);

			benchmark << "\t" << compression_ratio << "\t" << write_speed << "\t" << read_speed << "\t" << psnr;
		} // end filter loop
		//printf("Complete compression + decompression time: %.f ms.\n", time);

		benchmark << "\n";
	} // end file loop

	benchmark.close();

	cudaEventRecord(g_end);

	cudaEventSynchronize(g_end);
	cudaEventElapsedTime(&time, g_start, g_end);
	return 0;
}
