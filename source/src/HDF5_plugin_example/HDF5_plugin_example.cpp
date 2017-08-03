/*---TODO---*/
/*--LICENSE-*/

#include <stdio.h>
#include <fstream>
#include <string>
#include <hdf5.h>
#include <stdlib.h> // necessary for malloc
#include <chrono>
#include <math.h>
//#include "HDF5_Plugin.h"


using namespace std::chrono;


template<typename T>
double computeRange(const T* pData, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
	double min = std::numeric_limits<double>::max();
	double max = -std::numeric_limits<double>::max();

	for (unsigned int i = 0; i < count; i++) {
		double val = double(pData[i * numcomps + comp]);
		if (val < min)
			min = val;
		if (val > max)
			max = val;
	}

	return max - min;
}

template<typename T>
double computeRMSError(const T* pData, const T* pReconst, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
	double result = 0.0;
	for (unsigned int i = 0; i < count; i++) {
		double diff = double(pData[i * numcomps + comp]) - double(pReconst[i * numcomps + comp]);
		result += diff * diff;
	}
	result /= double(count);
	result = sqrt(result);

	return result;
}

template<typename T>
double computePSNR(const T* pData, const T* pReconst, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
	double range = computeRange(pData, count, numcomps, comp);
	double rmse = computeRMSError(pData, pReconst, count, numcomps, comp);

	return 20.0 * log10(range / rmse);
}


#define MAX_SIZE 3473451520		// 30016*220*263*2
#define NUM_FILES 1
#define NUM_FILTERS 3
#define FILTER_NOTHING 0
#define FILTER_GZIP 200

#define ORCA_FLASH4 2.1845*1000 // 65535/30000
#define ORCA_ER 0.2275*1000 // 4095/18000
#define ORCA_FLASH4_NOISE 1.6*1000 // 1.6 e- rms noise
#define ORCA_ER_NOISE 6*1000 // 6 e- rms noise
#define ONE 1000

int main(){
	int r;
	int iterations = 1;
	int return_code = 1;
#ifdef _WIN32
	std::string base_folder("R:\\");
#else
	std::string base_folder("/dev/shm/");
#endif
	std::string files[NUM_FILES] = { 
		"drosophila_D2re_BG-_masked24_768x1600x8"
	};
	std::string outfile;

	hsize_t shapes[NUM_FILES][3] = { { 8, 1600, 768 } };
	hsize_t chunkshapes[NUM_FILES][3] = { { 1, 1600, 768 } };

	hid_t types[NUM_FILES] = { H5T_NATIVE_UINT16  };
	int type_sizes[NUM_FILES] = { 2 };
	unsigned int bgLevels[NUM_FILES] = { 0 };
	unsigned int conversions[NUM_FILES] = { ORCA_FLASH4  };
	unsigned int readNoises[NUM_FILES] = { ORCA_FLASH4_NOISE };

	int num_files = NUM_FILES;

	void* data = malloc(MAX_SIZE);
	void* data_out = malloc(MAX_SIZE);
	hsize_t* shape;
	hsize_t* chunkshape;

	unsigned int filters[NUM_FILTERS] = { 1, 1, 2 };
	unsigned int quant_steps[NUM_FILTERS] = { 0, 1000, 1000 };
	unsigned int tiles[NUM_FILTERS] = { 64, 64, 64};
	char* filter_names[NUM_FILTERS] = { "B3D_lossless", "B3D_Mode1_1.00","B3D_Mode2_1.00"};

	int num_filters = NUM_FILTERS;
	hid_t fid, sid, dset, plist = 0;

	// open benhcmark file
	std::ofstream benchmark (base_folder + "benchmark.csv");
	// write header
	benchmark << "\t";
	for (size_t i = 0; i < NUM_FILTERS; i++)
	{
		benchmark << filter_names[i] <<"\t\t\t\t";
	}
	benchmark << "\n";
	benchmark << "filename";
	for (size_t i = 0; i < NUM_FILTERS; i++)
	{
		benchmark << "\tcompression ratio\tpsnr\twrite speed\tread speed";
	}
	benchmark << "\n";

	float read_speed, write_speed, compression_ratio;

	for (size_t f = 0; f < num_files; f++)
	{
		benchmark << files[f];

		std::ifstream file(base_folder + files[f] + ".raw", std::ifstream::binary);
		if (!file.good()) return 3;


		shape = shapes[f];
		file.read((char*)data, shape[0] * shape[1] * shape[2] * type_sizes[f]);
		file.close();

		fid = 0;
		sid = 0;
		dset = 0;
		plist = 0;

		chunkshape = chunkshapes[f];
		hsize_t size = shapes[f][0] * shapes[f][1] * shapes[f][2];
		hsize_t compressed_size = size;
		memset(data_out, 0, size);

		outfile = base_folder + files[f] + "_filtered.h5";
		//char* outfile = "/dev/shm/out.h5";
		fid = H5Fcreate(outfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

		fprintf(stdout, "Testing file %s\n", files[f].c_str());

		if (fid < 0) goto failed;

		
		for (size_t iFilter = 0; iFilter < num_filters; iFilter++)
		{
			sid = H5Screate_simple(3, shape, NULL);
			if (sid < 0) goto failed;

			fprintf(stdout, "  Testing filter %s\n", filter_names[iFilter]);
			
			plist = H5Pcreate(H5P_DATASET_CREATE);
			if (plist < 0) goto failed;

			/* Chunked layout required for filters */
			r = H5Pset_chunk(plist, 3, chunkshape);
			if (r < 0) goto failed;

			//GPUResources* res = new GPUResources;
			//GPUResources* res = nullptr;

			unsigned int cd_values[6] = {quant_steps[iFilter], filters[iFilter], conversions[f], bgLevels[f], readNoises[f], tiles[iFilter] };

			//benchmark << files[f];
			/* Note the "optional" flag is necessary, as with the DEFLATE filter */
			if (filters[iFilter] != FILTER_NOTHING) {
				if (filters[iFilter] == FILTER_GZIP)
					r = H5Pset_deflate(plist, 6);
				else
					r = H5Pset_filter(plist, 32016, H5Z_FLAG_OPTIONAL, 6, cd_values);
			}
			if (r < 0) goto failed;

			dset = H5Dcreate2(fid, filter_names[iFilter], types[f], sid, 0, plist, 0);
			if (dset < 0) goto failed;

			/*r = set_str_attrib(dset, "CLASS", "IMAGE");
			r |= set_str_attrib(dset, "IMAGE_VERSION", "1.3");
			r |= set_str_attrib(dset, "IMAGE_SUBCLASS", "IMAGE_GRAYSCALE");*/

			high_resolution_clock::time_point t1 = high_resolution_clock::now();
			for (size_t i = 0; i < iterations; i++)
			{
				r = H5Dwrite(dset, types[f], H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
			}
			high_resolution_clock::time_point t2 = high_resolution_clock::now();
			compressed_size = H5Dget_storage_size(dset) / sizeof(unsigned short);

			r = H5Fflush(fid, H5F_SCOPE_GLOBAL);
			H5Dclose(dset);

			//closeDirectCudaCompress(&res);

			auto duration = duration_cast<microseconds>(t2 - t1).count();

			write_speed = size * type_sizes[f] / (1024 * 1024) / (duration / 1000.0f / 1000 / iterations);
			compression_ratio = (float)size / float(compressed_size);
			fprintf(stdout, "    Time to write %d times: %.4f s\n", iterations, duration / 1000.f / 1000);
			fprintf(stdout, "    Write speed: %.4f MB/s\n", write_speed);
			fprintf(stdout, "    Compression ratio: %.6f\n", compression_ratio);

			// read test
			//fid = H5Fopen(outfile, H5F_ACC_RDONLY, H5P_DEFAULT);
			//if (fid < 0) goto failed;

			dset = H5Dopen(fid, filter_names[iFilter], H5P_DEFAULT);

			t1 = high_resolution_clock::now();
			size_t i = 0;
			for (i = 0; i < iterations; i++)
			{
				r = H5Dread(dset, H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_out);
			}
			t2 = high_resolution_clock::now();
			duration = duration_cast<microseconds>(t2 - t1).count();

			if (r < 0) goto failed;
			//r = 2;

			/*for (size_t i = 0; i < size; i++){
				if (data[i] != data_out[i]) goto failed;
			}*/

			double psnr = computePSNR((short*)data, (short*)data_out, size);

			read_speed = size * type_sizes[f] / (1024 * 1024) / (duration / 1000.0f / 1000 / iterations);
			fprintf(stdout, "    Time to read %d times: %.4f s\n", iterations, duration / 1000.f / 1000);
			fprintf(stdout, "    Read speed: %.4f MB/s\n", read_speed);
			fprintf(stdout, "    PSNR: %.2f\n", psnr);
			fprintf(stdout, "    Success!\n\n");

			benchmark << "\t" << compression_ratio << "\t" << psnr << "\t" << write_speed << "\t" << read_speed;

			H5Dclose(dset);
			H5Sclose(sid);
			H5Pclose(plist);
			dset = 0;
			sid = 0;
			plist = 0;
		}
		benchmark << "\n";
		H5Fclose(fid);
		fid = 0;

		r = 0;

	}
	benchmark.close();
failed:

	if (dset>0)  H5Dclose(dset);
	if (sid>0)   H5Sclose(sid);
	if (plist>0) H5Pclose(plist);
	if (fid>0)   H5Fclose(fid);

	return r;
}
