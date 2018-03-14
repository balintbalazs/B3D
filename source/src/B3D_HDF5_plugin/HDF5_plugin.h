#ifndef __HDF5_CUDACOMPRESS__
#define __HDF5_CUDACOMPRESS__


#include <H5PLextern.h>
#include <H5DOpublic.h>

#include <cudaCompress/B3D/B3DcompressFunctions.h>

#include "GPUResources.h"
//#include "CompressHeightfield.h"


/*
#include <cuda_runtime.h>

#include <cudaCompress/Encode.h>
#include <cudaCompress/util/DWT.h>
#include <cudaCompress/util/Quantize.h>
#include <cudaCompress/util/Predictors.h>
#include <cudaCompress/cudaUtil.h>




#include <cudaCompress/CPU/EncodeCPU.h>
#include <cudaCompress/CPU/QuantizeCPU.h>
#include <cudaCompress/CPU/PredictorsCPU.h>*/

using namespace cudaCompress;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32

#ifdef HDF5_PLUGIN_EXPORTS  // this is defined in the c/c++ page on the project properties (this only shows up if there is a cpp file in the project)
#define DLL __declspec(dllexport) 
#else
#define DLL __declspec(dllimport) 
#endif

#else
#define DLL
#endif

#define H5Z_FILTER_B3D 32016
//#define PUSH_ERR(func, minor, str)  H5Epush1(__FILE__, func, __LINE__, H5E_PLINE, minor, str)

#define H5Z_FILTER_B3D_VERSION 1
#define N_CD_VALUES 7

enum DATA_TYPE
{
	UINT8_TYPE = 0,
	UINT16_TYPE = 1,
	UINT32_TYPE = 2,
	UINT64_TYPE = 3,
	INT8_TYPE = 4,
	INT16_TYPE = 5,
	INT32_TYPE = 6,
	INT64_TYPE = 7,
	FLOAT32_TYPE = 8,
	FLOAT64_TYPE = 9
};

DLL int register_cudaCompress(void);

htri_t H5Z_cudaCompress_can_apply(hid_t dcpl, hid_t type, hid_t space);

herr_t H5Z_cudaCompress_set_local(hid_t dcpl, hid_t type, hid_t space);

size_t H5Z_cudaCompress_filter(unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[], size_t nbytes, size_t *buf_size, void **buf);

DLL int initDirectCudaCompress(const size_t* size, int dwtLevels, GPUResources** res);
DLL int closeDirectCudaCompress(GPUResources** res);
DLL int directCudaCompress(hid_t dset_id, hsize_t* offset, size_t* size, void* data, uint dwtLevels, float quantStep, float bgLevel, int tileSize, float conversion, float readNoise, int onDevice, GPUResources** res);

#ifdef __cplusplus
}
#endif

#endif
