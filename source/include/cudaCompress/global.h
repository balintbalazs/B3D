#ifndef __TUM3D_CUDACOMPRESS__GLOBAL_H__
#define __TUM3D_CUDACOMPRESS__GLOBAL_H__

#define LOG2_SYMBOL_COUNT_MAX 16
namespace cudaCompress {

typedef unsigned char byte;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef long long int int64;
typedef unsigned long long int uint64;

}

#if defined(_WIN32)
	#if defined(CUCOMP_BUILD_STANDALONE)
		#define CUCOMP_DLL
	#elif defined(CUCOMP_BUILD_DLL)
		#define CUCOMP_DLL __declspec(dllexport)
	#else
		#define CUCOMP_DLL __declspec(dllimport)
	#endif
#elif (__GNUC__ >= 4)  /* GCC 4.x has support for visibility options */
  #define CUCOMP_DLL __attribute__ ((visibility("default")))
#else
  #define CUCOMP_DLL
#endif



#endif
