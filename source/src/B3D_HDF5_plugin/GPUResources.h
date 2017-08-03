#ifndef __TUM3D__GPU_RESOURCES_H__
#define __TUM3D__GPU_RESOURCES_H__


#include "global.h"

#include <vector>
#include <inttypes.h>

#include <cudaCompress/Instance.h>


inline size_t getAlignedSize(size_t size, size_t numBytes = 128)
{
    return (size + numBytes - 1) / numBytes * numBytes;
}

template<typename T>
inline void align(T*& pData, size_t numBytes = 128)
{
    pData = (T*)getAlignedSize(size_t(pData), numBytes);
}



// Helper class to manage shared GPU resources (scratch buffers and cudaCompress instance).
// Usage: Fill a Config (use merge to build max required sizes etc over all clients), then call create() to initialize.
class Resources
{
public:
    Resources();
	//Resources(uint sizeX, uint sizeY, uint sizeZ, uint levels, int cudaDevice=-1);
    ~Resources();

    struct Config
    {
        Config();

        int cudaDevice; // set to -1 (default) to use the current one

        uint blockCountMax;
        uint elemCountPerBlockMax;
        uint codingBlockSize;
        uint log2HuffmanDistinctSymbolCountMax;

        size_t bufferSize;

        void merge(const Config& other);
    };


    virtual bool create(const Config& config) =0;
    virtual void destroy() =0;

    const Config& getConfig() const { return m_config; }


    cudaCompress::Instance* m_pCuCompInstance;


    // get a buffer of the specified size in GPU memory
    byte* getByteBuffer(size_t bytes);
    template<typename T>
    T* getBuffer(size_t count) { return (T*)getByteBuffer(count * sizeof(T)); }

    // release the last buffer(s) returned from getBuffer
    void releaseBuffer();
    void releaseBuffers(uint bufferCount);


protected:
    Config m_config;

    byte* m_dpBuffer;

    size_t m_bufferOffset;
    std::vector<size_t> m_allocatedSizes;
};


class GPUResources : public Resources
{
public:
	GPUResources(uint sizeX, uint sizeY, uint sizeZ, uint levels, int cudaDevice = -1);
	bool create(const Config& config);
	void destroy();

};


class CPUResources : public Resources
{
public:
	CPUResources(uint sizeX, uint sizeY, uint sizeZ, uint levels, int cudaDevice = -1);
	bool create(const Config& config);
	void destroy();

};

#endif
