#include "GPUResources.h"

#include <algorithm>
#include <cassert>

#include "cudaUtil.h"


Resources::Config::Config()
    : cudaDevice(-1), blockCountMax(0), elemCountPerBlockMax(0), codingBlockSize(0), log2HuffmanDistinctSymbolCountMax(0), bufferSize(0)
{
}

void Resources::Config::merge(const Resources::Config& other)
{
    if(cudaDevice == -1) {
        cudaDevice = other.cudaDevice;
    }

    if(blockCountMax == 0) {
        blockCountMax = other.blockCountMax;
    } else {
        blockCountMax = std::max(blockCountMax, other.blockCountMax);
    }

    if(elemCountPerBlockMax == 0) {
        elemCountPerBlockMax = other.elemCountPerBlockMax;
    } else {
        elemCountPerBlockMax = std::max(elemCountPerBlockMax, other.elemCountPerBlockMax);
    }

    if(codingBlockSize == 0) {
        codingBlockSize = other.codingBlockSize;
    } else {
        codingBlockSize = std::min(codingBlockSize, other.codingBlockSize);
    }

    if(log2HuffmanDistinctSymbolCountMax == 0) {
        log2HuffmanDistinctSymbolCountMax = other.log2HuffmanDistinctSymbolCountMax;
    } else {
        log2HuffmanDistinctSymbolCountMax = std::max(log2HuffmanDistinctSymbolCountMax, other.log2HuffmanDistinctSymbolCountMax);
    }

    if(bufferSize == 0) {
        bufferSize = other.bufferSize;
    } else {
        bufferSize = std::max(bufferSize, other.bufferSize);
    }
}


Resources::Resources()
    : m_pCuCompInstance(nullptr)
    , m_dpBuffer(nullptr)
    , m_bufferOffset(0)
{
}

Resources::~Resources()
{
    assert(m_pCuCompInstance == nullptr);
    assert(m_dpBuffer == nullptr);
}

byte* Resources::getByteBuffer(size_t bytes)
{
	assert(m_bufferOffset + bytes <= m_config.bufferSize);
	if (m_bufferOffset + bytes > m_config.bufferSize) {
		printf("ERROR: Resources::getByteBuffer: out of memory!\n");
		return nullptr;
	}

	byte* dpResult = m_dpBuffer + m_bufferOffset;
	m_allocatedSizes.push_back(bytes);
	m_bufferOffset += getAlignedSize(bytes, 128);

	return dpResult;
}

void Resources::releaseBuffer()
{
	assert(!m_allocatedSizes.empty());
	if (m_allocatedSizes.empty()) {
		printf("ERROR: Resources::releaseBuffer: no more buffers to release\n");
		return;
	}

	size_t lastSize = m_allocatedSizes.back();
	m_allocatedSizes.pop_back();

	m_bufferOffset -= getAlignedSize(lastSize, 128);
	assert(m_bufferOffset % 128 == 0);
}

void Resources::releaseBuffers(uint bufferCount)
{
	for (uint i = 0; i < bufferCount; i++) {
		releaseBuffer();
	}
}

GPUResources::GPUResources(uint sizeX, uint sizeY, uint sizeZ, int cudaDevice)
{
	uint blockCount = 1;

	uint elemCount = sizeX * sizeY * sizeZ;
	uint elemCountPerBlock = elemCount / blockCount;

	// accumulate GPU buffer size
	size_t size = 0;

	// dpBuffer + dpScratch1 + dpScratch2
	size += getAlignedSize(elemCount * sizeof(float), 128);
	size += getAlignedSize(elemCount * sizeof(float), 128);
	size += getAlignedSize(elemCount * sizeof(float), 128);

	// dpSymbolStreams
	size += getAlignedSize(blockCount * elemCountPerBlock * sizeof(uint16_t), 128);

	// dpImage
	size += getAlignedSize(elemCount * sizeof(short), 128);

	// build Resources config
	Resources::Config config;
	config.blockCountMax = blockCount;
	config.elemCountPerBlockMax = elemCount * 2;
	config.bufferSize = size;
	config.log2HuffmanDistinctSymbolCountMax = LOG2_SYMBOL_COUNT_MAX;
	config.cudaDevice = cudaDevice;

	this->create(config);
}


bool GPUResources::create(const Config& config)
{
    m_config = config;

    assert(m_pCuCompInstance == nullptr);
    m_pCuCompInstance = cudaCompress::createInstance(m_config.cudaDevice, m_config.blockCountMax, m_config.elemCountPerBlockMax, m_config.codingBlockSize, m_config.log2HuffmanDistinctSymbolCountMax);
    if(!m_pCuCompInstance) {
        return false;
    }

    //TODO don't use cudaSafeCall, but manually check for out of memory?
    assert(m_dpBuffer == nullptr);
    cudaSafeCall(cudaMalloc(&m_dpBuffer, m_config.bufferSize));

    return true;
}

void GPUResources::destroy()
{
    cudaSafeCall(cudaFree(m_dpBuffer));
    m_dpBuffer = nullptr;

    cudaCompress::destroyInstance(m_pCuCompInstance);
    m_pCuCompInstance = nullptr;
}



CPUResources::CPUResources(uint sizeX, uint sizeY, uint sizeZ, int cudaDevice)
{
	uint blockCount = 1;

	uint elemCount = sizeX * sizeY * sizeZ;
	uint elemCountPerBlock = elemCount / blockCount;

	// accumulate GPU buffer size
	size_t size = 0;

	// dpBuffer + dpScratch1 + dpScratch2
	size += getAlignedSize(elemCount * sizeof(float), 128);
	size += getAlignedSize(elemCount * sizeof(float), 128);
	size += getAlignedSize(elemCount * sizeof(float), 128);

	// dpSymbolStreams
	size += getAlignedSize(blockCount * elemCountPerBlock * sizeof(uint16_t), 128);

	// dpImage
	size += getAlignedSize(elemCount * sizeof(short), 128);

	// build Resources config
	Resources::Config config;
	config.blockCountMax = blockCount;
	config.elemCountPerBlockMax = elemCount * 2;
	config.bufferSize = size;
	config.log2HuffmanDistinctSymbolCountMax = LOG2_SYMBOL_COUNT_MAX;
	config.cudaDevice = cudaDevice;

	this->create(config);
}


bool CPUResources::create(const Config& config)
{
	m_config = config;

	/*assert(m_pCuCompInstance == nullptr);
	m_pCuCompInstance = cudaCompress::createInstance(m_config.cudaDevice, m_config.blockCountMax, m_config.elemCountPerBlockMax, m_config.codingBlockSize, m_config.log2HuffmanDistinctSymbolCountMax);
	if (!m_pCuCompInstance) {
		return false;
	}*/
	m_pCuCompInstance = nullptr;

	assert(m_dpBuffer == nullptr);
	//cudaSafeCall(cudaMalloc(&m_dpBuffer, m_config.bufferSize));
	m_dpBuffer = (byte*)malloc(m_config.bufferSize);

	return true;
}

void CPUResources::destroy()
{
	free(m_dpBuffer);
	m_dpBuffer = nullptr;

	//cudaCompress::destroyInstance(m_pCuCompInstance);
	m_pCuCompInstance = nullptr;
}

