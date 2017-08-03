#ifndef __TUM3D_CUDACOMPRESS__ENCODE_CPU_H__
#define __TUM3D_CUDACOMPRESS__ENCODE_CPU_H__


#include <cudaCompress/global.h>

#include <string>
#include <vector>

#include <cudaCompress/BitStream.h>

#include <cudaCompress/EncodeCommon.h>


namespace cudaCompress {

enum ECoder {
    CODER_ARITHMETIC = 0,
    CODER_GOLOMBRICE,
    CODER_HUFFMAN,
    CODER_RBUC,
    CODER_RUNLENGTH,
    CODER_RUNLENGTH_GOLOMBRICE,
    CODER_RUNLENGTH_HUFFMAN,
    CODER_COUNT
};
std::string getCoderName(ECoder coder);


CUCOMP_DLL bool encodeCPU(ECoder coder, BitStream* pBitStream, /*const*/ std::vector<Symbol16>* pSymbolStream, uint codingBlockSize);
CUCOMP_DLL bool decodeCPU(ECoder coder, BitStreamReadOnly* pBitStream, std::vector<Symbol16>* pSymbolStream, uint symbolCount, uint codingBlockSize);

CUCOMP_DLL bool encodeCPU(ECoder coder, BitStream* ppBitStreams[], /*const*/ std::vector<Symbol16>* ppSymbolStreams[], uint blockCount, uint codingBlockSize);
CUCOMP_DLL bool decodeCPU(ECoder coder, BitStreamReadOnly* ppBitStreams[], std::vector<Symbol16>* ppSymbolStreams[], uint symbolCountPerBlock, uint blockCount, uint codingBlockSize);


CUCOMP_DLL bool encodeRLHuffCPU(BitStream* ppBitStreams[], /*const*/ std::vector<Symbol16>* ppSymbolStreams[], uint blockCount, uint codingBlockSize, bool zeroRunsOnly = true);
CUCOMP_DLL bool decodeRLHuffCPU(BitStreamReadOnly* ppBitStreams[], std::vector<Symbol16>* ppSymbolStreams[], uint symbolCountPerBlock, uint blockCount, uint codingBlockSize, bool zeroRunsOnly = true);

CUCOMP_DLL bool encodeHuffCPU(BitStream* ppBitStreams[], /*const*/ std::vector<Symbol16>* ppSymbolStreams[], uint blockCount, uint codingBlockSize);
CUCOMP_DLL bool decodeHuffCPU(BitStreamReadOnly* ppBitStreams[], std::vector<Symbol16>* ppSymbolStreams[], uint symbolCountPerBlock, uint blockCount, uint codingBlockSize);

bool encodeRLGolombRiceCPU(BitStream* ppBitStreams[], /*const*/ std::vector<Symbol16>* ppSymbolStreams[], uint blockCount, uint codingBlockSize, bool zeroRunsOnly = true);
bool decodeRLGolombRiceCPU(BitStreamReadOnly* ppBitStreams[], std::vector<Symbol16>* ppSymbolStreams[], uint symbolCountPerBlock, uint blockCount, uint codingBlockSize, bool zeroRunsOnly = true);

bool encodeGolombRiceCPU(BitStream* ppBitStreams[], /*const*/ std::vector<Symbol16>* ppSymbolStreams[], uint blockCount, uint codingBlockSize);
bool decodeGolombRiceCPU(BitStreamReadOnly* ppBitStreams[], std::vector<Symbol16>* ppSymbolStreams[], uint symbolCountPerBlock, uint blockCount, uint codingBlockSize);

bool encodeRBUCCPU(BitStream* ppBitStreams[], /*const*/ std::vector<Symbol16>* ppSymbolStreams[], uint blockCount);
bool decodeRBUCCPU(BitStreamReadOnly* ppBitStreams[], std::vector<Symbol16>* ppSymbolStreams[], uint symbolCountPerBlock, uint blockCount);

bool encodeArithmeticCPU(BitStream* ppBitStreams[], /*const*/ std::vector<Symbol16>* ppSymbolStreams[], uint blockCount, uint codingBlockSize);
bool decodeArithmeticCPU(BitStreamReadOnly* ppBitStreams[], std::vector<Symbol16>* ppSymbolStreams[], uint symbolCountPerBlock, uint blockCount, uint codingBlockSize);

bool encodeRunLengthCPU(BitStream* ppBitStreams[], /*const*/ std::vector<Symbol16>* ppSymbolStreams[], uint blockCount, bool zeroRunsOnly = true);
bool decodeRunLengthCPU(BitStreamReadOnly* ppBitStreams[], std::vector<Symbol16>* ppSymbolStreams[], uint symbolCountPerBlock, uint blockCount, bool zeroRunsOnly = true);

}


#endif
