#include <cudaCompress/util/Predictors.h>
#include <math.h>
#include <cudaCompress/cudaUtil.h>
#include "device_launch_parameters.h"


namespace cudaCompress {

	namespace util {

#define IND2D(arr,p,x,y) ( *(( int16_t*)((char*)arr + (y) * p) + (x)) )
#define NB_X(arr) IND2D(arr, pitch,   x,   y)
#define NB_A(arr) IND2D(arr, pitch, x-1,   y)
#define NB_B(arr) IND2D(arr, pitch,   x, y-1)
#define NB_C(arr) IND2D(arr, pitch, x-1, y-1)



		__global__ void _predictor0(const int16_t* in, int16_t* out, int pitch, int width, int height)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;
			int p = pitch / sizeof(int16_t);

			if (x < width && y < height) {
				// *((float*)((char*)out + y * pitch) + x) = *((float*)((char*)in1 + y * pitch) + x) - *((float*)((char*)in2 + y * pitch) + x);
				if (x>0) {
					out[p * y + x] = in[p * y + x];
				}
				else {
					out[p * y + x] = in[p * y + x];
				}
			}
		}
		__global__ void _unPredictor0(const int16_t* in, int16_t* out, int pitch, int width, int height)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x < width && y < height) {
				IND2D(out, pitch, x, y) = NB_X(in);
			}
		}
		__global__ void _unPredictor0(const int16_t* in, int16_t* out, int pitch, int width, int height, int it)
		{
			int y = blockIdx.x * blockDim.x + threadIdx.x;
			int x = it;

			if (x < width && y < height) {
				IND2D(out, pitch, x, y) = NB_X(in);
			}
		}

		__global__ void _predictor1(const int16_t* in, int16_t* out, int p, int width, int height)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
			int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x < width && y < height) {
				out[p*y + x] = in[p*y + x] - in[p*y + x - 1];
			}
		}
		__global__ void _unPredictor1(const  int16_t* in, int16_t* out, int p, int width, int height, int it)
		{
			int y = blockIdx.x * blockDim.x + threadIdx.x;
			int x = it + 1;

			if (x < width && y < height) {
				out[p*y + x] = in[p*y + x] + out[p*y + x - 1];
			}
		}
		__global__ void _predictor1_tiles(const int16_t* __restrict__ in,
			int16_t* __restrict__ out,
			int p, int width, int height, int tileSize) //x is tiled, y is not
		{
			int xi = blockIdx.x * blockDim.x + threadIdx.x + 1;
			int y  = blockIdx.y * blockDim.y + threadIdx.y;
			int xo = blockIdx.z * blockDim.z + threadIdx.z;

			//int offset = ty*p*height + tx*width;

			int x = xi + xo * tileSize;

			if (xi < tileSize && x < width && y < height) {
				/*if (xi == 0) {
					if (y > 0) {
						out[p*y + x] = in[p*y + x] - in[p*(y - 1) + x];
					}
				}
				else*/
					out[p*y + x] = in[p*y + x] - in[p*y + x - 1];
				//}
			}
		}

		__global__ void _unPredictor1_tiles(const int16_t* __restrict__ in,
			int16_t* __restrict__ out,
			int p, int width, int height, int tileSize, int it)
		{
			int y  = blockIdx.x * blockDim.x + threadIdx.x;
			int xi = it;
			int xo = blockIdx.y * blockDim.y + threadIdx.y;

			int x = xi + xo * tileSize;

			if (xi < tileSize && x < width && y < height) {
				out[p*y + x] = in[p*y + x] + out[p*y + x - 1];
			}
		}

		__global__ void _predictor2(const int16_t* in, int16_t* out, int p, int width, int height)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

			if (x < width && y < height) {
				out[p*y + x] = in[p*y + x] - in[p*(y - 1) + x];
			}
		}
		__global__ void _unPredictor2(const  int16_t* in, int16_t* out, int p, int width, int height, int it)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = it + 1;

			if (x < width && y < height) {
				out[p*y + x] = in[p*y + x] + out[p*(y - 1) + x];
			}
		}

		__global__ void _predictor4(const int16_t* in, int16_t* out, int p, int width, int height)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
			int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

			if (x < width && y < height) {
				out[p*y + x] = in[p*y + x] - (in[p*y + x - 1] + in[p*(y - 1) + x] - in[p*(y - 1) + x - 1]);
			}

		}

		__global__ void _unPredictor4(const int16_t* in, int16_t* out, int p, int width, int height, int it)
		{
			int th = blockIdx.x * blockDim.x + threadIdx.x;
			int x = max(it - th + 1, 1);
			int y = th + 1;

			if (x < width && y < height) {
				out[p*y + x] = in[p*y + x] + (out[p*y + x - 1] + out[p*(y - 1) + x] - out[p*(y - 1) + x - 1]);
			}
		}


		__global__ void _predictor7(const int16_t* in, int16_t* out, int p, int width, int height)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
			int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

			if (x < width && y < height) {
				out[p*y + x] = in[p*y + x] - ((in[p*y + x - 1] + in[p*(y - 1) + x]) >> 1);
			}

		}

		__global__ void _unPredictor7(const int16_t* in, int16_t* out, int p, int width, int height, int it)
		{
			int th = blockIdx.x * blockDim.x + threadIdx.x;
			int x = min(max(it - th + 1, 1), width-1);
			int y = min(th + 1, height-1);

			//if (x < width && y < height) {
				out[p*y + x] = in[p*y + x] + ((out[p*y + x - 1] + out[p*(y - 1) + x]) >> 1);
			//}
		}

		__global__ void _predictor7_tiles(const int16_t* __restrict__ in,
			                             int16_t* __restrict__ out,
										 int p, int width, int height, int tileSize)
		{
			int i = blockIdx.x * blockDim.x + threadIdx.x;
			int u = (i % tileSize);						// +1 to omit first column
			int v = (i / tileSize);						// +1 to omit first row

			int tx = blockIdx.y * blockDim.y + threadIdx.y;
			int ty = blockIdx.z * blockDim.z + threadIdx.z;

			//int offset = ty*p*height + tx*width;

			int x = u + tx * tileSize;
			int y = v + ty * tileSize;

			if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) {
				if (u == 0) {
					if (v > 0) {
						out[p*y + x] = in[p*y + x] - in[p*(y - 1) + x];
					}
					else {
						out[p*y + x] = in[p*y + x];
					}
				}
				else if (v == 0) {
					out[p*y + x] = in[p*y + x] - in[p*y + x - 1];
				}
				else {
					out[p*y + x] = in[p*y + x] - ((in[p*y + x - 1] + in[p*(y - 1) + x]) >> 1);
				}
			}			
		}

		__global__ void _unPredictor7_tiles(const int16_t* __restrict__ in,
										   int16_t* __restrict__ out,
										   int p, int width, int height, int tileSize, int it)
		{
			int th = blockIdx.x * blockDim.x + threadIdx.x;
			int u = it - th;
			int v = th;
			int tx = blockIdx.y * blockDim.y + threadIdx.y;
			int ty = blockIdx.z * blockDim.z + threadIdx.z;

			int x = u + tx * tileSize;
			int y = v + ty * tileSize;

			if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) {
				if (u == 0) {
					if (v > 0) {
						out[p*y + x] = in[p*y + x] + out[p*(y - 1) + x];
					}
					else {
						out[p*y + x] = in[p*y + x];
					}
				}
				else if (v == 0) {
					out[p*y + x] = in[p*y + x] + out[p*y + x - 1];
				}
				else {
					out[p*y + x] = in[p*y + x] + ((out[p*y + x - 1] + out[p*(y - 1) + x]) >> 1);
				}
			}
		}

		__global__ void _predictor7_tiles(const uint16_t* __restrict__ in,
			uint16_t* __restrict__ out,
			int p, int width, int height, int tileSize)
		{
			int i = blockIdx.x * blockDim.x + threadIdx.x;
			int u = (i % tileSize);						// +1 to omit first column
			int v = (i / tileSize);						// +1 to omit first row

			int tx = blockIdx.y * blockDim.y + threadIdx.y;
			int ty = blockIdx.z * blockDim.z + threadIdx.z;

			//int offset = ty*p*height + tx*width;

			int x = u + tx * tileSize;
			int y = v + ty * tileSize;

			if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) {
				if (u == 0) {
					if (v > 0) {
						out[p*y + x] = in[p*y + x] - in[p*(y - 1) + x];
					}
					else {
						out[p*y + x] = in[p*y + x];
					}
				}
				else if (v == 0) {
					out[p*y + x] = in[p*y + x] - in[p*y + x - 1];
				}
				else {
					out[p*y + x] = in[p*y + x] - (uint16_t)((int32_t)(in[p*y + x - 1] + (int32_t)in[p*(y - 1) + x]) >> 1);
				}
			}
		}

		__global__ void _unPredictor7_tiles(const uint16_t* __restrict__ in,
			uint16_t* __restrict__ out,
			int p, int width, int height, int tileSize, int it)
		{
			int th = blockIdx.x * blockDim.x + threadIdx.x;
			int u = it - th;
			int v = th;
			int tx = blockIdx.y * blockDim.y + threadIdx.y;
			int ty = blockIdx.z * blockDim.z + threadIdx.z;

			int x = u + tx * tileSize;
			int y = v + ty * tileSize;

			if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) {
				if (u == 0) {
					if (v > 0) {
						out[p*y + x] = in[p*y + x] + out[p*(y - 1) + x];
					}
					else {
						out[p*y + x] = in[p*y + x];
					}
				}
				else if (v == 0) {
					out[p*y + x] = in[p*y + x] + out[p*y + x - 1];
				}
				else {
					out[p*y + x] = in[p*y + x] + (uint16_t)((int32_t)(out[p*y + x - 1] + (int32_t)out[p*(y - 1) + x]) >> 1);
				}
			}
		}

		__global__ void _predictor7_tiles_nll(const int16_t* __restrict__ in, int16_t* __restrict__ buffer, int16_t* __restrict__ out, int p, int width, int height, int it, int16_t delta)
		{
			int th = blockIdx.x * blockDim.x + threadIdx.x;
			int x = min(max(it - th + 1, 1), width - 1);
			int y = min(th + 1, height - 1);
			int tx = blockIdx.y * blockDim.y + threadIdx.y;
			int ty = blockIdx.z * blockDim.z + threadIdx.z;

			int offset = ty*p*(height)+tx*(width);
			if (x < width && y < height) {
				int eps = in[offset + p*y + x] - ((buffer[offset + p*y + x - 1] + buffer[offset + p*(y - 1) + x]) >> 1);
				int Q = (eps < 0 ? -1 : 1) * ((abs(eps) + delta) / (2 * delta + 1));
				out[offset + p*y + x] = Q;
				buffer[offset + p*y + x] = fmaxf(0, (Q < 0 ? -1 : 1) * abs(Q) * (2 * delta + 1) +
					((buffer[offset + p*y + x - 1] + buffer[offset + p*(y - 1) + x]) >> 1));
			}
		}

		__global__ void _unPredictor7_tiles_nll(const int16_t* in, int16_t* out, int p, int width, int height, int it, int16_t delta)
		{
			int th = blockIdx.x * blockDim.x + threadIdx.x;
			int x = min(max(it - th + 1, 1), width - 1);
			int y = min(th + 1, height - 1);
			int tx = blockIdx.y * blockDim.y + threadIdx.y;
			int ty = blockIdx.z * blockDim.z + threadIdx.z;

			int offset = ty*p*(height)+tx*(width);
			if (x < width && y < height) {
				int Q = in[offset + p*y + x];
				out[offset + p*y + x] =  fmaxf(0, (Q < 0 ? -1 : 1) * abs(Q) * (2 * delta + 1) +
					((out[offset + p*y + x - 1] + out[offset + p*(y - 1) + x]) >> 1));
			}
		}

		__global__ void _predictor1_wnll(const float* in, float* buffer, int16_t* out, int p, int width, int height, int it)
		{
			int y = blockIdx.x * blockDim.x + threadIdx.x;
			int x = it + 1;
			int tx = blockIdx.y * blockDim.y + threadIdx.y;

			int offset = tx * width;
			if (x < width && y < height) {
				float eps = in[offset + p*y + x] - buffer[offset + p*y + x - 1];
				int Q = eps + 0.5 - signbit(eps);
				out[offset + p*y + x] = Q;
				buffer[offset + p*y + x] = fmaxf(0, Q + buffer[offset + p*y + x - 1]);
			}
		}

		__global__ void _unPredictor1_wnll(const int16_t* in, float* out, int p, int width, int height, int it)
		{
			int y = blockIdx.x * blockDim.x + threadIdx.x;
			int x = it + 1;
			int tx = blockIdx.y * blockDim.y + threadIdx.y;

			int offset = tx * width;
			if (x < width && y < height) {
				int Q = in[offset + p*y + x];
				out[offset + p*y + x] = fmaxf(0, Q + out[offset + p*y + x - 1]);
			}
		}

		template <typename T>
		__global__ void _predictor7_tiles_wnll(const float* __restrict__ in,
			float* __restrict__ buffer,
			T* __restrict__ out,
			int p, int width, int height, int tileSize, int it)
		{
			int th = blockIdx.x * blockDim.x + threadIdx.x;
			int u = it - th;
			int v = th;
			int tx = blockIdx.y * blockDim.y + threadIdx.y;
			int ty = blockIdx.z * blockDim.z + threadIdx.z;

			int x = u + tx * tileSize;
			int y = v + ty * tileSize;

			if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) {
				if (u == 0) {
					if (v > 0) {
						float eps = in[p*y + x] - buffer[p*(y - 1) + x];
						int Q = eps + (eps < 0 ? -0.5 : 0.5);
						out[p*y + x] = Q;
						buffer[p*y + x] = Q + buffer[p*(y - 1) + x];
					}
					else {
						float eps = in[p*y + x];
						int Q = eps + (eps < 0 ? -0.5 : 0.5);
						out[p*y + x] = Q;
						buffer[p*y + x] = Q;
					}
				}
				else if (v == 0) {
					float eps = in[p*y + x] - buffer[p*y + x - 1];
					int Q = eps + (eps < 0 ? -0.5 : 0.5);
					out[p*y + x] = Q;
					buffer[p*y + x] = Q + buffer[p*y + x - 1];
				}
				else {
					float eps = in[p*y + x] - ((buffer[p*y + x - 1] + buffer[p*(y - 1) + x]) / 2);
					int Q = eps + (eps < 0 ? -0.5 : 0.5);
					//int Q = eps + 0.5;
					out[p*y + x] = Q;
					buffer[p*y + x] = Q + ((buffer[p*y + x - 1] + buffer[p*(y - 1) + x]) / 2);
				}
			}
		}

		__global__ void _setup_rng(curandState *rngStates) {
			int id = threadIdx.x + blockIdx.x * blockDim.x;
			/* Each thread gets same seed, a different sequence number, no offset */
			curand_init(333, id, 0, &rngStates[id]);
		}

		template <typename T>
		__global__ void _predictor7_tiles_wnll2(const float* __restrict__ in,
			float* __restrict__ buffer,
			T* __restrict__ out,
			int p, int width, int height, int tileSize, int it, curandState *rngStates)
		{
			int th = blockIdx.x * blockDim.x + threadIdx.x;
			int u = it - th;
			int v = th;
			int tx = blockIdx.y * blockDim.y + threadIdx.y;
			int ty = blockIdx.z * blockDim.z + threadIdx.z;

			int x = u + tx * tileSize;
			int y = v + ty * tileSize;

			curandState localState = rngStates[th];
			float r = curand_uniform(&localState);
			rngStates[th] = localState;

			if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) {
				if (u == 0) {
					if (v > 0) {
						float eps = in[p*y + x] - buffer[p*(y - 1) + x];
						int Q = eps + (eps < 0 ? -r : r);
						out[p*y + x] = Q;
						buffer[p*y + x] = Q + buffer[p*(y - 1) + x];
					}
					else {
						float eps = in[p*y + x];
						int Q = eps + (eps < 0 ? -0.5 : 0.5);
						out[p*y + x] = Q;
						buffer[p*y + x] = Q;
					}
				}
				else if (v == 0) {
					float eps = in[p*y + x] - buffer[p*y + x - 1];
					int Q = eps + (eps < 0 ? -r : r);
					out[p*y + x] = Q;
					buffer[p*y + x] = Q + buffer[p*y + x - 1];
				}
				else {
					float eps = in[p*y + x] - ((buffer[p*y + x - 1] + buffer[p*(y - 1) + x]) / 2);
					int Q = eps + (eps < 0 ? -r : r);
					//int Q = eps + 0.5;
					out[p*y + x] = Q;
					buffer[p*y + x] = Q + ((buffer[p*y + x - 1] + buffer[p*(y - 1) + x]) / 2);
				}
			}
		}

		template <typename T>
		__global__ void _unPredictor7_tiles_wnll(const T* __restrict__ in,
												float* __restrict__ out,
												int p, int width, int height, int tileSize, int it)
		{
			int th = blockIdx.x * blockDim.x + threadIdx.x;
			int u = it - th;
			int v = th;
			int tx = blockIdx.y * blockDim.y + threadIdx.y;
			int ty = blockIdx.z * blockDim.z + threadIdx.z;

			int x = u + tx * tileSize;
			int y = v + ty * tileSize;

			if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) {
				//int Q = in[p*y + x];
				if (u == 0) {
					if (v > 0) {
						out[p*y + x] = in[p*y + x] + out[p*(y - 1) + x];
					}
					else {
						out[p*y + x] = in[p*y + x];
					}
				}
				else if (v == 0) {
					out[p*y + x] = in[p*y + x] + out[p*y + x - 1];
				}
				else {
					out[p*y + x] = in[p*y + x] + ((out[p*y + x - 1] + out[p*(y - 1) + x]) / 2);
				}
			}
		}

		__global__ void _predictor3D7(const int16_t* in, int16_t* out, int p, int sp, int width, int height, int depth)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
			int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
			int z = blockIdx.z * blockDim.z + threadIdx.z + 1;

			if (x < width && y < height && z < depth) {
				out[sp*z + p*y + x] = in[sp*z + p*y + x] - ((in[sp*z + p*y + x - 1] + in[sp*z + p*(y - 1) + x] + in[sp*(z-1) + p*y + x]) / 3);
			}

		}

		__global__ void _predictor8(const int16_t* in, int16_t* out, int p, int width, int height)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
			int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

			if (x < width && y < height) {
				out[p*y + x] = in[p*y + x] - ((3 * in[p*y + x - 1] + 3 * in[p*(y - 1) + x] - 2 * in[p*(y - 1) + x - 1]) >> 2);
			}

		}

		__global__ void _unPredictor8(const int16_t* in, int16_t* out, int p, int width, int height, int it)
		{
			int th = blockIdx.x * blockDim.x + threadIdx.x;
			int x = max(it - th + 1, 1);
			int y = th + 1;

			if (x < width && y < height) {
				out[p*y + x] = in[p*y + x] + ((3 * out[p*y + x - 1] + 3 * out[p*(y - 1) + x] - 2 * out[p*(y - 1) + x - 1]) >> 2);
			}
		}

		__global__ void _predictorMed_tiles(const int16_t* in, int16_t* out, int p, int width, int height)
		{
			int i = blockIdx.x * blockDim.x + threadIdx.x;
			int x = (i % width) + 1;
			int y = (i / width) + 1;

			int tx = blockIdx.y * blockDim.y + threadIdx.y;
			int ty = blockIdx.z * blockDim.z + threadIdx.z;

			int offset = ty*p*height + tx*width;			

			if (x < width && y < height) {
				float A = in[offset + p*y + x - 1];
				float B = in[offset + p*(y - 1) + x];
				float C = in[offset + p*(y - 1) + x - 1];
				int prediction = (C >= max(A, B) ? min(A, B) : (C <= min(A, B) ? max(A, B) : A + B - C));
				out[offset + p*y + x] = in[offset + p*y + x] - prediction;
			}

		}

		__global__ void _unPredictorMed_tiles(const int16_t* in, int16_t* out, int p, int width, int height, int it)
		{
			int th = blockIdx.x * blockDim.x + threadIdx.x;
			int x = min(max(it - th + 1, 1), width - 1);
			int y = min(th + 1, height - 1);
			int tx = blockIdx.y * blockDim.y + threadIdx.y;
			int ty = blockIdx.z * blockDim.z + threadIdx.z;

			int offset = ty*p*(height)+tx*(width);
			if (x < width && y < height) {
				float A = out[offset + p*y + x - 1];
				float B = out[offset + p*(y - 1) + x];
				float C = out[offset + p*(y - 1) + x - 1];
				int prediction = (C >= max(A, B) ? min(A, B) : (C <= min(A, B) ? max(A, B) : A + B - C));
				out[offset + p*y + x] = in[offset + p*y + x] + prediction;
			}
		}


		__global__ void _quantize(const float* __restrict__ in, int16_t* __restrict__  out, int p, int width, int height)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;
			if (x < width && y < height) {
				int eps = in[p*y + x] + 0.5 - signbit(in[p*y + x]);
				//out[p*y + x] = eps + (eps < 0 ? -1 : 1) * 0.5;
				out[p*y + x] = eps;
			}
		}

		__global__ void _unQuantize(const int16_t* __restrict__ in, float* __restrict__ out, int p, int width, int height)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;
			if (x < width && y < height) {
				out[p*y + x] = in[p*y + x];
			}
		}

		/*__global__ void quantize(const float* in, int16_t* out, int p, int width, int height, float delta)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;
			int eps = in[p*y + x];
			if (x < width && y < height) {
				out[p*y + x] = (eps < 0 ? -1 : 1) * ((abs(eps) + delta) / (2 * delta));
			}
		}

		__global__ void unQuantize(const int16_t* in, float* out, int p, int width, int height, float delta)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;
			int Q = in[p*y + x];
			if (x < width && y < height) {
				out[p*y + x] = (Q < 0 ? -1 : 1) * abs(Q) * (2 * delta);
			}
		}*/

		/*__global__ void quantize(const float* in, float* out, int p, int width, int height, float delta)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;
			int eps = in[p*y + x];
			if (x < width && y < height) {
				out[p*y + x] = (eps < 0 ? -1 : 1) * ((abs(eps) + delta) / (2 * delta));
			}
		}

		__global__ void unQuantize(const float* in, float* out, int p, int width, int height, float delta)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;
			int Q = in[p*y + x];
			if (x < width && y < height) {
				out[p*y + x] = (Q < 0 ? -1 : 1) * abs(Q) * (2 * delta);
			}
		}*/

		__global__ void _sqrt(float* in, float* out, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = sqrt(in[x]);
		}

		__global__ void _sqr(float* in, float* out, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = in[x] * in[x];
		}

		__global__ void _sqrt(const int16_t* __restrict__ in, float* __restrict__ out, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = sqrt((float)in[x]);
		}

		__global__ void _sqr(const float* __restrict__ in, int16_t* __restrict__ out, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = in[x] * in[x] + 0.5;
		}

		__global__ void _s2f(int16_t* in, float* out, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = in[x];
		}

		__global__ void _f2s(float* in, int16_t* out, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = in[x] + (in[x] < 0 ? -1 : 1) * 0.5;
		}

		__global__ void _u2f(uint16_t* in, float* out, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = in[x];
		}

		__global__ void _f2u(float* in, uint16_t* out, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = in[x] + (in[x] < 0 ? -1 : 1) * 0.5;
		}

		__global__ void _u8tou16(uint8_t* in, uint16_t* out, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = in[x];
		}

		__global__ void _u16tou8(uint16_t* in, uint8_t* out, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = min(max(in[x],0),255);
		}

		__global__ void _multiply(float* in, float* out, float factor, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = in[x] * factor;
		}

		__global__ void _multiply(int16_t* in, float* out, float factor, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = in[x] * factor;
		}

		__global__ void _offset(float* in, float* out, float level, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = in[x] + level;
			if (level < 0)
				out[x] = out[x] * (out[x] > 0);
		}

		__global__ void _offsetAbs(float* in, float* out, uint16_t* signs, float level, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = in[x] + level;
			signs[x] = signbit(out[x]) ? 1 : 0;
			out[x] = abs(out[x]);
		}

		__global__ void _applySignOffset(float* in, float* out, uint16_t* signs, float level, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = in[x] * (signs[x] ? -1 : 1);
			out[x] = out[x] + level;
		}

		__global__ void _offsetSeparate(float* in, float* out, float* negatives, float level, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = in[x] + level;
			if (out[x] < 0) {
				negatives[x] = -out[x];
				out[x] = 0;
			}
		}

		__global__ void _mergeOffset(float* in, float* out, float* negatives, float level, int num) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = in[x] - negatives[x] + level;
		}
		
		__global__ void _vst(float* in, float* out, int num, float offset, float conversion, float sigma) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			//float z = in[x];
			//z = (z - offset) / conversion;
			//z = 2 * sqrtf(z + sigma*sigma) - 2 * sigma;
			//out[x] = z;
			out[x] = 2 * sqrtf((fmaxf(in[x] - offset, 0)) / conversion + sigma*sigma) - 2 * sigma;
		}

		__global__ void _invVst(float* in, float* out, int num, float offset, float conversion, float sigma) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			float D = in[x];
			D = D + 2 * sigma; // remove offset
			if (D >= 2 * sigma) {
				out[x] = ((D*D/4)-sigma*sigma)*conversion + offset;
			}
			else {
				out[x] = offset ;
			}
		}

		__global__ void _anscombe(float* in, float* out, int num, float sigma) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			out[x] = 2 * sqrtf(in[x] + 3. / 8. + sigma*sigma) - sqrtf(3. / 2.); // -sqrt(3./2.) offset to move 0 to 0 and assure 0 quantization error
																				//out[x] = 2 * sqrtf(in[x] + 3. / 8. + sigma*sigma);
		}

		__global__ void _invAnscombe(float* in, float* out, int num, float sigma) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			float D = in[x];
			D = D + sqrtf(3. / 2.); // remove offset
			if (D > 2 * sqrtf(3. / 8. + sigma * sigma)) {
				out[x] = 0.25 * D * D + 0.25 * sqrtf(1.5) * powf(D, -1) - 1.375 * powf(D, -2) + 0.625 * sqrtf(1.5) * powf(D, -3) - 0.125 - sigma * sigma;
			}
			else {
				out[x] = 0;
			}
		}

		void predictor1(const int16_t* in, int16_t* out, int pitch, int width, int height)
		{
			dim3 dimBlock(32, 32);
			dim3 dimGrid((width - 1 + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
			int p = pitch / sizeof(int16_t);

			_predictor1 << <dimGrid, dimBlock >> >(in, out, p, width, height);
			return;
		}

		void unPredictor1(const int16_t* in, int16_t* out, int pitch, int width, int height)
		{
			int dimBlock = 512;
			int dimGrid = (height + dimBlock - 1) / dimBlock;
			int p = pitch / sizeof(int16_t);

			for (int i = 0; i < width - 1; i++)
				_unPredictor1 << <dimGrid, dimBlock >> >(in, out, p, width, height, i);

			return;
		}

		void predictor1_tiles(const int16_t* in, int16_t* out, int pitch, int width, int height, int tileSize)
		{
			//const int num_tiles = 4;
			dim3 dimBlock(32, 32, 1);

			int numTilesX = (width + tileSize - 1) / tileSize;
			//int tilesY = (height + tileSize - 1) / tileSize;

			dim3 dimGrid((tileSize + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, numTilesX);
			int p = pitch / sizeof(int16_t);

			_predictor1_tiles << <dimGrid, dimBlock >> >(in, out, p, width, height, tileSize);

			return;
		}

		void unPredictor1_tiles(const int16_t* in, int16_t* out, int pitch, int width, int height, int tileSize)
		{
			dim3 dimBlock(512, 1, 1);

			int numTilesX = (width + tileSize - 1) / tileSize;

			dim3 dimGrid(((height - 1) + dimBlock.x - 1) / dimBlock.x, numTilesX, 1);
			int p = pitch / sizeof(int16_t);

			for (int it = 1; it < tileSize; it++)
				_unPredictor1_tiles << <dimGrid, dimBlock >> >(in, out, p, width, height, tileSize, it);

			return;

		}

		void predictor2(const int16_t* in, int16_t* out, int pitch, int width, int height)
		{
			dim3 dimBlock(32, 32);
			dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height - 1 + dimBlock.y - 1) / dimBlock.y);
			int p = pitch / sizeof(int16_t);

			_predictor2 << <dimGrid, dimBlock >> >(in, out, p, width, height);
			return;
		}

		void unPredictor2(const int16_t* in, int16_t* out, int pitch, int width, int height)
		{
			int dimBlock = 512;
			int dimGrid = (width + dimBlock - 1) / dimBlock;
			int p = pitch / sizeof(int16_t);

			for (int i = 0; i < height - 1; i++)
				_unPredictor2 << <dimGrid, dimBlock >> >(in, out, p, width, height, i);

			return;
		}

		void predictor2_tiles(const int16_t* in, int16_t* out, int pitch, int width, int height, int tileSize)
		{
			//const int num_tiles = 4;
			dim3 dimBlock(32, 32, 1);

			int numTilesX = (width + tileSize - 1) / tileSize;
			//int tilesY = (height + tileSize - 1) / tileSize;

			dim3 dimGrid((tileSize + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, numTilesX);
			int p = pitch / sizeof(int16_t);

			_predictor1_tiles << <dimGrid, dimBlock >> >(in, out, p, width, height, tileSize);

			return;
		}

		void unPredictor2_tiles(const int16_t* in, int16_t* out, int pitch, int width, int height, int tileSize)
		{
			dim3 dimBlock(512, 1, 1);

			int numTilesX = (width + tileSize - 1) / tileSize;

			dim3 dimGrid(((height - 1) + dimBlock.x - 1) / dimBlock.x, numTilesX, 1);
			int p = pitch / sizeof(int16_t);

			for (int it = 1; it < tileSize; it++)
				_unPredictor1_tiles << <dimGrid, dimBlock >> >(in, out, p, width, height, tileSize, it);

			return;

		}
		
		void predictor4(const int16_t* in, int16_t* out, int pitch, int width, int height)
		{
			dim3 dimBlock(32, 32);
			dim3 dimGrid((width - 1 + dimBlock.x - 1) / dimBlock.x, (height - 1 + dimBlock.y - 1) / dimBlock.y);
			int p = pitch / sizeof(int16_t);

			_predictor4 << <dimGrid, dimBlock >> >(in, out, p, width, height);
			return;
		}

		void unPredictor4(const int16_t* in, int16_t* out, int pitch, int width, int height)
		{
			int dimBlock = 512;
			int dimGrid = (height - 1 + dimBlock - 1) / dimBlock;
			int p = pitch / sizeof(int16_t);

			for (int i = 0; i < width + height - 1; i++)
				_unPredictor4 << <dimGrid, dimBlock >> >(in, out, p, width, height, i);

			return;

		}

		void predictor7(const int16_t* in, int16_t* out, int pitch, int width, int height)
		{
			dim3 dimBlock(32, 32);
			dim3 dimGrid((width - 1 + dimBlock.x - 1) / dimBlock.x, (height - 1 + dimBlock.y - 1) / dimBlock.y);
			int p = pitch / sizeof(int16_t);

			_predictor7 << <dimGrid, dimBlock >> >(in, out, p, width, height);
			return;
		}

		void unPredictor7(const int16_t* in, int16_t* out, int pitch, int width, int height)
		{
			int dimBlock = 512;
			int dimGrid = (height - 1 + dimBlock - 1) / dimBlock;
			int p = pitch / sizeof(int16_t);

			for (int i = 0; i < width + height - 1; i++)
				_unPredictor7 << <dimGrid, dimBlock >> >(in, out, p, width, height, i);

			return;

		}

		void predictor7_tiles(const int16_t* in, int16_t* out, int pitch, int width, int height, int tileSize)
		{
			//const int num_tiles = 4;
			dim3 dimBlock(1024, 1, 1);

			//int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
			//int tile_height = (height + num_tiles - 1) / num_tiles;

			int tilesX = (width + tileSize - 1) / tileSize;
			int tilesY = (height + tileSize - 1) / tileSize;

			dim3 dimGrid(( (tileSize * tileSize) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
			int p = pitch / sizeof(int16_t);

			_predictor7_tiles <<<dimGrid, dimBlock >>>(in, out, p, width, height, tileSize);

			return;
		}


		void unPredictor7_tiles(const int16_t* in, int16_t* out, int pitch, int width, int height, int tileSize)
		{
			//const int num_tiles = 4;
			dim3 dimBlock(tileSize, 1, 1);

			//int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
			//int tile_height = (height + num_tiles - 1) / num_tiles;

			int tilesX = (width + tileSize - 1) / tileSize;
			int tilesY = (height + tileSize - 1) / tileSize;

			dim3 dimGrid(( (tileSize - 1) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
			int p = pitch / sizeof(int16_t);
			//int offset = 0;

			for (int it = 0; it < tileSize + tileSize - 1; it++) {
				_unPredictor7_tiles << <dimGrid, dimBlock >> >(in, out, p, width, height, tileSize, it);
				cudaCheckMsg("unpredictor failed");
			}
			return;

		}

		void predictor7_tiles_nll(const int16_t* in, int16_t* buffer, int16_t* out, int pitch, int width, int height, int num_tiles, int16_t delta = 0)
		{			
			//const int num_tiles = 4;
			dim3 dimBlock(512, 1, 1);

			int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
			int tile_height = (height + num_tiles - 1) / num_tiles;

			dim3 dimGrid((tile_height - 1 + dimBlock.x - 1) / dimBlock.x, num_tiles, num_tiles);
			int p = pitch / sizeof(int16_t);
			//int offset = 0;

			for (int it = 0; it < tile_width + tile_height - 1; it++)
				_predictor7_tiles_nll << <dimGrid, dimBlock >> >(in, buffer, out, p, tile_width, tile_height, it, delta);
			return;
		}

		void unPredictor7_tiles_nll(const int16_t* in, int16_t* out, int pitch, int width, int height, int num_tiles, int16_t delta = 0)
		{
			//const int num_tiles = 4;
			dim3 dimBlock(512, 1, 1);

			int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
			int tile_height = (height + num_tiles - 1) / num_tiles;

			dim3 dimGrid((tile_height - 1 + dimBlock.x - 1) / dimBlock.x, num_tiles, num_tiles);
			int p = pitch / sizeof(int16_t);
			//int offset = 0;			


			for (int it = 0; it < tile_width + tile_height - 1; it++)
				_unPredictor7_tiles_nll << <dimGrid, dimBlock >> >(in, out, p, tile_width, tile_height, it, delta);
			return;

		}

		void predictor1_tiles_wnll(const float* in, float* buffer, int16_t* out, int pitch, int width, int height, int num_tiles)
		{
			//quantize 
			cudaCompress::util::quantize(in, out, pitch, width, height);
			// unquantize, put it in buffer
			cudaCompress::util::unQuantize(out, buffer, pitch, width, height);

			dim3 dimBlock(512, 1);

			int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up

			dim3 dimGrid((height + dimBlock.x - 1) / dimBlock.x, num_tiles);
			

			for (int it = 0; it < width-1; it++)
				_predictor1_wnll << <dimGrid, dimBlock >> >(in, buffer, out, pitch, tile_width, height, it);
			return;
		}

		void unPredictor1_tiles_wnll(const int16_t* in, float* out, int pitch, int width, int height, int num_tiles)
		{
			dim3 dimBlock(512, 1);

			int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up

			dim3 dimGrid((height + dimBlock.x - 1) / dimBlock.x, num_tiles);

			for (int it = 0; it < width-1; it++)
				_unPredictor1_wnll << <dimGrid, dimBlock >> >(in, out, pitch, tile_width, height, it);
			return;

		}

		void predictor7_tiles_wnll(const float* in, float* buffer, int16_t* out, int pitch, int width, int height, int tileSize)
		{
			/*//quantize 
			cudaCompress::util::quantize(in, out, pitch, width, height);
			// unquantize, put it in buffer
			cudaCompress::util::unQuantize(out, buffer, pitch, width, height);
			*/
			//const int num_tiles = 4;
			dim3 dimBlock(512, 1, 1);

			//int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
			//int tile_height = (height + num_tiles - 1) / num_tiles;

			int tilesX = (width + tileSize - 1) / tileSize;
			int tilesY = (height + tileSize - 1) / tileSize;

			dim3 dimGrid((tileSize - 1 + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
			int p = pitch;
			//int offset = 0;

			for (int it = 0; it < tileSize + tileSize - 1; it++)
				_predictor7_tiles_wnll << <dimGrid, dimBlock >> >(in, buffer, out, p, width, height, tileSize, it);

			cudaCheckMsg("predictor7_tiles_wnll execution failed");
			return;
		}

		void predictor7_tiles_wnll2(const float* in, float* buffer, int16_t* out, int pitch, int width, int height, int tileSize)
		{
			/*//quantize 
			cudaCompress::util::quantize(in, out, pitch, width, height);
			// unquantize, put it in buffer
			cudaCompress::util::unQuantize(out, buffer, pitch, width, height);
			*/

			//const int num_tiles = 4;
			dim3 dimBlock(512, 1, 1);

			//int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
			//int tile_height = (height + num_tiles - 1) / num_tiles;

			int tilesX = (width + tileSize - 1) / tileSize;
			int tilesY = (height + tileSize - 1) / tileSize;

			dim3 dimGrid((tileSize - 1 + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
			int p = pitch;
			//int offset = 0;

			// init random generator for dithering
			curandState *rngStates;
			cudaMalloc((void **)&rngStates, 512 * sizeof(curandState));
			_setup_rng << <dimGrid, dimBlock >> > (rngStates);

			for (int it = 0; it < tileSize + tileSize - 1; it++)
				_predictor7_tiles_wnll2 << <dimGrid, dimBlock >> >(in, buffer, out, p, width, height, tileSize, it, rngStates);

			cudaFree(rngStates);

			cudaCheckMsg("predictor7_tiles_wnll2 execution failed");
			return;
		}

		void unPredictor7_tiles_wnll(const int16_t* in, float* out, int pitch, int width, int height, int tileSize)
		{
			//const int num_tiles = 4;
			dim3 dimBlock(512, 1, 1);

			//int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
			//int tile_height = (height + num_tiles - 1) / num_tiles;

			int tilesX = (width + tileSize - 1) / tileSize;
			int tilesY = (height + tileSize - 1) / tileSize;

			dim3 dimGrid((tileSize - 1 + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
			int p = pitch / sizeof(int16_t);

			for (int it = 0; it < tileSize + tileSize - 1; it++)
				_unPredictor7_tiles_wnll << <dimGrid, dimBlock >> >(in, out, p, width, height, tileSize, it);
			return;

		}

		void predictor3D7(const int16_t* in, int16_t* out, int pitch, int slicePitch, int width, int height, int depth)
		{
			dim3 dimBlock(8, 8, 8);
			dim3 dimGrid((width - 1 + dimBlock.x - 1) / dimBlock.x, (height - 1 + dimBlock.y - 1) / dimBlock.y, (depth - 1 + dimBlock.z - 1) / dimBlock.z);
			
			_predictor3D7 << <dimGrid, dimBlock >> >(in, out, pitch / sizeof(int16_t), slicePitch / sizeof(int16_t), width, height, depth);
			return;
		}


		void predictor8(const int16_t* in, int16_t* out, int pitch, int width, int height)
		{
			dim3 dimBlock(32, 32);
			dim3 dimGrid((width - 1 + dimBlock.x - 1) / dimBlock.x, (height - 1 + dimBlock.y - 1) / dimBlock.y);
			int p = pitch / sizeof(int16_t);

			_predictor8 << <dimGrid, dimBlock >> >(in, out, p, width, height);
			return;
		}

		void unPredictor8(const int16_t* in, int16_t* out, int pitch, int width, int height)
		{
			int dimBlock = 512;
			int dimGrid = (height - 1 + dimBlock - 1) / dimBlock;
			int p = pitch / sizeof(int16_t);

			for (int i = 0; i < width + height - 1; i++)
				_unPredictor8 << <dimGrid, dimBlock >> >(in, out, p, width, height, i);

			return;

		}

		void predictorMed_tiles(const int16_t* in, int16_t* out, int pitch, int width, int height)
		{
			const int num_tiles = 4;
			dim3 dimBlock(512, 1, 1);

			int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
			int tile_height = (height + num_tiles - 1) / num_tiles;

			dim3 dimGrid(((tile_width)* (tile_height)+dimBlock.x - 1) / dimBlock.x, num_tiles, num_tiles);
			int p = pitch / sizeof(int16_t);

			_predictorMed_tiles << <dimGrid, dimBlock >> >(in, out, p, tile_width, tile_height);

			return;
		}

		void unPredictorMed_tiles(const int16_t* in, int16_t* out, int pitch, int width, int height)
		{
			const int num_tiles = 4;
			dim3 dimBlock(512, 1, 1);

			int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
			int tile_height = (height + num_tiles - 1) / num_tiles;

			dim3 dimGrid((tile_height - 1 + dimBlock.x - 1) / dimBlock.x, num_tiles, num_tiles);
			int p = pitch / sizeof(int16_t);
			//int offset = 0;			


			for (int it = 0; it < tile_width + tile_height - 1; it++)
				_unPredictorMed_tiles << <dimGrid, dimBlock >> >(in, out, p, tile_width, tile_height, it);

			return;

		}

		void quantize(const float* in, int16_t* out, int pitch, int width, int height)
		{
			dim3 dimBlock(32, 32);
			dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
			//int p = pitch / sizeof(int16_t);

			_quantize << <dimGrid, dimBlock >> >(in, out, pitch, width, height);
			return;
		}

		void unQuantize(const int16_t* in, float* out, int pitch, int width, int height)
		{
			dim3 dimBlock(32, 32);
			dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
			//int p = pitch / sizeof(int16_t);

			_unQuantize << <dimGrid, dimBlock >> >(in, out, pitch, width, height);
			return;
		}

		/*void quantize(const float* in, float* out, int pitch, int width, int height, float delta)
		{
			dim3 dimBlock(32, 32);
			dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
			//int p = pitch / sizeof(int16_t);

			quantize << <dimGrid, dimBlock >> >(in, out, pitch, width, height, delta);
			return;
		}

		void unQuantize(const float* in, float* out, int pitch, int width, int height, float delta)
		{
			dim3 dimBlock(32, 32);
			dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
			//int p = pitch / sizeof(int16_t);

			unQuantize << <dimGrid, dimBlock >> >(in, out, pitch, width, height, delta);
			return;
		}*/

		void Anscombe(float* in, float* out, int num, float sigma)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_anscombe << <dimGrid, dimBlock >> >(in, out, num, sigma);
			return;
		}

		void invAnscombe(float* in, float* out, int num, float sigma)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_invAnscombe << <dimGrid, dimBlock >> >(in, out, num, sigma);
			return;
		}

		void vst(float* in, float* out, int num, float offset, float conversion, float sigma)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_vst << <dimGrid, dimBlock >> >(in, out, num, offset, conversion, sigma);
			return;
		}

		void invVst(float* in, float* out, int num, float offset, float conversion, float sigma)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_invVst << <dimGrid, dimBlock >> >(in, out, num, offset, conversion, sigma);
			return;
		}

		void sqrtArray(float* in, float* out, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_sqrt << <dimGrid, dimBlock >> >(in, out, num);
			return;
		}

		void sqrArray(float* in, float* out, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_sqr << <dimGrid, dimBlock >> >(in, out, num);
			return;
		}

		void sqrtArray(int16_t* in, float* out, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_sqrt << <dimGrid, dimBlock >> >(in, out, num);
			return;
		}

		void sqrArray(float* in, int16_t* out, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_sqr << <dimGrid, dimBlock >> >(in, out, num);
			return;
		}

		void s2f(int16_t* in, float* out, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_s2f << <dimGrid, dimBlock >> >(in, out, num);
			return;
		}

		void f2s(float* in, int16_t* out, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_f2s << <dimGrid, dimBlock >> >(in, out, num);
			return;
		}

		void u2f(uint16_t* in, float* out, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_u2f << <dimGrid, dimBlock >> >(in, out, num);
			return;
		}

		void f2u(float* in, uint16_t* out, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_f2u << <dimGrid, dimBlock >> >(in, out, num);
			return;
		}

		void u8tou16(uint8_t* in, uint16_t* out, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_u8tou16 << <dimGrid, dimBlock >> >(in, out, num);
			return;
		}

		void u16tou8(uint16_t* in, uint8_t* out, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_u16tou8 << <dimGrid, dimBlock >> >(in, out, num);
			return;
		}

		void multiply(float* in, float* out, float factor, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_multiply << <dimGrid, dimBlock >> >(in, out, factor, num);
			return;
		}

		void multiply(int16_t* in, float* out, float factor, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_multiply << <dimGrid, dimBlock >> >(in, out, factor, num);
			return;
		}

		void offset(float* in, float* out, float level, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_offset << <dimGrid, dimBlock >> >(in, out, level, num);
			return;
		}

		void offsetAbs(float* in, float* out, uint16_t* signs, float level, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_offsetAbs << <dimGrid, dimBlock >> >(in, out, signs, level, num);
			return;
		}

		void applySignOffset(float* in, float* out, uint16_t* signs, float level, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_applySignOffset << <dimGrid, dimBlock >> >(in, out, signs, level, num);
			return;
		}

		void offsetSeparate(float* in, float* out, float* negatives, float level, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_offsetSeparate << <dimGrid, dimBlock >> >(in, out, negatives, level, num);
			return;
		}

		void mergeOffset(float* in, float* out, float* negatives, float level, int num)
		{
			int dimBlock = 1024;
			int dimGrid = (num + dimBlock - 1) / dimBlock;

			_mergeOffset << <dimGrid, dimBlock >> >(in, out, negatives, level, num);
			return;
		}



	}

}
